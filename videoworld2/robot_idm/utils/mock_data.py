from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .runtime import ensure_dir, save_json


@dataclass
class MockEpisodeConfig:
    image_size: int = 64
    horizon: int = 40
    dt: float = 0.18
    action_scale: float = 0.12


def render_mock_frame(position: torch.Tensor, target: torch.Tensor, image_size: int) -> torch.Tensor:
    ys = torch.linspace(0.0, 1.0, image_size)
    xs = torch.linspace(0.0, 1.0, image_size)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    frame = torch.zeros(3, image_size, image_size)

    agent_sigma = 0.015
    target_sigma = 0.02
    agent = torch.exp(-((grid_x - position[0]) ** 2 + (grid_y - position[1]) ** 2) / agent_sigma)
    goal = torch.exp(-((grid_x - target[0]) ** 2 + (grid_y - target[1]) ** 2) / target_sigma)

    frame[2] = agent
    frame[1] = goal
    frame[0] = 0.35 * goal
    return frame.clamp(0.0, 1.0)


def _rotate90(vector: torch.Tensor, direction: float) -> torch.Tensor:
    return direction * torch.tensor([-vector[1], vector[0]], dtype=vector.dtype)


def rollout_oracle_episode(seed: int, cfg: MockEpisodeConfig) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)
    target = torch.rand(2, generator=generator) * 0.6 + 0.2
    position = torch.rand(2, generator=generator) * 0.5 + 0.15
    velocity = torch.zeros(2)
    swirl = -1.0 if torch.rand((), generator=generator) < 0.5 else 1.0
    embodiment_id = int(torch.randint(0, 2, (), generator=generator))
    embodiment_gain = 0.85 if embodiment_id == 0 else 1.15

    frames: list[torch.Tensor] = []
    proprio: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []

    for _ in range(cfg.horizon):
        delta = target - position
        distance = delta.norm().clamp_min(1e-5)
        radial = delta / distance
        tangent = _rotate90(radial, swirl)
        orbit_weight = torch.sigmoid((distance - 0.16) * 12.0)
        desired = 0.9 * radial + 0.45 * orbit_weight * tangent
        action = 0.55 * velocity + 0.45 * desired * cfg.action_scale * embodiment_gain
        action = action + 0.01 * torch.randn(2, generator=generator)
        action = action.clamp(-cfg.action_scale, cfg.action_scale)

        frames.append(render_mock_frame(position, target, cfg.image_size))
        proprio.append(torch.cat([position, velocity]))
        actions.append(action)

        velocity = action
        position = (position + cfg.dt * velocity).clamp(0.05, 0.95)

    frames.append(render_mock_frame(position, target, cfg.image_size))
    proprio.append(torch.cat([position, velocity]))

    rgb_static = torch.stack(frames)
    proprio_tensor = torch.stack(proprio)
    action_tensor = torch.stack(actions)

    return {
        "rgb_static": rgb_static,
        "proprio": proprio_tensor,
        "action": action_tensor,
        "lang": "",
        "task_id": 0,
        "embodiment_id": embodiment_id,
        "episode_id": f"mock_{seed:05d}",
        "meta": {
            "target": target,
            "swirl": swirl,
            "embodiment_gain": embodiment_gain,
            "dt": cfg.dt,
            "action_scale": cfg.action_scale,
            "image_size": cfg.image_size,
        },
    }


def generate_mock_dataset(root: str | Path, train_episodes: int = 64, val_episodes: int = 16) -> dict[str, str]:
    root_path = ensure_dir(root)
    train_dir = ensure_dir(root_path / "train")
    val_dir = ensure_dir(root_path / "val")

    manifests: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    cfg = MockEpisodeConfig()

    for split, count, split_dir, offset in (
        ("train", train_episodes, train_dir, 0),
        ("val", val_episodes, val_dir, 10_000),
    ):
        for episode_idx in range(count):
            episode = rollout_oracle_episode(seed=offset + episode_idx, cfg=cfg)
            episode_path = split_dir / f"{episode['episode_id']}.pt"
            torch.save(episode, episode_path)
            manifests[split].append({"path": str(episode_path), "episode_id": episode["episode_id"]})

    manifest_paths = {
        split: str(root_path / f"{split}_manifest.json")
        for split in ("train", "val")
    }
    for split, manifest_path in manifest_paths.items():
        save_json({"episodes": manifests[split]}, manifest_path)

    return manifest_paths


def oracle_action(position: torch.Tensor, velocity: torch.Tensor, target: torch.Tensor, swirl: float, action_scale: float, embodiment_gain: float) -> torch.Tensor:
    delta = target - position
    distance = delta.norm().clamp_min(1e-5)
    radial = delta / distance
    tangent = _rotate90(radial, swirl)
    orbit_weight = torch.sigmoid((distance - 0.16) * 12.0)
    desired = 0.9 * radial + 0.45 * orbit_weight * tangent
    action = 0.55 * velocity + 0.45 * desired * action_scale * embodiment_gain
    return action.clamp(-action_scale, action_scale)


def rollout_mock_env(
    initial_position: torch.Tensor,
    initial_velocity: torch.Tensor,
    target: torch.Tensor,
    swirl: float,
    embodiment_gain: float,
    actions: torch.Tensor,
    dt: float = 0.18,
) -> dict[str, torch.Tensor]:
    position = initial_position.clone()
    velocity = initial_velocity.clone()
    positions = [position.clone()]
    for action in actions:
        velocity = action
        position = (position + dt * velocity).clamp(0.05, 0.95)
        positions.append(position.clone())
    return {"final_position": position, "positions": torch.stack(positions)}
