from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from videoworld2.robot_idm.data.robot_window_dataset import _normalise_manifest_path, load_episode
from videoworld2.robot_idm.train.common import prepare_mock_data_if_needed, resolve_config_path
from videoworld2.robot_idm.utils.config import load_config
from videoworld2.robot_idm.utils.metrics import rollout_success
from videoworld2.robot_idm.utils.runtime import load_json, save_json


def evaluate_oracle_replay(cfg: dict, split: str, max_episodes: int | None = None) -> dict[str, object]:
    if cfg["data"].get("dataset_type") != "mock":
        raise ValueError("Oracle replay is currently implemented for dataset_type=mock only.")

    prepare_mock_data_if_needed(cfg)
    manifest_path = resolve_config_path(cfg, cfg["data"][f"{split}_manifest"])
    manifest = load_json(manifest_path)
    episode_entries = manifest["episodes"]
    if max_episodes is not None:
        episode_entries = episode_entries[:max_episodes]

    results = []
    for entry in episode_entries:
        episode_path = _normalise_manifest_path(entry["path"], base_dir=manifest_path.parent)
        episode = load_episode(episode_path)
        initial_position = episode["proprio"][0, :2]
        initial_velocity = episode["proprio"][0, 2:4]
        meta = episode["meta"]
        rollout = episode["action"]
        position = initial_position.clone()
        velocity = initial_velocity.clone()
        for action in rollout:
            velocity = action
            position = (position + float(meta["dt"]) * velocity).clamp(0.05, 0.95)
        target = meta["target"].float()
        final_distance = float((position - target).norm())
        success = float(rollout_success(position.unsqueeze(0), target.unsqueeze(0)).item())
        results.append(
            {
                "episode_id": episode["episode_id"],
                "success": success,
                "final_distance": final_distance,
                "num_actions": int(rollout.size(0)),
            }
        )

    success_rate = sum(item["success"] for item in results) / max(len(results), 1)
    mean_final_distance = sum(item["final_distance"] for item in results) / max(len(results), 1)
    mean_num_actions = sum(item["num_actions"] for item in results) / max(len(results), 1)
    return {
        "split": split,
        "episodes": len(results),
        "rollout_success": success_rate,
        "mean_final_distance": mean_final_distance,
        "mean_num_actions": mean_num_actions,
        "details": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--split", type=str, default="val", choices=("train", "val"))
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    metrics = evaluate_oracle_replay(cfg, split=args.split, max_episodes=args.max_episodes)
    print(
        {
            "split": metrics["split"],
            "episodes": metrics["episodes"],
            "rollout_success": round(float(metrics["rollout_success"]), 4),
            "mean_final_distance": round(float(metrics["mean_final_distance"]), 4),
            "mean_num_actions": round(float(metrics["mean_num_actions"]), 2),
        }
    )
    if args.output_json:
        save_json(metrics, Path(args.output_json))


if __name__ == "__main__":
    main()
