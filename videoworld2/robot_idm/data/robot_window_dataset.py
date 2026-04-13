from __future__ import annotations

import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from videoworld2.robot_idm.utils.runtime import load_json, save_json


@dataclass(frozen=True)
class WindowSpec:
    history_frames: int = 4
    past_action_hist: int = 4
    future_video_horizon: int = 8
    action_chunk: int = 8
    stride: int = 4


def _to_tensor(value: Any, dtype: torch.dtype | None = None) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


def _resize_video(video: torch.Tensor, image_size: int | None) -> torch.Tensor:
    if image_size is None or video.shape[-1] == image_size:
        return video
    resized = F.interpolate(video, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return resized


def _load_h5(path: Path) -> dict[str, Any]:
    def visit(group):
        payload: dict[str, Any] = {}
        for key, value in group.items():
            if isinstance(value, h5py.Dataset):
                payload[key] = value[()]
            else:
                payload[key] = visit(value)
        return payload

    with h5py.File(path, "r") as handle:
        return visit(handle)


def load_episode(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix in {".pt", ".pth"}:
        raw = torch.load(source, map_location="cpu", weights_only=False)
    elif suffix == ".npz":
        with np.load(source, allow_pickle=True) as handle:
            raw = {key: handle[key] for key in handle.files}
    elif suffix in {".pkl", ".pickle"}:
        with source.open("rb") as handle:
            raw = pickle.load(handle)
    elif suffix in {".h5", ".hdf5"}:
        raw = _load_h5(source)
    elif suffix == ".json":
        raw = load_json(source)
    else:
        raise ValueError(f"Unsupported episode format: {source}")

    rgb_static = _to_tensor(raw["rgb_static"], dtype=torch.float32)
    if rgb_static.ndim != 4:
        raise ValueError(f"Expected rgb_static as [T, C, H, W], got {rgb_static.shape}")
    if rgb_static.max() > 1.5:
        rgb_static = rgb_static / 255.0

    proprio = _to_tensor(raw["proprio"], dtype=torch.float32)
    action = _to_tensor(raw["action"], dtype=torch.float32)

    episode = {
        "rgb_static": rgb_static,
        "proprio": proprio,
        "action": action,
        "lang": raw.get("lang", ""),
        "task_id": int(raw.get("task_id", 0)),
        "embodiment_id": int(raw.get("embodiment_id", 0)),
        "episode_id": raw.get("episode_id", source.stem),
        "meta": raw.get("meta", {}),
    }
    if "rgb_gripper" in raw:
        rgb_gripper = _to_tensor(raw["rgb_gripper"], dtype=torch.float32)
        if rgb_gripper.max() > 1.5:
            rgb_gripper = rgb_gripper / 255.0
        episode["rgb_gripper"] = rgb_gripper
    return episode


def build_window_index(
    episode_entries: list[dict[str, Any]],
    spec: WindowSpec,
) -> list[dict[str, Any]]:
    index: list[dict[str, Any]] = []
    min_t = max(spec.history_frames - 1, spec.past_action_hist)
    for entry in episode_entries:
        episode = load_episode(entry["path"])
        num_frames = int(episode["rgb_static"].size(0))
        num_actions = int(episode["action"].size(0))
        max_t = min(num_actions - spec.action_chunk, num_frames - spec.future_video_horizon - 1)
        if max_t < min_t:
            continue
        for t in range(min_t, max_t + 1, spec.stride):
            index.append(
                {
                    "episode_path": entry["path"],
                    "episode_id": episode["episode_id"],
                    "task_id": episode["task_id"],
                    "embodiment_id": episode["embodiment_id"],
                    "t": t,
                }
            )
    return index


class RobotWindowDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        index_path: str | Path | None = None,
        spec: WindowSpec | None = None,
        image_size: int | None = None,
        limit_windows: int | None = None,
        rebuild_index: bool = False,
    ) -> None:
        manifest = load_json(manifest_path)
        self.episode_entries = manifest["episodes"]
        self.spec = spec or WindowSpec()
        self.image_size = image_size
        self._episode_cache: dict[str, dict[str, Any]] = {}

        if index_path is not None:
            self.index_path = Path(index_path)
            if not self.index_path.is_absolute():
                self.index_path = Path(manifest_path).resolve().parent / self.index_path
            if rebuild_index or not self.index_path.exists():
                index = build_window_index(self.episode_entries, self.spec)
                save_json({"windows": index, "spec": asdict(self.spec)}, self.index_path)
            else:
                index = load_json(self.index_path)["windows"]
        else:
            index = build_window_index(self.episode_entries, self.spec)

        if limit_windows is not None:
            index = index[:limit_windows]
        self.index = index

    def __len__(self) -> int:
        return len(self.index)

    def _get_episode(self, episode_path: str) -> dict[str, Any]:
        if episode_path not in self._episode_cache:
            self._episode_cache[episode_path] = load_episode(episode_path)
        return self._episode_cache[episode_path]

    def __getitem__(self, item: int) -> dict[str, Any]:
        record = self.index[item]
        episode = self._get_episode(record["episode_path"])
        t = int(record["t"])
        spec = self.spec

        rgb_static = episode["rgb_static"]
        proprio = episode["proprio"]
        actions = episode["action"]

        rgb_hist = rgb_static[t - spec.history_frames + 1 : t + 1]
        future_clip = rgb_static[t : t + spec.future_video_horizon + 1]
        proprio_hist = proprio[t - spec.history_frames + 1 : t + 1]
        past_action_hist = actions[t - spec.past_action_hist : t]
        action_chunk = actions[t : t + spec.action_chunk]

        rgb_hist = _resize_video(rgb_hist, self.image_size)
        future_clip = _resize_video(future_clip, self.image_size)

        sample = {
            "rgb_hist": rgb_hist,
            "future_clip": future_clip,
            "proprio_hist": proprio_hist,
            "past_action_hist": past_action_hist,
            "action_chunk": action_chunk,
            "lang": episode["lang"],
            "task_id": episode["task_id"],
            "embodiment_id": episode["embodiment_id"],
            "episode_id": episode["episode_id"],
            "t": t,
            "meta": episode.get("meta", {}),
        }
        if "rgb_gripper" in episode:
            sample["rgb_gripper_hist"] = episode["rgb_gripper"][t - spec.history_frames + 1 : t + 1]
        return sample
