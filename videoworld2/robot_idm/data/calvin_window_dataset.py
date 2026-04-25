from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from videoworld2.robot_idm.data.robot_window_dataset import WindowSpec, _resize_video, _to_tensor
from videoworld2.robot_idm.utils.latent_cache import LatentCodeCache
from videoworld2.robot_idm.utils.runtime import load_json, save_json


def _load_frame_npz(path: str | Path) -> dict[str, Any]:
    with np.load(Path(path), allow_pickle=True) as handle:
        return {key: handle[key] for key in handle.files}


def _select_first(raw: dict[str, Any], keys: list[str], kind: str) -> Any:
    for key in keys:
        if key in raw:
            return raw[key]
    raise KeyError(f"Missing {kind}. Tried keys: {keys}")


def build_calvin_window_index(episode_entries: list[dict[str, Any]], spec: WindowSpec) -> list[dict[str, Any]]:
    index: list[dict[str, Any]] = []
    min_t = max(spec.history_frames - 1, spec.past_action_hist)
    for entry in episode_entries:
        num_frames = int(entry["end"]) - int(entry["start"]) + 1
        num_actions = num_frames
        max_t = min(num_actions - spec.action_chunk, num_frames - spec.future_video_horizon - 1)
        if max_t < min_t:
            continue
        for t in range(min_t, max_t + 1, spec.stride):
            index.append(
                {
                    "episode_id": entry["episode_id"],
                    "task_id": int(entry.get("task_id", 0)),
                    "embodiment_id": int(entry.get("embodiment_id", 0)),
                    "t": t,
                }
            )
    return index


def _calvin_frame_fingerprint(root_value: str | Path, frame_idx: int) -> dict[str, Any]:
    root = Path(root_value).expanduser()
    frame_path = (root / f"episode_{frame_idx:07d}.npz").resolve(strict=False)
    payload: dict[str, Any] = {"path": frame_path.as_posix(), "exists": frame_path.exists()}
    if frame_path.exists() and frame_path.is_file():
        stat = frame_path.stat()
        payload.update({"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)})
    return payload


def calvin_index_metadata(episode_entries: list[dict[str, Any]], spec: WindowSpec, image_size: int | None) -> dict[str, Any]:
    return {
        "metadata_version": 2,
        "spec": asdict(spec),
        "image_size": image_size,
        "episodes": [
            {
                "episode_id": entry.get("episode_id"),
                "root": entry.get("root"),
                "start": int(entry.get("start", -1)),
                "end": int(entry.get("end", -1)),
            }
            for entry in episode_entries
        ],
        "episode_frame_files": [
            {
                "episode_id": entry.get("episode_id"),
                "first": _calvin_frame_fingerprint(entry.get("root", ""), int(entry.get("start", -1))),
                "last": _calvin_frame_fingerprint(entry.get("root", ""), int(entry.get("end", -1))),
            }
            for entry in episode_entries
        ],
    }


def _load_or_rebuild_index(index_path: Path, episode_entries: list[dict[str, Any]], spec: WindowSpec, image_size: int | None, rebuild_index: bool) -> list[dict[str, Any]]:
    expected_metadata = calvin_index_metadata(episode_entries, spec, image_size)
    if rebuild_index or not index_path.exists():
        index = build_calvin_window_index(episode_entries, spec)
        save_json({"windows": index, "metadata": expected_metadata}, index_path)
        return index

    payload = load_json(index_path)
    if payload.get("metadata") != expected_metadata:
        raise ValueError(f"CALVIN window index metadata mismatch in {index_path}; rebuild the index.")
    return payload["windows"]


class CalvinWindowDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        index_path: str | Path | None = None,
        spec: WindowSpec | None = None,
        image_size: int | None = None,
        limit_windows: int | None = None,
        rebuild_index: bool = False,
        cache_size: int = 8,
    ) -> None:
        manifest = load_json(manifest_path)
        self.episode_entries = manifest["episodes"]
        self.entry_by_id = {entry["episode_id"]: entry for entry in self.episode_entries}
        self.spec = spec or WindowSpec()
        self.image_size = image_size
        self.cache_size = cache_size
        self._episode_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()

        if index_path is not None:
            self.index_path = Path(index_path)
            if not self.index_path.is_absolute():
                self.index_path = Path(manifest_path).resolve().parent / self.index_path
            index = _load_or_rebuild_index(self.index_path, self.episode_entries, self.spec, self.image_size, rebuild_index)
        else:
            index = build_calvin_window_index(self.episode_entries, self.spec)

        if limit_windows is not None:
            index = index[:limit_windows]
        self.index = index

    def __len__(self) -> int:
        return len(self.index)

    def _load_episode(self, entry: dict[str, Any]) -> dict[str, Any]:
        root = Path(entry["root"])
        start = int(entry["start"])
        end = int(entry["end"])
        frames = []
        gripper_frames = []
        proprio = []
        actions = []
        for frame_idx in range(start, end + 1):
            frame_path = root / f"episode_{frame_idx:07d}.npz"
            raw = _load_frame_npz(frame_path)
            image = _to_tensor(_select_first(raw, ["rgb_static", "rgb"], "rgb_static"), dtype=torch.float32)
            if image.ndim == 3:
                image = image.permute(2, 0, 1)
            if image.max() > 1.5:
                image = image / 255.0
            frames.append(image)

            if "rgb_gripper" in raw:
                gripper = _to_tensor(raw["rgb_gripper"], dtype=torch.float32)
                if gripper.ndim == 3:
                    gripper = gripper.permute(2, 0, 1)
                if gripper.max() > 1.5:
                    gripper = gripper / 255.0
                gripper_frames.append(gripper)

            proprio.append(_to_tensor(_select_first(raw, ["robot_obs", "proprio"], "proprio"), dtype=torch.float32))
            actions.append(_to_tensor(_select_first(raw, ["rel_actions", "actions"], "action"), dtype=torch.float32))

        episode = {
            "rgb_static": torch.stack(frames),
            "proprio": torch.stack(proprio),
            "action": torch.stack(actions),
            "lang": entry.get("lang", ""),
            "task_id": int(entry.get("task_id", 0)),
            "embodiment_id": int(entry.get("embodiment_id", 0)),
            "episode_id": entry["episode_id"],
            "meta": {
                "root": str(root),
                "start": start,
                "end": end,
            },
        }
        if gripper_frames:
            episode["rgb_gripper"] = torch.stack(gripper_frames)
        return episode

    def _get_episode(self, episode_id: str) -> dict[str, Any]:
        if episode_id in self._episode_cache:
            self._episode_cache.move_to_end(episode_id)
            return self._episode_cache[episode_id]

        episode = self._load_episode(self.entry_by_id[episode_id])
        self._episode_cache[episode_id] = episode
        while len(self._episode_cache) > self.cache_size:
            self._episode_cache.popitem(last=False)
        return episode

    def __getitem__(self, item: int) -> dict[str, Any]:
        record = self.index[item]
        episode = self._get_episode(record["episode_id"])
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
            "meta": episode["meta"],
        }
        if "rgb_gripper" in episode:
            sample["rgb_gripper_hist"] = episode["rgb_gripper"][t - spec.history_frames + 1 : t + 1]
        return sample


class CalvinLatentWindowDataset(CalvinWindowDataset):
    def __init__(
        self,
        manifest_path: str | Path,
        cache_path: str | Path,
        index_path: str | Path | None = None,
        spec: WindowSpec | None = None,
        image_size: int | None = None,
        limit_windows: int | None = None,
        rebuild_index: bool = False,
    ) -> None:
        super().__init__(
            manifest_path=manifest_path,
            index_path=index_path,
            spec=spec,
            image_size=image_size,
            limit_windows=limit_windows,
            rebuild_index=rebuild_index,
        )
        self.cache = LatentCodeCache(cache_path)

    def __getitem__(self, item: int) -> dict[str, Any]:
        sample = super().__getitem__(item)
        record = self.cache.get(sample["episode_id"], int(sample["t"]))
        sample["future_codes"] = record["codes"].long()
        sample["future_code_embeds"] = record["embeds"].float()
        return sample
