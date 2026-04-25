from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from videoworld2.robot_idm.data.calvin_window_dataset import build_calvin_window_index
from videoworld2.robot_idm.data.robot_window_dataset import WindowSpec
from videoworld2.robot_idm.utils.factory import validate_manifest_pair


def load_json(path: Path) -> dict[str, Any] | list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(root: Path, rel_path: str) -> dict[str, Any]:
    path = root / rel_path
    return {
        "path": rel_path,
        "size": path.stat().st_size,
        "sha256": sha256(path),
    }


def load_windows(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    if isinstance(payload, dict) and "windows" in payload:
        return payload["windows"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unrecognised window index payload: {path}")


def cache_record_count(path: Path) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return len(payload["records"])


def same_root_overlap_pairs(entries: list[dict[str, Any]]) -> int:
    ranges_by_root: dict[str, list[tuple[int, int]]] = {}
    for entry in entries:
        ranges_by_root.setdefault(str(entry["root"]), []).append((int(entry["start"]), int(entry["end"])))

    total = 0
    for ranges in ranges_by_root.values():
        ordered = sorted(ranges)
        for i, (_, end) in enumerate(ordered):
            for start_j, _ in ordered[i + 1 :]:
                if start_j > end:
                    break
                total += 1
    return total


def audit_rescue(root: Path) -> dict[str, Any]:
    train_manifest_path = root / "datasets/calvin_static/train_manifest.json"
    val_manifest_path = root / "datasets/calvin_static/val_manifest.json"
    train_windows_path = root / "cache/train_windows.json"
    val_windows_path = root / "cache/val_windows.json"
    train_cache_path = root / "cache/train_local_codes.pt"
    val_cache_path = root / "cache/val_local_codes.pt"

    train_manifest = load_json(train_manifest_path)
    val_manifest = load_json(val_manifest_path)
    assert isinstance(train_manifest, dict)
    assert isinstance(val_manifest, dict)
    validate_manifest_pair(train_manifest_path, val_manifest_path)

    spec = WindowSpec(history_frames=4, past_action_hist=4, future_video_horizon=8, action_chunk=8, stride=4)
    train_expected_windows = build_calvin_window_index(train_manifest["episodes"], spec)
    val_expected_windows = build_calvin_window_index(val_manifest["episodes"], spec)
    train_rescued_windows = load_windows(train_windows_path)
    val_rescued_windows = load_windows(val_windows_path)

    controller_eval_files = [
        "models/bc_vis_calvin_4090/offline_eval.json",
        "models/bc_vis_proprio_calvin_4090/offline_eval.json",
        "models/history_gt_calvin_4090/offline_eval.json",
        "models/pair_idm_calvin_4090/offline_eval.json",
        "models/vw2_hidden_mlp_action_head_calvin_4090/offline_eval.json",
    ]

    return {
        "source_bundle": "vw2_rescue_20260411",
        "manifest_pair_validation": "passed",
        "window_spec": {
            "history_frames": spec.history_frames,
            "past_action_hist": spec.past_action_hist,
            "future_video_horizon": spec.future_video_horizon,
            "action_chunk": spec.action_chunk,
            "stride": spec.stride,
        },
        "window_rebuild": {
            "train": {
                "manifest_episodes": len(train_manifest["episodes"]),
                "expected_windows": len(train_expected_windows),
                "rescued_windows": len(train_rescued_windows),
                "payload_matches_current_builder": train_expected_windows == train_rescued_windows,
                "same_root_within_split_overlap_pairs": same_root_overlap_pairs(train_manifest["episodes"]),
            },
            "val": {
                "manifest_episodes": len(val_manifest["episodes"]),
                "expected_windows": len(val_expected_windows),
                "rescued_windows": len(val_rescued_windows),
                "payload_matches_current_builder": val_expected_windows == val_rescued_windows,
                "same_root_within_split_overlap_pairs": same_root_overlap_pairs(val_manifest["episodes"]),
            },
        },
        "latent_cache": {
            "train_records": cache_record_count(train_cache_path),
            "val_records": cache_record_count(val_cache_path),
            "configured_limit_train_windows": 8192,
            "configured_limit_val_windows": 1024,
            "val_cache_is_configured_subset_of_full_val_index": cache_record_count(val_cache_path) == 1024
            and len(val_expected_windows) == 12881,
        },
        "interpretation_boundary": (
            "The recovered offline metrics are annotation-window-weighted values over the configured latent-cache subset. "
            "Same-root within-split span overlaps are treated as CALVIN language-annotation windows, not independent raw-frame claims."
        ),
        "source_files": [
            file_record(root, "datasets/calvin_static/train_manifest.json"),
            file_record(root, "datasets/calvin_static/val_manifest.json"),
            file_record(root, "cache/train_windows.json"),
            file_record(root, "cache/val_windows.json"),
            file_record(root, "cache/train_local_codes.pt"),
            file_record(root, "cache/val_local_codes.pt"),
            *[file_record(root, rel_path) for rel_path in controller_eval_files],
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit non-versioned rescued Phase 1 artifacts without committing raw data.")
    parser.add_argument("--rescue-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=Path("results/rescued_artifact_audit.json"))
    args = parser.parse_args()

    write_json(args.output_json, audit_rescue(args.rescue_root))


if __name__ == "__main__":
    main()
