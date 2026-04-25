from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from videoworld2.robot_idm.utils.config import dump_config
from videoworld2.robot_idm.utils.mock_data import generate_mock_dataset
from videoworld2.robot_idm.utils.runtime import ensure_dir


def repo_root_from_config_path(config_path: str | Path) -> Path:
    return Path(config_path).resolve().parents[2]


def prepare_phase0_overfit_cfg(
    base_cfg: dict[str, Any],
    run_name: str,
    episodes: int,
    max_epochs: int,
) -> dict[str, Any]:
    cfg = deepcopy(base_cfg)
    repo_root = repo_root_from_config_path(cfg["_meta"]["config_path"])
    dataset_root = repo_root / "datasets" / "mock_robot_phase0" / run_name
    cache_root = repo_root / "cache" / "phase0" / run_name
    output_dir = repo_root / "outputs" / "vw2_idm" / run_name

    generate_mock_dataset(dataset_root, train_episodes=episodes, val_episodes=0)

    train_manifest = dataset_root / "train_manifest.json"
    train_index = cache_root / "train_windows.json"
    train_cache = cache_root / "train_local_codes.pt"

    cfg["experiment_name"] = run_name
    cfg["data"]["dataset_type"] = "mock"
    cfg["data"]["mock_root"] = str(dataset_root)
    cfg["data"]["mock_train_episodes"] = episodes
    cfg["data"]["mock_val_episodes"] = 0
    cfg["data"]["train_manifest"] = str(train_manifest)
    cfg["data"]["val_manifest"] = str(train_manifest)
    cfg["data"]["train_index"] = str(train_index)
    cfg["data"]["val_index"] = str(train_index)
    cfg["data"]["train_cache"] = str(train_cache)
    cfg["data"]["val_cache"] = str(train_cache)
    cfg["data"]["overwrite_cache"] = True
    cfg["data"]["validate_split_disjoint"] = False
    cfg["data"]["limit_train_windows"] = None
    cfg["data"]["limit_val_windows"] = None
    if not cfg.get("idm", {}).get("use_future_codes", True) or cfg.get("idm", {}).get("code_source", "gt") != "predicted":
        cfg.setdefault("idm", {})["planner_checkpoint"] = ""
    cfg["training"]["output_dir"] = str(output_dir)
    cfg["training"]["max_epochs"] = max_epochs
    cfg.setdefault("logging", {}).setdefault("wandb", {})["enabled"] = False
    cfg.setdefault("evaluation", {})["num_rollouts"] = episodes
    return cfg


def write_runtime_config(cfg: dict[str, Any], output_dir: str | Path, file_name: str = "phase0_runtime.yaml") -> Path:
    destination = Path(output_dir) / file_name
    ensure_dir(destination.parent)
    dump_config(cfg, destination)
    return destination


def resolve_best_checkpoint(output_dir: str | Path) -> Path:
    checkpoint_path = Path(output_dir) / "checkpoints" / "best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Best checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def load_metric_history(output_dir: str | Path) -> list[dict[str, Any]]:
    metrics_path = Path(output_dir) / "metrics.jsonl"
    if not metrics_path.exists():
        return []
    records: list[dict[str, Any]] = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def summarize_metric_history(output_dir: str | Path) -> dict[str, float]:
    records = load_metric_history(output_dir)
    if not records:
        return {}

    summary: dict[str, float] = {
        "epochs": float(len(records)),
    }
    keys = [
        "train/action_mse",
        "val/action_mse",
        "train/action_nll",
        "val/action_nll",
        "train/planner_code_accuracy",
        "val/planner_code_accuracy",
    ]
    for key in keys:
        values = [float(record[key]) for record in records if key in record]
        if not values:
            continue
        summary[f"best_{key.replace('/', '_')}"] = min(values) if "mse" in key or "nll" in key else max(values)
        summary[f"final_{key.replace('/', '_')}"] = values[-1]
    return summary
