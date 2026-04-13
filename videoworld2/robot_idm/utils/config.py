from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if isinstance(config.get("adapter"), dict):
        config["adapter"] = deepcopy(config["adapter"])
        config["adapter"]["_config_dir"] = str(config_path.parent)

    extends = config.pop("extends", [])
    if isinstance(extends, (str, Path)):
        extends = [extends]

    merged: dict[str, Any] = {}
    for parent in extends:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = config_path.parent / parent_path
        merged = _deep_merge(merged, load_config(parent_path))

    merged = _deep_merge(merged, config)
    merged.setdefault("_meta", {})
    merged["_meta"]["config_path"] = str(config_path)
    return merged


def dump_config(config: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
