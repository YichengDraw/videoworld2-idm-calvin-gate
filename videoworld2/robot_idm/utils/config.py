from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _is_unaddressable_posix_absolute(path_value: str | Path) -> bool:
    raw_path = str(path_value)
    return raw_path.startswith("/") and not Path(raw_path).expanduser().is_absolute()


def _guard_config_path(path_value: str | Path) -> None:
    if _is_unaddressable_posix_absolute(path_value):
        raise ValueError(f"Config path {path_value!s} is a POSIX absolute path on this platform; remap it to a local path.")


def _is_path_like_key(key: str) -> bool:
    return key.endswith(("_manifest", "_index", "_cache", "_checkpoint", "_path", "_dir")) or key in {
        "mock_root",
        "output_dir",
        "checkpoint",
        "config",
        "output_json",
        "report_path",
    }


def _collect_path_sources(payload: Any, source_dir: Path) -> dict[str, dict[str, Any]]:
    sources_by_key: dict[str, str] = {}
    dirs_by_value: dict[str, set[str]] = {}

    def visit(node: Any, path_parts: tuple[str, ...] = ()) -> None:
        if isinstance(node, dict):
            for child_key, child_value in node.items():
                visit(child_value, path_parts + (str(child_key),))
        elif isinstance(node, list):
            for item in node:
                visit(item, path_parts)
        elif isinstance(node, (str, Path)) and path_parts and _is_path_like_key(path_parts[-1]) and str(node):
            raw_path = str(node)
            source = str(source_dir)
            sources_by_key[".".join(path_parts)] = source
            dirs_by_value.setdefault(raw_path, set()).add(source)

    visit(payload)
    return {
        "by_key": sources_by_key,
        "dirs_by_value": {raw_path: sorted(source_dirs) for raw_path, source_dirs in dirs_by_value.items()},
    }


def _merge_path_source_meta(meta: dict[str, Any], path_sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    sources_by_key = deepcopy(meta.get("path_sources_by_key", {}))
    sources_by_key.update(path_sources.get("by_key", {}))

    dirs_by_value = {
        raw_path: set(source_dirs)
        for raw_path, source_dirs in meta.get("path_source_dirs_by_value", {}).items()
    }
    for raw_path, source_dirs in path_sources.get("dirs_by_value", {}).items():
        dirs_by_value.setdefault(raw_path, set()).update(source_dirs)

    sorted_dirs = {raw_path: sorted(source_dirs) for raw_path, source_dirs in dirs_by_value.items()}
    unique_sources = {
        raw_path: source_dirs[0]
        for raw_path, source_dirs in sorted_dirs.items()
        if len(source_dirs) == 1
    }
    conflicts = {
        raw_path: source_dirs
        for raw_path, source_dirs in sorted_dirs.items()
        if len(source_dirs) > 1
    }
    return {
        "path_sources_by_key": sources_by_key,
        "path_source_dirs_by_value": sorted_dirs,
        "path_sources_by_value": unique_sources,
        "path_source_conflicts_by_value": conflicts,
    }


def config_path_source_dir(cfg: dict[str, Any], path_value: str | Path, key_path: str | None = None) -> Path | None:
    raw_path = str(path_value)
    meta = cfg.get("_meta", {})
    if key_path:
        source_dir = meta.get("path_sources_by_key", {}).get(key_path)
        if source_dir:
            return Path(source_dir)
    conflicts = meta.get("path_source_conflicts_by_value", {})
    if raw_path in conflicts:
        raise ValueError(
            f"Ambiguous relative config path {raw_path!r}; keys from different config directories share this value. "
            "Pass the config key path to resolve it safely."
        )
    source_dir = meta.get("path_sources_by_value", {}).get(raw_path)
    return Path(source_dir) if source_dir else None


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    _guard_config_path(path)
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    path_sources = _collect_path_sources(config, config_path.parent)
    if isinstance(config.get("adapter"), dict) and "checkpoint_path" in config["adapter"]:
        config["adapter"] = deepcopy(config["adapter"])
        config["adapter"]["_config_dir"] = str(config_path.parent)

    extends = config.pop("extends", [])
    if isinstance(extends, (str, Path)):
        extends = [extends]

    merged: dict[str, Any] = {}
    for parent in extends:
        _guard_config_path(parent)
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = config_path.parent / parent_path
        merged = _deep_merge(merged, load_config(parent_path))

    merged = _deep_merge(merged, config)
    merged.setdefault("_meta", {})
    merged["_meta"].update(_merge_path_source_meta(merged["_meta"], path_sources))
    merged["_meta"]["config_path"] = str(config_path)
    return merged


def dump_config(config: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
