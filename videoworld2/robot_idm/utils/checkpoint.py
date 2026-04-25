from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch


def resolve_checkpoint_path(path: str | Path, base_dir: str | Path | None = None, label: str = "Checkpoint path") -> Path:
    raw_path = str(path)
    candidate = Path(path).expanduser()
    if raw_path.startswith("/") and not candidate.is_absolute():
        raise ValueError(f"{label} {raw_path} is a POSIX absolute path on this platform; remap it to a local path.")
    if candidate.is_absolute() or base_dir is None:
        return candidate
    return (Path(base_dir) / candidate).resolve()


def checkpoint_paths(output_dir: str | Path) -> tuple[Path, Path]:
    checkpoint_dir = Path(output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir / "last.pt", checkpoint_dir / "best.pt"


def save_checkpoint(
    output_dir: str | Path,
    payload: dict[str, Any],
    is_best: bool = False,
) -> Path:
    last_path, best_path = checkpoint_paths(output_dir)
    torch.save(payload, last_path)
    if is_best:
        torch.save(payload, best_path)
        return best_path
    return last_path


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(resolve_checkpoint_path(path), map_location=map_location, weights_only=False)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_reference(path: str | Path) -> dict[str, Any]:
    resolved = resolve_checkpoint_path(path).resolve(strict=False)
    payload: dict[str, Any] = {"path": resolved.as_posix(), "exists": resolved.exists()}
    if resolved.exists() and resolved.is_file():
        stat = resolved.stat()
        payload.update({"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns), "sha256": _file_sha256(resolved)})
    return payload


def idm_conditioning_metadata(cfg: dict[str, Any]) -> dict[str, Any]:
    idm_cfg = cfg.get("idm", {})
    return {
        "variant": idm_cfg.get("variant", ""),
        "use_future_codes": bool(idm_cfg.get("use_future_codes", True)),
        "use_past_actions": bool(idm_cfg.get("use_past_actions", True)),
        "code_source": idm_cfg.get("code_source", "gt"),
        "mixed_code_training": bool(idm_cfg.get("mixed_code_training", False)),
        "mixed_code_ratio_gt": float(idm_cfg.get("mixed_code_ratio_gt", 0.0)),
        "mixed_code_ratio_pred": float(idm_cfg.get("mixed_code_ratio_pred", 0.0)),
        "mixed_code_ratio_noisy": float(idm_cfg.get("mixed_code_ratio_noisy", 0.0)),
    }


def adapter_checkpoint_metadata(cfg: dict[str, Any]) -> dict[str, Any]:
    adapter_cfg = cfg.get("adapter", {})
    metadata = {
        "backend": adapter_cfg.get("backend", ""),
        "embed_dim": int(adapter_cfg.get("embed_dim", 0)),
        "vocab_size": int(adapter_cfg.get("vocab_size", 0)),
        "n_codes": int(adapter_cfg.get("n_codes", 0)),
        "init_seed": int(adapter_cfg.get("init_seed", 0)),
        "hidden_dim": int(adapter_cfg.get("hidden_dim", 0)),
        "official_kwargs": adapter_cfg.get("official_kwargs", {}),
        "allow_partial_checkpoint": bool(adapter_cfg.get("allow_partial_checkpoint", False)),
    }
    checkpoint_path = adapter_cfg.get("checkpoint_path")
    if checkpoint_path:
        config_dir = Path(adapter_cfg["_config_dir"]) if adapter_cfg.get("_config_dir") else Path(cfg.get("_meta", {}).get("config_path", ".")).parent
        resolved = resolve_checkpoint_path(checkpoint_path, base_dir=config_dir, label="Adapter checkpoint path").resolve(strict=False)
        metadata["checkpoint"] = checkpoint_reference(resolved)
    return metadata


def policy_checkpoint_metadata(
    cfg: dict[str, Any],
    checkpoint_key: str,
    action_dim: int,
    checkpoint_refs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = {
        "checkpoint_key": checkpoint_key,
        "policy_variant": cfg.get("policy", {}).get("variant", ""),
        "idm_variant": cfg.get("idm", {}).get("variant", ""),
        "idm_conditioning": idm_conditioning_metadata(cfg),
        "action_dim": int(action_dim),
        "action_chunk": int(cfg["data"].get("action_chunk", 8)),
        "model": cfg.get("model", {}),
        "adapter": adapter_checkpoint_metadata(cfg),
        "data": {
            "use_proprio": bool(cfg["data"].get("use_proprio", True)),
            "use_lang": bool(cfg["data"].get("use_lang", True)),
            "proprio_dim": int(cfg["data"].get("proprio_dim", 4)),
        },
    }
    if checkpoint_refs:
        metadata["checkpoint_refs"] = checkpoint_refs
    return metadata


def teacher_policy_checkpoint_metadata(
    cfg: dict[str, Any],
    action_dim: int,
    checkpoint_refs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    teacher_cfg = dict(cfg)
    teacher_cfg["policy"] = {}
    return policy_checkpoint_metadata(teacher_cfg, "idm", action_dim, checkpoint_refs=checkpoint_refs)


def auxiliary_checkpoint_metadata(cfg: dict[str, Any], checkpoint_key: str, action_dim: int | None = None) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "checkpoint_key": checkpoint_key,
        "model": cfg.get("model", {}),
        "data": {
            "use_proprio": bool(cfg["data"].get("use_proprio", True)),
            "use_lang": bool(cfg["data"].get("use_lang", True)),
            "proprio_dim": int(cfg["data"].get("proprio_dim", 4)),
            "action_chunk": int(cfg["data"].get("action_chunk", 8)),
        },
        "adapter": {
            **adapter_checkpoint_metadata(cfg),
        },
    }
    if action_dim is not None:
        metadata["action_dim"] = int(action_dim)
    return metadata


def validate_checkpoint_metadata(
    checkpoint: dict[str, Any],
    expected_metadata: dict[str, Any],
    checkpoint_path: str | Path,
) -> None:
    metadata = checkpoint.get("model_metadata")
    if not metadata:
        raise ValueError(f"Checkpoint metadata missing for {checkpoint_path}; rerun training with current metadata guards.")
    mismatches = []
    for key, expected_value in expected_metadata.items():
        if metadata.get(key) != expected_value:
            mismatches.append((key, metadata.get(key, "<missing>"), expected_value))
    if mismatches:
        details = "; ".join(f"{key}: checkpoint={actual!r}, expected={expected!r}" for key, actual, expected in mismatches[:5])
        raise ValueError(f"Checkpoint metadata mismatch for {checkpoint_path}: {details}.")


def find_resume_path(output_dir: str | Path, explicit: str | None = None) -> Path | None:
    if explicit:
        candidate = resolve_checkpoint_path(explicit, label="Resume checkpoint path")
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Explicit resume checkpoint not found: {candidate}")
    last_path, _ = checkpoint_paths(output_dir)
    if last_path.exists():
        return last_path
    return None
