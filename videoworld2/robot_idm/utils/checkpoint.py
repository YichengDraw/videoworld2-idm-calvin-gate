from __future__ import annotations

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
