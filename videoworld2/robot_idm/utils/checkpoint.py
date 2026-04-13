from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


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
    return torch.load(Path(path), map_location=map_location, weights_only=False)


def find_resume_path(output_dir: str | Path, explicit: str | None = None) -> Path | None:
    if explicit:
        candidate = Path(explicit)
        if candidate.exists():
            return candidate
    last_path, _ = checkpoint_paths(output_dir)
    if last_path.exists():
        return last_path
    return None
