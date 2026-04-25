from __future__ import annotations

import json
import os
import random
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import torch


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_torch_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def make_worker_init_fn(seed: int) -> Callable[[int], None]:
    def _seed_worker(worker_id: int) -> None:
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed % (2**32))
        torch.manual_seed(worker_seed)

    return _seed_worker


def configure_determinism(seed: int, deterministic: bool = True) -> None:
    seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_json(data: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_device(requested: str | None = None) -> torch.device:
    if requested and requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@contextmanager
def autocast_if_cuda(device: torch.device, enabled: bool = True) -> Iterator[None]:
    if enabled and device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            yield
        return
    with nullcontext():
        yield


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved
