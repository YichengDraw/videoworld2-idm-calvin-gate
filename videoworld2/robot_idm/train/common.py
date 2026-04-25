from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from videoworld2.robot_idm.data.collate import robot_idm_collate
from videoworld2.robot_idm.models.dldm_local_adapter import _resolve_path_from_config_dir
from videoworld2.robot_idm.utils.checkpoint import find_resume_path, load_checkpoint
from videoworld2.robot_idm.utils.config import dump_config
from videoworld2.robot_idm.utils.factory import build_train_val_datasets
from videoworld2.robot_idm.utils.latent_cache import LatentCodeCache, extract_code_cache
from videoworld2.robot_idm.utils.mock_data import generate_mock_dataset
from videoworld2.robot_idm.utils.runtime import ensure_dir, load_json, make_torch_generator, make_worker_init_fn


def resolve_config_path(cfg: dict[str, Any], path_value: str) -> Path:
    raw_path = str(path_value)
    path = Path(path_value)
    if raw_path.startswith("/") and not path.is_absolute():
        raise ValueError(f"Config path {raw_path} is a POSIX absolute path on this platform; remap it to a local path.")
    if path.is_absolute():
        return path
    return (Path(cfg["_meta"]["config_path"]).parent / path).resolve()


def prepare_mock_data_if_needed(cfg: dict[str, Any]) -> None:
    data_cfg = cfg["data"]
    if data_cfg.get("dataset_type") != "mock":
        return

    train_manifest = resolve_config_path(cfg, data_cfg["train_manifest"])
    val_manifest = resolve_config_path(cfg, data_cfg["val_manifest"])
    if train_manifest.exists() and val_manifest.exists():
        return

    root_dir = resolve_config_path(cfg, data_cfg.get("mock_root", "datasets/mock_robot"))
    generate_mock_dataset(
        root=root_dir,
        train_episodes=int(data_cfg.get("mock_train_episodes", 64)),
        val_episodes=int(data_cfg.get("mock_val_episodes", 16)),
    )


def _adapter_cache_metadata(cfg: dict[str, Any], adapter) -> dict[str, Any]:
    adapter_cfg = cfg.get("adapter", {})
    metadata = {
        "adapter_backend": getattr(adapter, "backend", adapter_cfg.get("backend", "unknown")),
        "adapter_vocab_size": int(getattr(adapter, "vocab_size", int(adapter_cfg.get("vocab_size", 0)))),
        "adapter_n_codes": int(getattr(adapter, "n_codes", int(adapter_cfg.get("n_codes", 0)))),
        "adapter_embed_dim": int(getattr(adapter, "embed_dim", int(adapter_cfg.get("embed_dim", 0)))),
    }
    checkpoint_path = adapter_cfg.get("checkpoint_path")
    if checkpoint_path:
        config_dir = Path(adapter_cfg.get("_config_dir") or Path(cfg["_meta"]["config_path"]).parent)
        resolved = _resolve_path_from_config_dir(checkpoint_path, config_dir)
        metadata["adapter_checkpoint"] = str(resolved)
        if resolved.exists():
            stat = resolved.stat()
            metadata["adapter_checkpoint_size"] = int(stat.st_size)
            metadata["adapter_checkpoint_mtime_ns"] = int(stat.st_mtime_ns)
    return metadata


def _json_digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _is_unaddressable_posix_absolute(path_value: str | Path) -> bool:
    raw_path = str(path_value)
    return raw_path.startswith("/") and not Path(raw_path).expanduser().is_absolute()


def _normalise_manifest_path(path_value: str | Path, base_dir: Path | None = None) -> str:
    raw_path = str(path_value)
    path = Path(raw_path).expanduser()
    if raw_path.startswith("/") and not path.is_absolute():
        return path.as_posix()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path.resolve(strict=False).as_posix()


def _join_manifest_path(root_value: str | Path, name: str) -> str:
    root = str(root_value).rstrip("/\\")
    if _is_unaddressable_posix_absolute(root):
        return f"{root}/{name}"
    return (Path(root) / name).as_posix()


def _file_fingerprint(path_value: str | Path, base_dir: Path | None = None) -> dict[str, Any]:
    normalised_path = _normalise_manifest_path(path_value, base_dir=base_dir)
    if _is_unaddressable_posix_absolute(normalised_path):
        return {"path": normalised_path, "exists": False}
    resolved = Path(normalised_path)
    payload: dict[str, Any] = {"path": resolved.as_posix(), "exists": resolved.exists()}
    if resolved.exists() and resolved.is_file():
        stat = resolved.stat()
        payload.update({"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)})
    return payload


def _manifest_source_fingerprint(manifest_payload: dict[str, Any], manifest_dir: Path) -> list[dict[str, Any]]:
    fingerprints = []
    for entry in manifest_payload.get("episodes", []):
        item: dict[str, Any] = {"episode_id": entry.get("episode_id")}
        if "path" in entry:
            item["file"] = _file_fingerprint(entry["path"], base_dir=manifest_dir)
        if {"root", "start", "end"} <= set(entry):
            root = _normalise_manifest_path(entry["root"], base_dir=manifest_dir)
            item["frame_range"] = [int(entry["start"]), int(entry["end"])]
            item["frames"] = [
                _file_fingerprint(_join_manifest_path(root, f"episode_{frame_idx:07d}.npz"))
                for frame_idx in range(int(entry["start"]), int(entry["end"]) + 1)
            ]
        fingerprints.append(item)
    return fingerprints


def _dataset_cache_metadata(cfg: dict[str, Any], dataset, split: str, adapter) -> dict[str, Any]:
    spec = getattr(dataset, "spec", None)
    first_window = dataset.index[0] if getattr(dataset, "index", None) else {}
    last_window = dataset.index[-1] if getattr(dataset, "index", None) else {}
    manifest_path = resolve_config_path(cfg, cfg["data"][f"{split}_manifest"])
    manifest_payload = load_json(manifest_path)
    adapter_cfg = cfg.get("adapter", {})
    metadata = {
        "metadata_version": 2,
        "split": split,
        "dataset_type": cfg["data"].get("dataset_type", "standard"),
        "manifest_path": str(manifest_path),
        "manifest_digest": _json_digest(manifest_payload),
        "manifest_source_digest": _json_digest(_manifest_source_fingerprint(manifest_payload, manifest_path.parent)),
        "adapter_config_digest": _json_digest(adapter_cfg),
        "adapter_init_seed": _safe_int(getattr(adapter, "init_seed", adapter_cfg.get("init_seed", 0))),
        "image_size": cfg["data"].get("image_size"),
        "window_spec": vars(spec) if spec is not None else {},
        "num_windows": len(dataset),
        "first_window": {key: first_window.get(key) for key in ("episode_id", "t")},
        "last_window": {key: last_window.get(key) for key in ("episode_id", "t")},
    }
    metadata.update(_adapter_cache_metadata(cfg, adapter))
    return metadata


def _dataset_cache_keys(dataset) -> set[str]:
    return {LatentCodeCache.make_key(record["episode_id"], int(record["t"])) for record in dataset.index}


def _validate_non_empty_dataset(dataset, split: str) -> None:
    if len(dataset) == 0:
        raise ValueError(f"{split} dataset produced zero windows; check manifest paths, episode lengths, and window spec.")


def ensure_code_caches(cfg: dict[str, Any], adapter, device: torch.device) -> None:
    prepare_mock_data_if_needed(cfg)
    data_cfg = cfg["data"]
    train_cache = resolve_config_path(cfg, data_cfg["train_cache"])
    val_cache = resolve_config_path(cfg, data_cfg["val_cache"])
    overwrite_cache = bool(data_cfg.get("overwrite_cache", False))

    train_dataset, val_dataset = build_train_val_datasets(cfg, use_latent_cache=False)
    _validate_non_empty_dataset(train_dataset, "train")
    _validate_non_empty_dataset(val_dataset, "val")
    train_metadata = _dataset_cache_metadata(cfg, train_dataset, "train", adapter)
    val_metadata = _dataset_cache_metadata(cfg, val_dataset, "val", adapter)

    cache_specs = [
        ("train", train_dataset, train_cache, train_metadata, int(cfg["training"].get("seed", 7))),
        ("val", val_dataset, val_cache, val_metadata, int(cfg["training"].get("seed", 7)) + 1),
    ]
    for _split, dataset, cache_path, metadata, seed in cache_specs:
        expected_keys = _dataset_cache_keys(dataset)
        if cache_path.exists() and not overwrite_cache:
            LatentCodeCache(
                cache_path,
                expected_metadata=metadata,
                expected_keys=expected_keys,
                allow_legacy_metadata=False,
            )
            continue

        extract_code_cache(
            dataset,
            adapter,
            cache_path,
            batch_size=int(cfg["training"].get("cache_batch_size", 8)),
            num_workers=int(cfg["training"].get("num_workers", 0)),
            device=device,
            overwrite=overwrite_cache,
            metadata=metadata,
            seed=seed,
        )


def make_dataloaders(cfg: dict[str, Any], use_latent_cache: bool) -> tuple[DataLoader, DataLoader]:
    prepare_mock_data_if_needed(cfg)
    train_dataset, val_dataset = build_train_val_datasets(cfg, use_latent_cache=use_latent_cache)
    _validate_non_empty_dataset(train_dataset, "train")
    _validate_non_empty_dataset(val_dataset, "val")
    batch_size = int(cfg["training"].get("batch_size", 32))
    num_workers = int(cfg["training"].get("num_workers", 0))
    use_pin_memory = bool(cfg["training"].get("pin_memory", torch.cuda.is_available()))
    use_persistent_workers = num_workers > 0 and bool(cfg["training"].get("persistent_workers", True))
    prefetch_factor = int(cfg["training"].get("prefetch_factor", 2))
    worker_kwargs: dict[str, Any] = {}
    if num_workers > 0:
        worker_kwargs["persistent_workers"] = use_persistent_workers
        worker_kwargs["prefetch_factor"] = prefetch_factor
    seed = int(cfg["training"].get("seed", 7))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=robot_idm_collate,
        drop_last=False,
        pin_memory=use_pin_memory,
        generator=make_torch_generator(seed),
        worker_init_fn=make_worker_init_fn(seed),
        **worker_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=robot_idm_collate,
        drop_last=False,
        pin_memory=use_pin_memory,
        generator=make_torch_generator(seed + 1),
        worker_init_fn=make_worker_init_fn(seed + 1),
        **worker_kwargs,
    )
    return train_loader, val_loader


def make_output_dir(cfg: dict[str, Any]) -> Path:
    output_dir = resolve_config_path(cfg, cfg["training"]["output_dir"])
    ensure_dir(output_dir)
    dump_config(cfg, output_dir / "resolved_config.yaml")
    return output_dir


def maybe_resume_training(
    output_dir: str | Path,
    modules: dict[str, torch.nn.Module],
    optimizer: torch.optim.Optimizer | None = None,
    explicit_resume: str | None = None,
    strict: bool = True,
) -> tuple[int, int, float]:
    resume_path = find_resume_path(output_dir, explicit=explicit_resume)
    if resume_path is None:
        return 0, 0, float("-inf")

    state = load_checkpoint(resume_path)
    for name, module in modules.items():
        if name in state:
            module.load_state_dict(state[name], strict=strict)
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    return int(state.get("epoch", 0)), int(state.get("global_step", 0)), float(state.get("best_metric", float("-inf")))


def batch_to_state_encoder_inputs(batch: dict[str, Any]) -> dict[str, Any]:
    return {
        "rgb_hist": batch["rgb_hist"],
        "proprio_hist": batch.get("proprio_hist"),
        "lang_texts": batch.get("lang"),
        "embodiment_id": batch["embodiment_id"].long(),
    }


def sample_code_conditioning(
    cfg: dict[str, Any],
    batch: dict[str, Any],
    state_tokens: torch.Tensor,
    adapter,
    planner,
    training: bool,
) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
    idm_cfg = cfg.get("idm", {})
    if not idm_cfg.get("use_future_codes", True):
        return None, {}

    gt_codes = batch["future_codes"]
    gt_embeds = batch["future_code_embeds"]
    source = idm_cfg.get("code_source", "gt")
    require_predicted_codes = source == "predicted"
    planner_metrics: dict[str, torch.Tensor] = {}
    pred_codes = None
    pred_embeds = None
    if require_predicted_codes and planner is None:
        raise ValueError("Predicted-code conditioning requires a planner checkpoint.")
    if planner is not None:
        pred_codes = planner.sample(state_tokens)
        pred_embeds = adapter.code_embed(pred_codes)
        planner_metrics["planner_code_accuracy"] = (pred_codes == gt_codes).float().mean()

    if not training or not idm_cfg.get("mixed_code_training", False):
        if require_predicted_codes:
            return pred_embeds, planner_metrics
        return gt_embeds, planner_metrics

    probs = torch.rand(gt_embeds.size(0), device=gt_embeds.device)
    ratio_gt = float(idm_cfg.get("mixed_code_ratio_gt", 0.6))
    ratio_pred = float(idm_cfg.get("mixed_code_ratio_pred", 0.2))
    conditioned = gt_embeds.clone()

    noisy_mask = probs >= ratio_gt + ratio_pred
    if noisy_mask.any():
        noisy = gt_embeds[noisy_mask] + 0.05 * torch.randn_like(gt_embeds[noisy_mask])
        dropout_mask = torch.rand_like(noisy[..., :1]) < 0.1
        noisy = noisy.masked_fill(dropout_mask, 0.0)
        conditioned[noisy_mask] = noisy

    pred_mask = (probs >= ratio_gt) & (probs < ratio_gt + ratio_pred)
    if pred_mask.any() and pred_embeds is not None:
        conditioned[pred_mask] = pred_embeds[pred_mask]

    return conditioned, planner_metrics
