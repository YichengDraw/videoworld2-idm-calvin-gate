from __future__ import annotations

from collections import Counter
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch

from videoworld2.robot_idm.data.calvin_window_dataset import CalvinLatentWindowDataset, CalvinWindowDataset
from videoworld2.robot_idm.data.robot_latent_dataset import RobotLatentWindowDataset
from videoworld2.robot_idm.data.robot_window_dataset import RobotWindowDataset, WindowSpec, load_episode
from videoworld2.robot_idm.models.dldm_local_adapter import DLDMLocalAdapter
from videoworld2.robot_idm.models.direct_policy import DirectPolicy, MLPActionHead
from videoworld2.robot_idm.models.forward_verifier import ForwardVerifier
from videoworld2.robot_idm.models.inverse_dynamics import HistoryAwareIDM
from videoworld2.robot_idm.models.latent_planner import LatentPlanner
from videoworld2.robot_idm.models.state_encoder import StateEncoder
from videoworld2.robot_idm.utils.config import config_path_source_dir
from videoworld2.robot_idm.utils.runtime import load_json


def build_window_spec(cfg: dict[str, Any]) -> WindowSpec:
    return WindowSpec(
        history_frames=int(cfg["data"].get("history_frames", 4)),
        past_action_hist=int(cfg["data"].get("past_action_hist", 4)),
        future_video_horizon=int(cfg["data"].get("future_video_horizon", 8)),
        action_chunk=int(cfg["data"].get("action_chunk", 8)),
        stride=int(cfg["data"].get("stride", 4)),
    )


def _resolve_path(config: str | dict[str, Any], path_value: str, key_path: str | None = None) -> str:
    raw_path = str(path_value)
    candidate = Path(path_value)
    if raw_path.startswith("/") and not candidate.is_absolute():
        raise ValueError(f"Config path {raw_path} is a POSIX absolute path on this platform; remap it to a local path.")
    if candidate.is_absolute():
        return str(candidate)
    if isinstance(config, dict):
        source_dir = config_path_source_dir(config, raw_path, key_path=key_path)
        base_dir = source_dir if source_dir else Path(config["_meta"]["config_path"]).parent
    else:
        base_dir = Path(config).parent
    return str((base_dir / candidate).resolve())


def validate_manifest_pair(train_manifest: str | Path, val_manifest: str | Path) -> None:
    train_path = Path(train_manifest).resolve()
    val_path = Path(val_manifest).resolve()
    if train_path == val_path:
        raise ValueError(f"Train and validation manifests must be distinct: {train_path}")

    train_entries = load_json(train_path).get("episodes", [])
    val_entries = load_json(val_path).get("episodes", [])
    _validate_unique_episode_ids(train_entries, "train")
    _validate_unique_episode_ids(val_entries, "val")
    _validate_unique_episode_sources(train_entries, "train", train_path.parent)
    _validate_unique_episode_sources(val_entries, "val", val_path.parent)
    train_ids = {entry.get("episode_id") for entry in train_entries}
    val_ids = {entry.get("episode_id") for entry in val_entries}
    overlap_ids = sorted(train_ids & val_ids)
    if overlap_ids:
        raise ValueError(f"Train/val manifests share episode ids: {overlap_ids[:5]}")

    train_files = _manifest_files(train_entries, train_path.parent)
    val_files = _manifest_files(val_entries, val_path.parent)
    overlap_files = sorted(set(train_files) & set(val_files))
    if overlap_files:
        raise ValueError(f"Train/val manifests share episode source files: {overlap_files[:5]}")
    train_file_digests = _manifest_file_digests(train_entries, train_path.parent)
    val_file_digests = _manifest_file_digests(val_entries, val_path.parent)
    overlap_digests = sorted(set(train_file_digests) & set(val_file_digests))
    if overlap_digests:
        raise ValueError(f"Train/val manifests share episode source contents: {overlap_digests[:5]}")
    train_calvin_digests = _manifest_calvin_span_digests(train_entries, train_path.parent)
    val_calvin_digests = _manifest_calvin_span_digests(val_entries, val_path.parent)
    overlap_calvin_digests = sorted(set(train_calvin_digests) & set(val_calvin_digests))
    if overlap_calvin_digests:
        raise ValueError(f"Train/val manifests share CALVIN episode source contents: {overlap_calvin_digests[:5]}")
    overlap_calvin_frames = _overlapping_calvin_frame_content(train_entries, train_path.parent, val_entries, val_path.parent)
    if overlap_calvin_frames:
        raise ValueError(f"Train/val manifests share CALVIN frame contents: {overlap_calvin_frames[:5]}")

    train_spans = _manifest_spans(train_entries, train_path.parent)
    val_spans = _manifest_spans(val_entries, val_path.parent)
    for root, train_ranges in train_spans.items():
        for start, end in train_ranges:
            for val_start, val_end in val_spans.get(root, []):
                if start <= val_end and val_start <= end:
                    raise ValueError(f"Train/val manifests overlap on {root}: train={start}-{end}, val={val_start}-{val_end}")


def _validate_unique_episode_ids(entries: list[dict[str, Any]], split: str) -> None:
    seen: set[Any] = set()
    duplicates: list[Any] = []
    for entry in entries:
        episode_id = entry.get("episode_id")
        if episode_id in seen:
            duplicates.append(episode_id)
        seen.add(episode_id)
    if duplicates:
        raise ValueError(f"{split} manifest has duplicate episode ids: {duplicates[:5]}")


def _normalise_manifest_file(path_value: Any, manifest_dir: Path) -> str:
    raw_path = str(path_value)
    path = Path(raw_path).expanduser()
    if raw_path.startswith("/") and not path.is_absolute():
        return path.as_posix()
    if path.is_absolute():
        return path.resolve(strict=False).as_posix()
    return (manifest_dir / path).resolve(strict=False).as_posix()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_text(digest: "hashlib._Hash", value: str) -> None:
    payload = value.encode("utf-8")
    digest.update(str(len(payload)).encode("ascii"))
    digest.update(b":")
    digest.update(payload)


def _hash_bytes(digest: "hashlib._Hash", value: bytes) -> None:
    digest.update(str(len(value)).encode("ascii"))
    digest.update(b":")
    digest.update(value)


def _update_semantic_hash(digest: "hashlib._Hash", value: Any) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.layout != torch.strided:
            tensor = tensor.coalesce()
            _hash_text(digest, "sparse_tensor")
            _update_semantic_hash(digest, tensor.indices())
            _update_semantic_hash(digest, tensor.values())
            return
        tensor = tensor.contiguous()
        _hash_text(digest, "tensor")
        _hash_text(digest, str(tensor.dtype))
        _hash_text(digest, json.dumps(list(tensor.shape), separators=(",", ":")))
        try:
            _hash_bytes(digest, tensor.numpy().tobytes(order="C"))
        except (TypeError, RuntimeError):
            _hash_bytes(digest, tensor.view(torch.uint8).reshape(-1).numpy().tobytes(order="C"))
        return

    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        _hash_text(digest, "ndarray")
        _hash_text(digest, array.dtype.str)
        _hash_text(digest, json.dumps(list(array.shape), separators=(",", ":")))
        if array.dtype.hasobject:
            _update_semantic_hash(digest, array.tolist())
        else:
            _hash_bytes(digest, np.ascontiguousarray(array).tobytes(order="C"))
        return

    if isinstance(value, np.generic):
        _update_semantic_hash(digest, value.item())
        return

    if isinstance(value, dict):
        _hash_text(digest, "dict")
        _hash_text(digest, str(len(value)))
        for key, item in sorted(value.items(), key=lambda pair: (type(pair[0]).__name__, repr(pair[0]))):
            _update_semantic_hash(digest, key)
            _update_semantic_hash(digest, item)
        return

    if isinstance(value, (list, tuple)):
        _hash_text(digest, type(value).__name__)
        _hash_text(digest, str(len(value)))
        for item in value:
            _update_semantic_hash(digest, item)
        return

    if isinstance(value, (set, frozenset)):
        _hash_text(digest, type(value).__name__)
        item_hashes = sorted(_semantic_value_sha256(item) for item in value)
        _hash_text(digest, str(len(item_hashes)))
        for item_hash in item_hashes:
            _hash_text(digest, item_hash)
        return

    if isinstance(value, bytes):
        _hash_text(digest, "bytes")
        _hash_bytes(digest, value)
        return

    if isinstance(value, (str, int, float, bool)) or value is None:
        _hash_text(digest, type(value).__name__)
        _hash_text(digest, repr(value))
        return

    _hash_text(digest, type(value).__name__)
    _hash_text(digest, repr(value))


def _semantic_value_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    _update_semantic_hash(digest, value)
    return digest.hexdigest()


def _episode_model_payload(path: Path) -> dict[str, Any]:
    episode = load_episode(path)
    keys = ["rgb_static", "rgb_gripper", "proprio", "action", "lang", "task_id", "embodiment_id"]
    return {key: episode[key] for key in keys if key in episode}


def _load_npz_payload(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as handle:
        return {key: handle[key] for key in handle.files}


def _first_present(raw: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in raw:
            return raw[key]
    raise KeyError(keys)


def _image_tensor_for_digest(value: Any) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.ndim == 3:
        tensor = tensor.permute(2, 0, 1)
    if tensor.numel() and tensor.max() > 1.5:
        tensor = tensor / 255.0
    return tensor


def _calvin_frame_model_payload(path: Path) -> dict[str, Any]:
    raw = _load_npz_payload(path)
    payload = {
        "rgb_static": _image_tensor_for_digest(_first_present(raw, ["rgb_static", "rgb"])),
        "proprio": torch.as_tensor(_first_present(raw, ["robot_obs", "proprio"]), dtype=torch.float32),
        "action": torch.as_tensor(_first_present(raw, ["rel_actions", "actions"]), dtype=torch.float32),
    }
    if "rgb_gripper" in raw:
        payload["rgb_gripper"] = _image_tensor_for_digest(raw["rgb_gripper"])
    return payload


def _generic_loaded_payload(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        return torch.load(path, map_location="cpu", weights_only=False)
    if suffix in {".pkl", ".pickle"}:
        with path.open("rb") as handle:
            return pickle.load(handle)
    if suffix == ".json":
        return load_json(path)
    if suffix == ".npz":
        return _load_npz_payload(path)
    raise ValueError(f"Unsupported semantic digest format: {path}")


def _semantic_file_sha256(path: Path, *, calvin_frame: bool = False) -> str | None:
    suffix = path.suffix.lower()
    if suffix not in {".pt", ".pth", ".npz", ".pkl", ".pickle", ".json", ".h5", ".hdf5"}:
        return None
    try:
        if calvin_frame:
            payload = _calvin_frame_model_payload(path)
        else:
            payload = _episode_model_payload(path)
    except Exception:
        try:
            payload = _generic_loaded_payload(path)
        except Exception:
            return None
    return _semantic_value_sha256(payload)


def _episode_source_digest(path: Path, *, calvin_frame: bool = False) -> str:
    semantic_digest = _semantic_file_sha256(path, calvin_frame=calvin_frame)
    if semantic_digest is not None:
        return f"semantic:{semantic_digest}"
    return f"raw:{_file_sha256(path)}"


def _normalise_manifest_root(root_value: Any, manifest_dir: Path) -> str:
    raw_root = str(root_value)
    root_path = Path(raw_root).expanduser()
    if raw_root.startswith("/") and not root_path.is_absolute():
        return root_path.as_posix()
    if root_path.is_absolute():
        return root_path.resolve(strict=False).as_posix()
    return (manifest_dir / root_path).resolve(strict=False).as_posix()


def _manifest_files(entries: list[dict[str, Any]], manifest_dir: Path) -> list[str]:
    return [_normalise_manifest_file(entry["path"], manifest_dir) for entry in entries if "path" in entry]


def _manifest_file_digests(entries: list[dict[str, Any]], manifest_dir: Path) -> list[str]:
    digests = []
    for path_value in _manifest_files(entries, manifest_dir):
        if path_value.startswith("/") and not Path(path_value).expanduser().is_absolute():
            continue
        path = Path(path_value)
        if path.exists() and path.is_file():
            digests.append(_episode_source_digest(path))
    return digests


def _manifest_calvin_span_digests(entries: list[dict[str, Any]], manifest_dir: Path) -> list[str]:
    digests = []
    for entry in entries:
        if "root" not in entry or "start" not in entry or "end" not in entry:
            continue
        root = _normalise_manifest_root(entry["root"], manifest_dir)
        if root.startswith("/") and not Path(root).expanduser().is_absolute():
            continue
        root_path = Path(root)
        frame_hashes = []
        all_frames_present = True
        for frame_idx in range(int(entry["start"]), int(entry["end"]) + 1):
            frame_path = root_path / f"episode_{frame_idx:07d}.npz"
            if not frame_path.exists() or not frame_path.is_file():
                all_frames_present = False
                break
            frame_hashes.append(_episode_source_digest(frame_path, calvin_frame=True))
        if all_frames_present and frame_hashes:
            payload = "\n".join([str(len(frame_hashes)), *frame_hashes]).encode("utf-8")
            digests.append(hashlib.sha256(payload).hexdigest())
    return digests


def _manifest_calvin_frame_digests(entries: list[dict[str, Any]], manifest_dir: Path) -> dict[str, set[int]]:
    frame_digests: dict[str, set[int]] = {}
    for entry in entries:
        if "root" not in entry or "start" not in entry or "end" not in entry:
            continue
        root = _normalise_manifest_root(entry["root"], manifest_dir)
        if root.startswith("/") and not Path(root).expanduser().is_absolute():
            continue
        root_path = Path(root)
        for frame_idx in range(int(entry["start"]), int(entry["end"]) + 1):
            frame_path = root_path / f"episode_{frame_idx:07d}.npz"
            if frame_path.exists() and frame_path.is_file():
                frame_digests.setdefault(_episode_source_digest(frame_path, calvin_frame=True), set()).add(frame_idx)
    return frame_digests


def _manifest_calvin_frame_digest_locations(entries: list[dict[str, Any]], manifest_dir: Path) -> dict[str, list[dict[str, Any]]]:
    frame_digests: dict[str, list[dict[str, Any]]] = {}
    for entry_idx, entry in enumerate(entries):
        if "root" not in entry or "start" not in entry or "end" not in entry:
            continue
        root = _normalise_manifest_root(entry["root"], manifest_dir)
        if root.startswith("/") and not Path(root).expanduser().is_absolute():
            continue
        root_path = Path(root)
        for frame_idx in range(int(entry["start"]), int(entry["end"]) + 1):
            frame_path = root_path / f"episode_{frame_idx:07d}.npz"
            if frame_path.exists() and frame_path.is_file():
                frame_digests.setdefault(_episode_source_digest(frame_path, calvin_frame=True), []).append(
                    {
                        "entry_index": entry_idx,
                        "episode_id": entry.get("episode_id"),
                        "root": root,
                        "frame_index": frame_idx,
                    }
                )
    return frame_digests


def _overlapping_calvin_frame_content(
    train_entries: list[dict[str, Any]],
    train_dir: Path,
    val_entries: list[dict[str, Any]],
    val_dir: Path,
) -> list[dict[str, Any]]:
    train_frames = _manifest_calvin_frame_digests(train_entries, train_dir)
    val_frames = _manifest_calvin_frame_digests(val_entries, val_dir)
    overlaps = []
    for digest in sorted(set(train_frames) & set(val_frames)):
        overlaps.append(
            {
                "sha256": digest,
                "train_frame_indices": sorted(train_frames[digest])[:5],
                "val_frame_indices": sorted(val_frames[digest])[:5],
            }
        )
    return overlaps


def _duplicate_calvin_frame_content(entries: list[dict[str, Any]], manifest_dir: Path) -> list[dict[str, Any]]:
    frame_locations = _manifest_calvin_frame_digest_locations(entries, manifest_dir)
    duplicates = []
    for digest, locations in sorted(frame_locations.items()):
        roots = sorted({location["root"] for location in locations})
        if len(roots) > 1:
            duplicates.append(
                {
                    "sha256": digest,
                    "roots": roots[:5],
                    "locations": locations[:5],
                }
            )
    return duplicates


def _manifest_spans(entries: list[dict[str, Any]], manifest_dir: Path) -> dict[str, list[tuple[int, int]]]:
    spans: dict[str, list[tuple[int, int]]] = {}
    for entry in entries:
        if "root" not in entry or "start" not in entry or "end" not in entry:
            continue
        root = _normalise_manifest_root(entry["root"], manifest_dir)
        spans.setdefault(root, []).append((int(entry["start"]), int(entry["end"])))
    return spans


def _validate_unique_episode_sources(entries: list[dict[str, Any]], split: str, manifest_dir: Path) -> None:
    files = _manifest_files(entries, manifest_dir)
    duplicate_files = sorted(path for path, count in Counter(files).items() if count > 1)
    if duplicate_files:
        raise ValueError(f"{split} manifest has duplicate episode source files: {duplicate_files[:5]}")
    file_digests = _manifest_file_digests(entries, manifest_dir)
    duplicate_digests = sorted(digest for digest, count in Counter(file_digests).items() if count > 1)
    if duplicate_digests:
        raise ValueError(f"{split} manifest has duplicate episode source contents: {duplicate_digests[:5]}")
    calvin_digests = _manifest_calvin_span_digests(entries, manifest_dir)
    duplicate_calvin_digests = sorted(digest for digest, count in Counter(calvin_digests).items() if count > 1)
    if duplicate_calvin_digests:
        raise ValueError(f"{split} manifest has duplicate CALVIN episode source contents: {duplicate_calvin_digests[:5]}")
    duplicate_calvin_frames = _duplicate_calvin_frame_content(entries, manifest_dir)
    if duplicate_calvin_frames:
        raise ValueError(f"{split} manifest has duplicate CALVIN frame contents: {duplicate_calvin_frames[:5]}")
    spans = _manifest_spans(entries, manifest_dir)
    for root, ranges in spans.items():
        ordered = sorted(ranges)
        for idx in range(1, len(ordered)):
            previous_start, previous_end = ordered[idx - 1]
            start, end = ordered[idx]
            if start <= previous_end:
                raise ValueError(
                    f"{split} manifest has overlapping episode source spans on {root}: "
                    f"{previous_start}-{previous_end} and {start}-{end}"
                )


def build_adapter(cfg: dict[str, Any]) -> DLDMLocalAdapter:
    return DLDMLocalAdapter(cfg["adapter"])


def build_state_encoder(cfg: dict[str, Any]) -> StateEncoder:
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    return StateEncoder(
        image_backbone=model_cfg.get("image_backbone", "small_cnn"),
        d_model=int(model_cfg.get("d_model", 256)),
        temporal_depth=int(model_cfg.get("temporal_depth", 2)),
        n_heads=int(model_cfg.get("n_heads", 4)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        use_proprio=bool(data_cfg.get("use_proprio", True)),
        use_lang=bool(data_cfg.get("use_lang", True)),
        freeze_backbone=bool(model_cfg.get("freeze_backbone", False)),
        num_embodiments=int(model_cfg.get("num_embodiments", 8)),
        pretrained_backbone=bool(model_cfg.get("pretrained_backbone", False)),
        proprio_dim=int(data_cfg.get("proprio_dim", 4)),
    )


def build_planner(cfg: dict[str, Any], adapter: DLDMLocalAdapter) -> LatentPlanner:
    model_cfg = cfg["model"]
    return LatentPlanner(
        vocab_size=adapter.vocab_size,
        n_codes=adapter.n_codes,
        d_model=int(model_cfg.get("d_model", 256)),
        depth=int(model_cfg.get("planner_depth", 4)),
        n_heads=int(model_cfg.get("n_heads", 4)),
        dropout=float(model_cfg.get("dropout", 0.1)),
    )


def build_idm(cfg: dict[str, Any], action_dim: int) -> HistoryAwareIDM:
    model_cfg = cfg["model"]
    idm_cfg = cfg["idm"]
    return HistoryAwareIDM(
        action_dim=action_dim,
        chunk=int(cfg["data"].get("action_chunk", 8)),
        d_model=int(model_cfg.get("d_model", 256)),
        depth=int(model_cfg.get("idm_depth", 4)),
        n_heads=int(model_cfg.get("n_heads", 4)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        use_future_codes=bool(idm_cfg.get("use_future_codes", True)),
        use_past_actions=bool(idm_cfg.get("use_past_actions", True)),
        num_embodiments=int(model_cfg.get("num_embodiments", 8)),
    )


def build_verifier(cfg: dict[str, Any], adapter: DLDMLocalAdapter, action_dim: int) -> ForwardVerifier:
    model_cfg = cfg["model"]
    return ForwardVerifier(
        action_dim=action_dim,
        vocab_size=adapter.vocab_size,
        n_codes=adapter.n_codes,
        d_model=int(model_cfg.get("d_model", 256)),
        depth=int(model_cfg.get("verifier_depth", 3)),
        n_heads=int(model_cfg.get("n_heads", 4)),
        dropout=float(model_cfg.get("dropout", 0.1)),
    )


def build_direct_policy(cfg: dict[str, Any], action_dim: int) -> DirectPolicy:
    model_cfg = cfg["model"]
    policy_cfg = cfg.get("policy", {})
    if policy_cfg.get("variant", "") == "mlp":
        return MLPActionHead(
            action_dim=action_dim,
            chunk=int(cfg["data"].get("action_chunk", 8)),
            d_model=int(model_cfg.get("d_model", 256)),
            hidden_dim=int(policy_cfg.get("hidden_dim", model_cfg.get("d_model", 256) * 4)),
            num_embodiments=int(model_cfg.get("num_embodiments", 8)),
        )
    return DirectPolicy(
        action_dim=action_dim,
        chunk=int(cfg["data"].get("action_chunk", 8)),
        d_model=int(model_cfg.get("d_model", 256)),
        depth=int(model_cfg.get("idm_depth", 4)),
        n_heads=int(model_cfg.get("n_heads", 4)),
    )


def build_train_val_datasets(
    cfg: dict[str, Any],
    use_latent_cache: bool = False,
) -> tuple[RobotWindowDataset, RobotWindowDataset]:
    data_cfg = cfg["data"]
    spec = build_window_spec(cfg)
    image_size = int(data_cfg.get("image_size", 64))
    limit_train = data_cfg.get("limit_train_windows")
    limit_val = data_cfg.get("limit_val_windows")
    rebuild_index = bool(data_cfg.get("rebuild_index", data_cfg.get("overwrite_cache", False)))

    train_manifest = _resolve_path(cfg, data_cfg["train_manifest"], key_path="data.train_manifest")
    val_manifest = _resolve_path(cfg, data_cfg["val_manifest"], key_path="data.val_manifest")
    if bool(data_cfg.get("validate_split_disjoint", True)):
        validate_manifest_pair(train_manifest, val_manifest)
    train_index = _resolve_path(cfg, data_cfg["train_index"], key_path="data.train_index")
    val_index = _resolve_path(cfg, data_cfg["val_index"], key_path="data.val_index")
    dataset_type = data_cfg.get("dataset_type", "standard")
    cache_expected = cfg.get("_latent_cache_expected", {})
    if use_latent_cache and not cache_expected:
        raise ValueError("Latent cache validation metadata missing; call ensure_code_caches before building latent datasets.")
    train_cache_expected = cache_expected.get("train", {})
    val_cache_expected = cache_expected.get("val", {})

    if dataset_type == "calvin_static":
        if use_latent_cache:
            train_cache = _resolve_path(cfg, data_cfg["train_cache"], key_path="data.train_cache")
            val_cache = _resolve_path(cfg, data_cfg["val_cache"], key_path="data.val_cache")
            train_ds = CalvinLatentWindowDataset(
                train_manifest,
                train_cache,
                train_index,
                spec,
                image_size,
                limit_train,
                rebuild_index=rebuild_index,
                expected_cache_metadata=train_cache_expected.get("metadata"),
                expected_cache_keys=set(train_cache_expected.get("keys", [])),
            )
            val_ds = CalvinLatentWindowDataset(
                val_manifest,
                val_cache,
                val_index,
                spec,
                image_size,
                limit_val,
                rebuild_index=rebuild_index,
                expected_cache_metadata=val_cache_expected.get("metadata"),
                expected_cache_keys=set(val_cache_expected.get("keys", [])),
            )
        else:
            train_ds = CalvinWindowDataset(train_manifest, train_index, spec, image_size, limit_train, rebuild_index=rebuild_index)
            val_ds = CalvinWindowDataset(val_manifest, val_index, spec, image_size, limit_val, rebuild_index=rebuild_index)
    elif use_latent_cache:
        train_cache = _resolve_path(cfg, data_cfg["train_cache"], key_path="data.train_cache")
        val_cache = _resolve_path(cfg, data_cfg["val_cache"], key_path="data.val_cache")
        train_ds = RobotLatentWindowDataset(
            train_manifest,
            train_cache,
            train_index,
            spec,
            image_size,
            limit_train,
            rebuild_index=rebuild_index,
            expected_cache_metadata=train_cache_expected.get("metadata"),
            expected_cache_keys=set(train_cache_expected.get("keys", [])),
        )
        val_ds = RobotLatentWindowDataset(
            val_manifest,
            val_cache,
            val_index,
            spec,
            image_size,
            limit_val,
            rebuild_index=rebuild_index,
            expected_cache_metadata=val_cache_expected.get("metadata"),
            expected_cache_keys=set(val_cache_expected.get("keys", [])),
        )
    else:
        train_ds = RobotWindowDataset(train_manifest, train_index, spec, image_size, limit_train, rebuild_index=rebuild_index)
        val_ds = RobotWindowDataset(val_manifest, val_index, spec, image_size, limit_val, rebuild_index=rebuild_index)
    return train_ds, val_ds
