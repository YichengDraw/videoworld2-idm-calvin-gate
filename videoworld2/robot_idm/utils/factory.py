from __future__ import annotations

from pathlib import Path
from typing import Any

from videoworld2.robot_idm.data.calvin_window_dataset import CalvinLatentWindowDataset, CalvinWindowDataset
from videoworld2.robot_idm.data.robot_latent_dataset import RobotLatentWindowDataset
from videoworld2.robot_idm.data.robot_window_dataset import RobotWindowDataset, WindowSpec
from videoworld2.robot_idm.models.dldm_local_adapter import DLDMLocalAdapter
from videoworld2.robot_idm.models.direct_policy import DirectPolicy, MLPActionHead
from videoworld2.robot_idm.models.forward_verifier import ForwardVerifier
from videoworld2.robot_idm.models.inverse_dynamics import HistoryAwareIDM
from videoworld2.robot_idm.models.latent_planner import LatentPlanner
from videoworld2.robot_idm.models.state_encoder import StateEncoder
from videoworld2.robot_idm.utils.runtime import load_json


def build_window_spec(cfg: dict[str, Any]) -> WindowSpec:
    return WindowSpec(
        history_frames=int(cfg["data"].get("history_frames", 4)),
        past_action_hist=int(cfg["data"].get("past_action_hist", 4)),
        future_video_horizon=int(cfg["data"].get("future_video_horizon", 8)),
        action_chunk=int(cfg["data"].get("action_chunk", 8)),
        stride=int(cfg["data"].get("stride", 4)),
    )


def _resolve_path(config_path: str, path_value: str) -> str:
    raw_path = str(path_value)
    candidate = Path(path_value)
    if raw_path.startswith("/") and not candidate.is_absolute():
        raise ValueError(f"Config path {raw_path} is a POSIX absolute path on this platform; remap it to a local path.")
    if candidate.is_absolute():
        return str(candidate)
    return str((Path(config_path).parent / candidate).resolve())


def validate_manifest_pair(train_manifest: str | Path, val_manifest: str | Path) -> None:
    train_path = Path(train_manifest).resolve()
    val_path = Path(val_manifest).resolve()
    if train_path == val_path:
        raise ValueError(f"Train and validation manifests must be distinct: {train_path}")

    train_entries = load_json(train_path).get("episodes", [])
    val_entries = load_json(val_path).get("episodes", [])
    _validate_unique_episode_ids(train_entries, "train")
    _validate_unique_episode_ids(val_entries, "val")
    train_ids = {entry.get("episode_id") for entry in train_entries}
    val_ids = {entry.get("episode_id") for entry in val_entries}
    overlap_ids = sorted(train_ids & val_ids)
    if overlap_ids:
        raise ValueError(f"Train/val manifests share episode ids: {overlap_ids[:5]}")

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


def _normalise_manifest_root(root_value: Any, manifest_dir: Path) -> str:
    raw_root = str(root_value)
    root_path = Path(raw_root).expanduser()
    if raw_root.startswith("/") and not root_path.is_absolute():
        return root_path.as_posix()
    if root_path.is_absolute():
        return root_path.resolve(strict=False).as_posix()
    return (manifest_dir / root_path).resolve(strict=False).as_posix()


def _manifest_spans(entries: list[dict[str, Any]], manifest_dir: Path) -> dict[str, list[tuple[int, int]]]:
    spans: dict[str, list[tuple[int, int]]] = {}
    for entry in entries:
        if "root" not in entry or "start" not in entry or "end" not in entry:
            continue
        root = _normalise_manifest_root(entry["root"], manifest_dir)
        spans.setdefault(root, []).append((int(entry["start"]), int(entry["end"])))
    return spans


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
    config_path = cfg["_meta"]["config_path"]
    image_size = int(data_cfg.get("image_size", 64))
    limit_train = data_cfg.get("limit_train_windows")
    limit_val = data_cfg.get("limit_val_windows")
    rebuild_index = bool(data_cfg.get("rebuild_index", data_cfg.get("overwrite_cache", False)))

    train_manifest = _resolve_path(config_path, data_cfg["train_manifest"])
    val_manifest = _resolve_path(config_path, data_cfg["val_manifest"])
    if bool(data_cfg.get("validate_split_disjoint", True)):
        validate_manifest_pair(train_manifest, val_manifest)
    train_index = _resolve_path(config_path, data_cfg["train_index"])
    val_index = _resolve_path(config_path, data_cfg["val_index"])
    dataset_type = data_cfg.get("dataset_type", "standard")

    if dataset_type == "calvin_static":
        if use_latent_cache:
            train_cache = _resolve_path(config_path, data_cfg["train_cache"])
            val_cache = _resolve_path(config_path, data_cfg["val_cache"])
            train_ds = CalvinLatentWindowDataset(train_manifest, train_cache, train_index, spec, image_size, limit_train, rebuild_index=rebuild_index)
            val_ds = CalvinLatentWindowDataset(val_manifest, val_cache, val_index, spec, image_size, limit_val, rebuild_index=rebuild_index)
        else:
            train_ds = CalvinWindowDataset(train_manifest, train_index, spec, image_size, limit_train, rebuild_index=rebuild_index)
            val_ds = CalvinWindowDataset(val_manifest, val_index, spec, image_size, limit_val, rebuild_index=rebuild_index)
    elif use_latent_cache:
        train_cache = _resolve_path(config_path, data_cfg["train_cache"])
        val_cache = _resolve_path(config_path, data_cfg["val_cache"])
        train_ds = RobotLatentWindowDataset(train_manifest, train_cache, train_index, spec, image_size, limit_train, rebuild_index=rebuild_index)
        val_ds = RobotLatentWindowDataset(val_manifest, val_cache, val_index, spec, image_size, limit_val, rebuild_index=rebuild_index)
    else:
        train_ds = RobotWindowDataset(train_manifest, train_index, spec, image_size, limit_train, rebuild_index=rebuild_index)
        val_ds = RobotWindowDataset(val_manifest, val_index, spec, image_size, limit_val, rebuild_index=rebuild_index)
    return train_ds, val_ds
