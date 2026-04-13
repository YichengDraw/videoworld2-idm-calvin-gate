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


def build_window_spec(cfg: dict[str, Any]) -> WindowSpec:
    return WindowSpec(
        history_frames=int(cfg["data"].get("history_frames", 4)),
        past_action_hist=int(cfg["data"].get("past_action_hist", 4)),
        future_video_horizon=int(cfg["data"].get("future_video_horizon", 8)),
        action_chunk=int(cfg["data"].get("action_chunk", 8)),
        stride=int(cfg["data"].get("stride", 4)),
    )


def _resolve_path(config_path: str, path_value: str) -> str:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return str(candidate)
    return str((Path(config_path).parent / candidate).resolve())


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

    train_manifest = _resolve_path(config_path, data_cfg["train_manifest"])
    val_manifest = _resolve_path(config_path, data_cfg["val_manifest"])
    train_index = _resolve_path(config_path, data_cfg["train_index"])
    val_index = _resolve_path(config_path, data_cfg["val_index"])
    dataset_type = data_cfg.get("dataset_type", "standard")

    if dataset_type == "calvin_static":
        if use_latent_cache:
            train_cache = _resolve_path(config_path, data_cfg["train_cache"])
            val_cache = _resolve_path(config_path, data_cfg["val_cache"])
            train_ds = CalvinLatentWindowDataset(train_manifest, train_cache, train_index, spec, image_size, limit_train)
            val_ds = CalvinLatentWindowDataset(val_manifest, val_cache, val_index, spec, image_size, limit_val)
        else:
            train_ds = CalvinWindowDataset(train_manifest, train_index, spec, image_size, limit_train)
            val_ds = CalvinWindowDataset(val_manifest, val_index, spec, image_size, limit_val)
    elif use_latent_cache:
        train_cache = _resolve_path(config_path, data_cfg["train_cache"])
        val_cache = _resolve_path(config_path, data_cfg["val_cache"])
        train_ds = RobotLatentWindowDataset(train_manifest, train_cache, train_index, spec, image_size, limit_train)
        val_ds = RobotLatentWindowDataset(val_manifest, val_cache, val_index, spec, image_size, limit_val)
    else:
        train_ds = RobotWindowDataset(train_manifest, train_index, spec, image_size, limit_train)
        val_ds = RobotWindowDataset(val_manifest, val_index, spec, image_size, limit_val)
    return train_ds, val_ds
