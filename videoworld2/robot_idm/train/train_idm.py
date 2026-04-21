from __future__ import annotations

import argparse
from pathlib import Path

import torch

from videoworld2.robot_idm.models.dldm_local_adapter import DLDMLocalAdapter
from videoworld2.robot_idm.train.common import (
    batch_to_state_encoder_inputs,
    ensure_code_caches,
    make_dataloaders,
    make_output_dir,
    maybe_resume_training,
    sample_code_conditioning,
)
from videoworld2.robot_idm.utils.checkpoint import load_checkpoint, save_checkpoint
from videoworld2.robot_idm.utils.config import load_config
from videoworld2.robot_idm.utils.factory import build_direct_policy, build_idm, build_planner, build_state_encoder
from videoworld2.robot_idm.utils.logging_utils import ExperimentLogger
from videoworld2.robot_idm.utils.metrics import action_mse, detach_metrics, discounted_gaussian_nll
from videoworld2.robot_idm.utils.runtime import configure_determinism, resolve_device, to_device


def load_planner_bundle(cfg, adapter, device: torch.device):
    planner_ckpt = cfg.get("idm", {}).get("planner_checkpoint")
    if not planner_ckpt:
        return None, None
    planner_ckpt = str((Path(cfg["_meta"]["config_path"]).parent / planner_ckpt).resolve()) if not Path(planner_ckpt).is_absolute() else planner_ckpt
    checkpoint = load_checkpoint(planner_ckpt, map_location=device)
    planner_encoder = build_state_encoder(cfg).to(device)
    planner = build_planner(cfg, adapter).to(device)
    planner_encoder.load_state_dict(checkpoint["state_encoder"], strict=False)
    planner.load_state_dict(checkpoint["planner"], strict=False)
    planner_encoder.eval()
    planner.eval()
    return planner_encoder, planner


def build_trainable_policy(cfg, action_dim: int, device: torch.device):
    if cfg.get("idm", {}).get("variant") == "bc" or cfg.get("policy", {}).get("variant") == "mlp":
        return build_direct_policy(cfg, action_dim=action_dim).to(device), "direct_policy"
    return build_idm(cfg, action_dim=action_dim).to(device), "idm"


def checkpoint_model_metadata(cfg, checkpoint_key: str, action_dim: int) -> dict:
    return {
        "checkpoint_key": checkpoint_key,
        "policy_variant": cfg.get("policy", {}).get("variant", ""),
        "idm_variant": cfg.get("idm", {}).get("variant", ""),
        "action_dim": int(action_dim),
        "action_chunk": int(cfg["data"].get("action_chunk", 8)),
        "model": cfg.get("model", {}),
        "data": {
            "use_proprio": bool(cfg["data"].get("use_proprio", True)),
            "use_lang": bool(cfg["data"].get("use_lang", True)),
            "proprio_dim": int(cfg["data"].get("proprio_dim", 4)),
        },
    }


def run_epoch(
    cfg,
    data_loader,
    state_encoder,
    idm,
    adapter,
    planner_encoder,
    planner,
    optimizer,
    device: torch.device,
    training: bool,
) -> dict[str, float]:
    state_encoder.train(training)
    idm.train(training)
    totals = {"action_nll": 0.0, "action_mse": 0.0, "planner_code_accuracy": 0.0}
    count = 0

    for batch in data_loader:
        batch = to_device(batch, device)
        state_inputs = batch_to_state_encoder_inputs(batch)
        state_tokens, _ = state_encoder(**state_inputs)

        planning_state_tokens = state_tokens
        if planner_encoder is not None:
            with torch.no_grad():
                planning_state_tokens, _ = planner_encoder(**state_inputs)

        future_code_embeds, planner_metrics = sample_code_conditioning(
            cfg=cfg,
            batch=batch,
            state_tokens=planning_state_tokens,
            adapter=adapter,
            planner=planner,
            training=training,
        )

        mean, log_std = idm(
            state_tokens=state_tokens,
            future_code_embeds=future_code_embeds,
            past_action_hist=batch["past_action_hist"] if cfg["idm"].get("use_past_actions", True) else None,
            embodiment_id=batch["embodiment_id"].long(),
        )
        nll = discounted_gaussian_nll(mean, log_std, batch["action_chunk"], gamma=float(cfg["training"].get("gamma_discount", 0.97)))
        mse = action_mse(mean, batch["action_chunk"])
        loss = nll + float(cfg["training"].get("mse_weight", 0.25)) * mse

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        batch_size = batch["action_chunk"].size(0)
        totals["action_nll"] += float(nll.detach().cpu()) * batch_size
        totals["action_mse"] += float(mse.detach().cpu()) * batch_size
        totals["planner_code_accuracy"] += float(planner_metrics.get("planner_code_accuracy", torch.tensor(0.0)).detach().cpu()) * batch_size
        count += batch_size

    return {key: value / max(count, 1) for key, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    cfg = load_config(args.config)
    configure_determinism(int(cfg["training"].get("seed", 7)), deterministic=bool(cfg["training"].get("deterministic", True)))
    device = resolve_device(args.device)
    adapter = DLDMLocalAdapter(cfg["adapter"]).to(device)
    ensure_code_caches(cfg, adapter=adapter, device=device)

    train_loader, val_loader = make_dataloaders(cfg, use_latent_cache=True)
    action_dim = train_loader.dataset[0]["action_chunk"].size(-1)
    state_encoder = build_state_encoder(cfg).to(device)
    idm, checkpoint_key = build_trainable_policy(cfg, action_dim=action_dim, device=device)
    planner_encoder, planner = load_planner_bundle(cfg, adapter, device)

    optimizer = torch.optim.AdamW(
        list(state_encoder.parameters()) + list(idm.parameters()),
        lr=float(cfg["training"].get("idm_lr", 3e-4)),
        weight_decay=float(cfg["training"].get("weight_decay", 1e-4)),
    )

    output_dir = make_output_dir(cfg)
    logger = ExperimentLogger(output_dir, cfg)
    start_epoch, global_step, best_metric = maybe_resume_training(
        output_dir,
        modules={"state_encoder": state_encoder, checkpoint_key: idm},
        optimizer=optimizer,
        explicit_resume=args.resume,
    )
    if best_metric == float("-inf"):
        best_metric = float("inf")

    max_epochs = int(cfg["training"].get("max_epochs", 3))
    for epoch in range(start_epoch, max_epochs):
        train_metrics = run_epoch(cfg, train_loader, state_encoder, idm, adapter, planner_encoder, planner, optimizer, device, training=True)
        val_metrics = run_epoch(cfg, val_loader, state_encoder, idm, adapter, planner_encoder, planner, optimizer, device, training=False)
        metric = val_metrics["action_nll"]
        is_best = metric < best_metric
        best_metric = min(best_metric, metric)
        logger.log(global_step + epoch + 1, {f"train/{k}": v for k, v in train_metrics.items()} | {f"val/{k}": v for k, v in val_metrics.items()})
        save_checkpoint(
            output_dir,
            {
                "config": cfg,
                "epoch": epoch + 1,
                "global_step": global_step + epoch + 1,
                "best_metric": best_metric,
                "state_encoder": state_encoder.state_dict(),
                checkpoint_key: idm.state_dict(),
                "model_kind": checkpoint_key,
                "model_metadata": checkpoint_model_metadata(cfg, checkpoint_key, action_dim),
                "optimizer": optimizer.state_dict(),
                "planner_checkpoint": cfg.get("idm", {}).get("planner_checkpoint"),
            },
            is_best=is_best,
        )
        print(detach_metrics({f"train/{k}": torch.tensor(v) for k, v in train_metrics.items()} | {f"val/{k}": torch.tensor(v) for k, v in val_metrics.items()}))

    logger.finish()


if __name__ == "__main__":
    main()
