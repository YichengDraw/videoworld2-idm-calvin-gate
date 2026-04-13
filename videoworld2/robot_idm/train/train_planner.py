from __future__ import annotations

import argparse
from typing import Any

import torch
import torch.nn.functional as F

from videoworld2.robot_idm.models.dldm_local_adapter import DLDMLocalAdapter
from videoworld2.robot_idm.train.common import (
    batch_to_state_encoder_inputs,
    ensure_code_caches,
    make_dataloaders,
    make_output_dir,
    maybe_resume_training,
)
from videoworld2.robot_idm.utils.checkpoint import save_checkpoint
from videoworld2.robot_idm.utils.config import load_config
from videoworld2.robot_idm.utils.factory import build_planner, build_state_encoder
from videoworld2.robot_idm.utils.logging_utils import ExperimentLogger
from videoworld2.robot_idm.utils.metrics import detach_metrics, planner_accuracy
from videoworld2.robot_idm.utils.runtime import resolve_device, seed_all, to_device


def run_epoch(
    data_loader,
    state_encoder,
    planner,
    optimizer,
    device: torch.device,
    training: bool,
) -> dict[str, float]:
    state_encoder.train(training)
    planner.train(training)
    totals = {"loss": 0.0, "planner_code_accuracy": 0.0}
    count = 0

    for batch in data_loader:
        batch = to_device(batch, device)
        state_inputs = batch_to_state_encoder_inputs(batch)
        state_tokens, _ = state_encoder(**state_inputs)
        logits = planner(state_tokens, target_codes=batch["future_codes"])
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), batch["future_codes"].reshape(-1))
        accuracy = planner_accuracy(logits, batch["future_codes"])

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        batch_size = batch["future_codes"].size(0)
        totals["loss"] += float(loss.detach().cpu()) * batch_size
        totals["planner_code_accuracy"] += float(accuracy.detach().cpu()) * batch_size
        count += batch_size

    return {key: value / max(count, 1) for key, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_all(int(cfg["training"].get("seed", 7)))
    device = resolve_device(args.device)
    adapter = DLDMLocalAdapter(cfg["adapter"]).to(device)
    ensure_code_caches(cfg, adapter=adapter, device=device)

    train_loader, val_loader = make_dataloaders(cfg, use_latent_cache=True)
    state_encoder = build_state_encoder(cfg).to(device)
    planner = build_planner(cfg, adapter).to(device)
    optimizer = torch.optim.AdamW(
        list(state_encoder.parameters()) + list(planner.parameters()),
        lr=float(cfg["training"].get("planner_lr", 3e-4)),
        weight_decay=float(cfg["training"].get("weight_decay", 1e-4)),
    )

    output_dir = make_output_dir(cfg)
    logger = ExperimentLogger(output_dir, cfg)
    start_epoch, global_step, best_metric = maybe_resume_training(
        output_dir,
        modules={"state_encoder": state_encoder, "planner": planner},
        optimizer=optimizer,
        explicit_resume=args.resume,
    )

    max_epochs = int(cfg["training"].get("max_epochs", 3))
    for epoch in range(start_epoch, max_epochs):
        train_metrics = run_epoch(train_loader, state_encoder, planner, optimizer, device, training=True)
        val_metrics = run_epoch(val_loader, state_encoder, planner, optimizer, device, training=False)
        metric = val_metrics["planner_code_accuracy"]
        is_best = metric > best_metric
        best_metric = max(best_metric, metric)
        logger.log(global_step + epoch + 1, {f"train/{k}": v for k, v in train_metrics.items()} | {f"val/{k}": v for k, v in val_metrics.items()})
        save_checkpoint(
            output_dir,
            {
                "config": cfg,
                "epoch": epoch + 1,
                "global_step": global_step + epoch + 1,
                "best_metric": best_metric,
                "state_encoder": state_encoder.state_dict(),
                "planner": planner.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            is_best=is_best,
        )
        print(detach_metrics({f"train/{k}": torch.tensor(v) for k, v in train_metrics.items()} | {f"val/{k}": torch.tensor(v) for k, v in val_metrics.items()}))

    logger.finish()


if __name__ == "__main__":
    main()
