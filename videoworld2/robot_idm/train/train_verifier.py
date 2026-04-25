from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from videoworld2.robot_idm.models.dldm_local_adapter import DLDMLocalAdapter
from videoworld2.robot_idm.train.common import batch_to_state_encoder_inputs, ensure_code_caches, make_dataloaders, make_output_dir, maybe_resume_training
from videoworld2.robot_idm.utils.checkpoint import auxiliary_checkpoint_metadata, save_checkpoint
from videoworld2.robot_idm.utils.config import load_config
from videoworld2.robot_idm.utils.factory import build_state_encoder, build_verifier
from videoworld2.robot_idm.utils.logging_utils import ExperimentLogger
from videoworld2.robot_idm.utils.metrics import detach_metrics, planner_accuracy
from videoworld2.robot_idm.utils.runtime import configure_determinism, resolve_device, to_device


def run_epoch(data_loader, state_encoder, verifier, optimizer, device: torch.device, training: bool) -> dict[str, float]:
    state_encoder.train(training)
    verifier.train(training)
    totals = {"verifier_ce": 0.0, "verifier_code_accuracy": 0.0}
    count = 0
    for batch in data_loader:
        batch = to_device(batch, device)
        state_inputs = batch_to_state_encoder_inputs(batch)
        state_tokens, _ = state_encoder(**state_inputs)
        logits = verifier(state_tokens, batch["action_chunk"])
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), batch["future_codes"].reshape(-1))
        accuracy = planner_accuracy(logits, batch["future_codes"])

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        batch_size = batch["future_codes"].size(0)
        totals["verifier_ce"] += float(loss.detach().cpu()) * batch_size
        totals["verifier_code_accuracy"] += float(accuracy.detach().cpu()) * batch_size
        count += batch_size
    if count == 0:
        raise ValueError("Verifier training/evaluation epoch produced no samples.")
    return {key: value / count for key, value in totals.items()}


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
    verifier = build_verifier(cfg, adapter=adapter, action_dim=action_dim).to(device)
    optimizer = torch.optim.AdamW(
        list(state_encoder.parameters()) + list(verifier.parameters()),
        lr=float(cfg["training"].get("verifier_lr", 3e-4)),
        weight_decay=float(cfg["training"].get("weight_decay", 1e-4)),
    )

    output_dir = make_output_dir(cfg)
    logger = ExperimentLogger(output_dir, cfg)
    start_epoch, global_step, best_metric = maybe_resume_training(
        output_dir,
        modules={"state_encoder": state_encoder, "verifier": verifier},
        optimizer=optimizer,
        explicit_resume=args.resume,
        expected_metadata=auxiliary_checkpoint_metadata(cfg, "verifier", action_dim=action_dim),
    )

    max_epochs = int(cfg["training"].get("max_epochs", 3))
    for epoch in range(start_epoch, max_epochs):
        train_metrics = run_epoch(train_loader, state_encoder, verifier, optimizer, device, training=True)
        val_metrics = run_epoch(val_loader, state_encoder, verifier, optimizer, device, training=False)
        metric = val_metrics["verifier_code_accuracy"]
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
                "verifier": verifier.state_dict(),
                "model_metadata": auxiliary_checkpoint_metadata(cfg, "verifier", action_dim=action_dim),
                "optimizer": optimizer.state_dict(),
            },
            is_best=is_best,
        )
        print(detach_metrics({f"train/{k}": torch.tensor(v) for k, v in train_metrics.items()} | {f"val/{k}": torch.tensor(v) for k, v in val_metrics.items()}))

    logger.finish()


if __name__ == "__main__":
    main()
