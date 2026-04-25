from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from videoworld2.robot_idm.models.dldm_local_adapter import DLDMLocalAdapter
from videoworld2.robot_idm.train.common import (
    batch_to_state_encoder_inputs,
    ensure_code_caches,
    make_dataloaders,
    make_output_dir,
    maybe_resume_training,
    resolve_config_path,
    sample_code_conditioning,
)
from videoworld2.robot_idm.train.train_idm import load_planner_bundle
from videoworld2.robot_idm.utils.checkpoint import (
    checkpoint_reference,
    load_checkpoint,
    policy_checkpoint_metadata,
    save_checkpoint,
    teacher_policy_checkpoint_metadata,
    validate_checkpoint_metadata,
)
from videoworld2.robot_idm.utils.config import load_config
from videoworld2.robot_idm.utils.factory import build_direct_policy, build_state_encoder
from videoworld2.robot_idm.utils.logging_utils import ExperimentLogger
from videoworld2.robot_idm.utils.metrics import action_mse, detach_metrics, discounted_gaussian_nll
from videoworld2.robot_idm.utils.runtime import configure_determinism, resolve_device, to_device


def distill_teacher_metadata(cfg: dict, action_dim: int) -> dict:
    checkpoint_refs = {}
    idm_cfg = cfg.get("idm", {})
    planner_checkpoint = idm_cfg.get("planner_checkpoint")
    if planner_checkpoint and (
        idm_cfg.get("code_source", "gt") == "predicted"
        or (bool(idm_cfg.get("mixed_code_training", False)) and float(idm_cfg.get("mixed_code_ratio_pred", 0.0)) > 0.0)
    ):
        checkpoint_refs["planner_checkpoint"] = checkpoint_reference(
            resolve_config_path(cfg, planner_checkpoint, key_path="idm.planner_checkpoint")
        )
    return teacher_policy_checkpoint_metadata(cfg, action_dim, checkpoint_refs=checkpoint_refs or None)


def teacher_future_code_embeds(cfg: dict, batch: dict, teacher_tokens: torch.Tensor, adapter, teacher_planner):
    future_embeds, _ = sample_code_conditioning(
        cfg=cfg,
        batch=batch,
        state_tokens=teacher_tokens,
        adapter=adapter,
        planner=teacher_planner,
        training=False,
    )
    return future_embeds


def run_distillation_epoch(
    cfg: dict,
    data_loader,
    state_encoder,
    student,
    optimizer,
    device: torch.device,
    training: bool,
    adapter=None,
    teacher_encoder=None,
    teacher_idm=None,
    teacher_planner_encoder=None,
    teacher_planner=None,
) -> dict[str, float]:
    state_encoder.train(training)
    student.train(training)
    totals = {"action_nll": 0.0, "action_mse": 0.0}
    count = 0
    for batch in data_loader:
        batch = to_device(batch, device)
        state_inputs = batch_to_state_encoder_inputs(batch)
        state_tokens, _ = state_encoder(**state_inputs)
        mean, log_std = student(
            state_tokens=state_tokens,
            embodiment_id=batch["embodiment_id"].long(),
            past_action_hist=batch["past_action_hist"],
        )
        nll = discounted_gaussian_nll(mean, log_std, batch["action_chunk"], gamma=float(cfg["training"].get("gamma_discount", 0.97)))
        mse = action_mse(mean, batch["action_chunk"])
        loss = nll + float(cfg.get("distill", {}).get("lambda_bc", 1.0)) * mse
        if teacher_encoder is not None and teacher_idm is not None:
            with torch.no_grad():
                teacher_tokens, _ = teacher_encoder(**state_inputs)
                planning_tokens = teacher_tokens
                if teacher_planner_encoder is not None:
                    planning_tokens, _ = teacher_planner_encoder(**state_inputs)
                future_code_embeds = teacher_future_code_embeds(cfg, batch, planning_tokens, adapter, teacher_planner)
                teacher_mean, _ = teacher_idm(
                    state_tokens=teacher_tokens,
                    future_code_embeds=future_code_embeds if cfg["idm"].get("use_future_codes", True) else None,
                    past_action_hist=batch["past_action_hist"] if cfg["idm"].get("use_past_actions", True) else None,
                    embodiment_id=batch["embodiment_id"].long(),
                )
            loss = loss + float(cfg.get("distill", {}).get("lambda_kl", 0.5)) * F.mse_loss(mean, teacher_mean)

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        batch_size = batch["action_chunk"].size(0)
        totals["action_nll"] += float(nll.detach().cpu()) * batch_size
        totals["action_mse"] += float(mse.detach().cpu()) * batch_size
        count += batch_size
    if count == 0:
        raise ValueError("Distillation epoch produced no samples.")
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
    cfg["data"]["action_dim"] = int(action_dim)

    teacher_checkpoint = cfg.get("distill", {}).get("teacher_checkpoint")
    teacher_encoder = None
    teacher_idm = None
    teacher_planner_encoder = None
    teacher_planner = None
    if teacher_checkpoint:
        teacher_metadata = distill_teacher_metadata(cfg, action_dim)
        teacher_planner_encoder, teacher_planner = load_planner_bundle(cfg, adapter, device)
        teacher_checkpoint = str(resolve_config_path(cfg, teacher_checkpoint, key_path="distill.teacher_checkpoint"))
        teacher_state = load_checkpoint(teacher_checkpoint, map_location=device)
        validate_checkpoint_metadata(teacher_state, teacher_metadata, teacher_checkpoint)
        from videoworld2.robot_idm.utils.factory import build_idm

        teacher_encoder = build_state_encoder(cfg).to(device)
        teacher_idm = build_idm(cfg, action_dim=action_dim).to(device)
        teacher_encoder.load_state_dict(teacher_state["state_encoder"], strict=True)
        teacher_idm.load_state_dict(teacher_state["idm"], strict=True)
        teacher_encoder.eval()
        teacher_idm.eval()
    student_checkpoint_refs = {}
    if teacher_checkpoint:
        student_checkpoint_refs["teacher_checkpoint"] = checkpoint_reference(teacher_checkpoint)
    state_encoder = build_state_encoder(cfg).to(device)
    student = build_direct_policy(cfg, action_dim=action_dim).to(device)
    optimizer = torch.optim.AdamW(
        list(state_encoder.parameters()) + list(student.parameters()),
        lr=float(cfg["training"].get("idm_lr", 3e-4)),
        weight_decay=float(cfg["training"].get("weight_decay", 1e-4)),
    )
    output_dir = make_output_dir(cfg)
    logger = ExperimentLogger(output_dir, cfg)
    start_epoch, global_step, best_metric = maybe_resume_training(
        output_dir,
        modules={"state_encoder": state_encoder, "direct_policy": student},
        optimizer=optimizer,
        explicit_resume=args.resume,
        expected_metadata=policy_checkpoint_metadata(cfg, "direct_policy", action_dim, checkpoint_refs=student_checkpoint_refs or None),
    )
    if best_metric == float("-inf"):
        best_metric = float("inf")

    max_epochs = int(cfg["training"].get("max_epochs", 3))
    for epoch in range(start_epoch, max_epochs):
        train_metrics = run_distillation_epoch(
            cfg,
            train_loader,
            state_encoder,
            student,
            optimizer,
            device,
            training=True,
            adapter=adapter,
            teacher_encoder=teacher_encoder,
            teacher_idm=teacher_idm,
            teacher_planner_encoder=teacher_planner_encoder,
            teacher_planner=teacher_planner,
        )
        val_metrics = run_distillation_epoch(
            cfg,
            val_loader,
            state_encoder,
            student,
            optimizer,
            device,
            training=False,
            adapter=adapter,
            teacher_encoder=teacher_encoder,
            teacher_idm=teacher_idm,
            teacher_planner_encoder=teacher_planner_encoder,
            teacher_planner=teacher_planner,
        )
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
                "direct_policy": student.state_dict(),
                "model_kind": "direct_policy",
                "model_metadata": policy_checkpoint_metadata(cfg, "direct_policy", action_dim, checkpoint_refs=student_checkpoint_refs or None),
                "optimizer": optimizer.state_dict(),
            },
            is_best=is_best,
        )
        print(detach_metrics({f"train/{k}": torch.tensor(v) for k, v in train_metrics.items()} | {f"val/{k}": torch.tensor(v) for k, v in val_metrics.items()}))

    logger.finish()


if __name__ == "__main__":
    main()
