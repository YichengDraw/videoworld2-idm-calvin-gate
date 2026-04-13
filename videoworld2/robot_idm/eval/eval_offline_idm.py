from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from videoworld2.robot_idm.models.dldm_local_adapter import DLDMLocalAdapter
from videoworld2.robot_idm.models.forward_verifier import ForwardVerifier
from videoworld2.robot_idm.models.inverse_dynamics import HistoryAwareIDM
from videoworld2.robot_idm.models.latent_planner import LatentPlanner
from videoworld2.robot_idm.models.state_encoder import StateEncoder
from videoworld2.robot_idm.train.common import batch_to_state_encoder_inputs, ensure_code_caches, make_dataloaders
from videoworld2.robot_idm.utils.checkpoint import load_checkpoint
from videoworld2.robot_idm.utils.config import load_config
from videoworld2.robot_idm.utils.factory import build_direct_policy, build_idm, build_planner, build_state_encoder, build_verifier
from videoworld2.robot_idm.utils.metrics import action_mse, detach_metrics, discounted_gaussian_nll, jerk_metric
from videoworld2.robot_idm.utils.runtime import resolve_device, save_json, to_device


def load_policy_bundle(cfg: dict[str, Any], checkpoint_path: str, device: torch.device):
    adapter = DLDMLocalAdapter(cfg["adapter"]).to(device)
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    state_encoder = build_state_encoder(cfg).to(device)
    action_dim = int(cfg["data"].get("action_dim", 2))
    use_predicted_codes = cfg.get("idm", {}).get("code_source", "gt") == "predicted"
    if "direct_policy" in checkpoint:
        idm = build_direct_policy(cfg, action_dim=action_dim).to(device)
        idm.load_state_dict(checkpoint["direct_policy"], strict=False)
    else:
        idm = build_idm(cfg, action_dim=action_dim).to(device)
        idm.load_state_dict(checkpoint["idm"], strict=False)
    state_encoder.load_state_dict(checkpoint["state_encoder"], strict=False)
    state_encoder.eval()
    idm.eval()

    planner_encoder = None
    planner = None
    planner_checkpoint = cfg.get("idm", {}).get("planner_checkpoint")
    if planner_checkpoint and use_predicted_codes:
        if not Path(planner_checkpoint).is_absolute():
            planner_checkpoint = str((Path(cfg["_meta"]["config_path"]).parent / planner_checkpoint).resolve())
        planner_state = load_checkpoint(planner_checkpoint, map_location=device)
        planner_encoder = build_state_encoder(cfg).to(device)
        planner = build_planner(cfg, adapter).to(device)
        planner_encoder.load_state_dict(planner_state["state_encoder"], strict=False)
        planner.load_state_dict(planner_state["planner"], strict=False)
        planner_encoder.eval()
        planner.eval()

    verifier = None
    verifier_checkpoint = cfg.get("evaluation", {}).get("verifier_checkpoint")
    if verifier_checkpoint:
        if not Path(verifier_checkpoint).is_absolute():
            verifier_checkpoint = str((Path(cfg["_meta"]["config_path"]).parent / verifier_checkpoint).resolve())
        verifier_state = load_checkpoint(verifier_checkpoint, map_location=device)
        verifier = build_verifier(cfg, adapter, action_dim=action_dim).to(device)
        verifier.load_state_dict(verifier_state["verifier"], strict=False)
        verifier.eval()
    return adapter, state_encoder, idm, planner_encoder, planner, verifier


@torch.no_grad()
def evaluate_offline(cfg: dict[str, Any], checkpoint_path: str, device: torch.device | None = None) -> dict[str, float]:
    device = device or resolve_device("auto")
    ensure_code_caches(cfg, adapter=DLDMLocalAdapter(cfg["adapter"]).to(device), device=device)
    _, val_loader = make_dataloaders(cfg, use_latent_cache=True)
    sample_batch = next(iter(val_loader))
    cfg["data"]["action_dim"] = int(sample_batch["action_chunk"].size(-1))
    adapter, state_encoder, idm, planner_encoder, planner, verifier = load_policy_bundle(cfg, checkpoint_path, device)

    totals = {"action_nll": 0.0, "action_mse": 0.0, "jerk": 0.0, "planner_code_accuracy": 0.0}
    count = 0
    use_predicted_codes = cfg.get("idm", {}).get("code_source", "gt") == "predicted"
    if use_predicted_codes and planner is None:
        raise ValueError("Predicted-code offline evaluation requires a planner checkpoint.")

    for batch in val_loader:
        batch = to_device(batch, device)
        state_inputs = batch_to_state_encoder_inputs(batch)
        state_tokens, _ = state_encoder(**state_inputs)
        planning_tokens = state_tokens
        if planner_encoder is not None:
            planning_tokens, _ = planner_encoder(**state_inputs)

        future_embeds = batch["future_code_embeds"]
        target_codes = batch["future_codes"]
        if use_predicted_codes and planner is not None:
            predicted_codes = planner.sample(planning_tokens)
            totals["planner_code_accuracy"] += float((predicted_codes == batch["future_codes"]).float().mean().detach().cpu()) * batch["action_chunk"].size(0)
            future_embeds = adapter.code_embed(predicted_codes)
            target_codes = predicted_codes

        mean, log_std = idm(
            state_tokens=state_tokens,
            future_code_embeds=future_embeds if cfg["idm"].get("use_future_codes", True) else None,
            past_action_hist=batch["past_action_hist"] if cfg["idm"].get("use_past_actions", True) else None,
            embodiment_id=batch["embodiment_id"].long(),
        )

        if verifier is not None and cfg.get("evaluation", {}).get("use_verifier", False):
            samples = []
            for _ in range(int(cfg["evaluation"].get("num_candidates", 4))):
                samples.append(mean + torch.randn_like(mean) * log_std.exp())
            candidate_actions = torch.stack(samples, dim=1)
            reranked, _ = verifier.rerank(state_tokens, candidate_actions, target_codes)
            mean = reranked
            log_std = torch.zeros_like(reranked)

        nll = discounted_gaussian_nll(mean, log_std, batch["action_chunk"], gamma=float(cfg["training"].get("gamma_discount", 0.97)))
        mse = action_mse(mean, batch["action_chunk"])
        jerk = jerk_metric(mean)
        batch_size = batch["action_chunk"].size(0)
        totals["action_nll"] += float(nll.detach().cpu()) * batch_size
        totals["action_mse"] += float(mse.detach().cpu()) * batch_size
        totals["jerk"] += float(jerk.detach().cpu()) * batch_size
        count += batch_size

    metrics = {key: value / max(count, 1) for key, value in totals.items()}
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    cfg = load_config(args.config)
    metrics = evaluate_offline(cfg, checkpoint_path=args.checkpoint, device=resolve_device(args.device))
    print(detach_metrics({key: torch.tensor(value) for key, value in metrics.items()}))
    if args.output_json:
        save_json(metrics, args.output_json)


if __name__ == "__main__":
    main()
