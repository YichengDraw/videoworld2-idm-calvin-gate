from __future__ import annotations

import argparse
from typing import Any

import torch

from videoworld2.robot_idm.eval.eval_offline_idm import load_policy_bundle
from videoworld2.robot_idm.models.dldm_local_adapter import DLDMLocalAdapter
from videoworld2.robot_idm.train.common import ensure_code_caches
from videoworld2.robot_idm.utils.config import load_config
from videoworld2.robot_idm.utils.factory import build_train_val_datasets, build_window_spec
from videoworld2.robot_idm.utils.metrics import jerk_metric, rollout_success
from videoworld2.robot_idm.utils.mock_data import oracle_action, render_mock_frame
from videoworld2.robot_idm.utils.runtime import configure_determinism, resolve_device, save_json


def _oracle_future_clip(sample: dict[str, Any], horizon: int) -> torch.Tensor:
    meta = sample["meta"]
    position = sample["proprio_hist"][-1, :2].clone()
    velocity = sample["proprio_hist"][-1, 2:4].clone()
    target = meta["target"].clone()
    swirl = float(meta["swirl"])
    embodiment_gain = float(meta["embodiment_gain"])
    dt = float(meta["dt"])
    action_scale = float(meta["action_scale"])
    image_size = int(meta["image_size"])
    frames = [render_mock_frame(position, target, image_size)]
    for _ in range(horizon):
        action = oracle_action(position, velocity, target, swirl, action_scale, embodiment_gain)
        velocity = action
        position = (position + dt * velocity).clamp(0.05, 0.95)
        frames.append(render_mock_frame(position, target, image_size))
    return torch.stack(frames)


@torch.no_grad()
def evaluate_closed_loop(cfg: dict[str, Any], checkpoint_path: str, device: torch.device | None = None) -> dict[str, float]:
    if cfg["data"].get("dataset_type") != "mock":
        raise ValueError("Closed-loop evaluation currently requires dataset_type=mock.")
    configure_determinism(int(cfg["training"].get("seed", 7)), deterministic=bool(cfg["training"].get("deterministic", True)))
    device = device or resolve_device("auto")
    adapter = DLDMLocalAdapter(cfg["adapter"]).to(device)
    ensure_code_caches(cfg, adapter=adapter, device=device)
    _, val_dataset = build_train_val_datasets(cfg, use_latent_cache=True)
    sample = val_dataset[0]
    cfg["data"]["action_dim"] = int(sample["action_chunk"].size(-1))
    adapter, state_encoder, idm, planner_encoder, planner, verifier = load_policy_bundle(cfg, checkpoint_path, device)

    max_rollouts = int(cfg.get("evaluation", {}).get("num_rollouts", 8))
    execute_per_replan = int(cfg.get("evaluation", {}).get("execute_per_replan", cfg["data"].get("action_chunk", 8) // 2))
    horizon = int(cfg.get("evaluation", {}).get("rollout_horizon", 16))
    code_source = cfg.get("idm", {}).get("code_source", "gt")
    use_verifier = bool(cfg.get("evaluation", {}).get("use_verifier", False))
    num_candidates = int(cfg.get("evaluation", {}).get("num_candidates", 4))
    if code_source == "predicted" and planner is None:
        raise ValueError("Predicted-code closed-loop evaluation requires a planner checkpoint.")

    successes = []
    jerk_scores = []
    planner_acc = []

    for idx in range(min(max_rollouts, len(val_dataset))):
        sample = val_dataset[idx]
        meta = sample["meta"]
        target = meta["target"].clone()
        swirl = float(meta["swirl"])
        embodiment_gain = float(meta["embodiment_gain"])
        dt = float(meta["dt"])
        action_scale = float(meta["action_scale"])
        image_size = int(meta["image_size"])

        obs_hist = sample["rgb_hist"].clone()
        proprio_hist = sample["proprio_hist"].clone()
        past_actions = sample["past_action_hist"].clone()
        position = proprio_hist[-1, :2].clone()
        velocity = proprio_hist[-1, 2:4].clone()
        executed_actions = []

        for _ in range(0, horizon, execute_per_replan):
            oracle_clip = _oracle_future_clip(sample | {"proprio_hist": proprio_hist, "rgb_hist": obs_hist}, build_window_spec(cfg).future_video_horizon)
            oracle_codes = adapter.encode_local_clip(oracle_clip.unsqueeze(0).to(device))["codes"]
            state_tokens, _ = state_encoder(
                rgb_hist=obs_hist.unsqueeze(0).to(device),
                proprio_hist=proprio_hist.unsqueeze(0).to(device),
                lang_texts=[sample["lang"]],
                embodiment_id=torch.tensor([sample["embodiment_id"]], device=device),
            )

            future_embeds = None
            target_codes = oracle_codes
            if cfg["idm"].get("use_future_codes", True):
                if code_source == "predicted" and planner is not None:
                    planning_tokens = state_tokens
                    if planner_encoder is not None:
                        planning_tokens, _ = planner_encoder(
                            rgb_hist=obs_hist.unsqueeze(0).to(device),
                            proprio_hist=proprio_hist.unsqueeze(0).to(device),
                            lang_texts=[sample["lang"]],
                            embodiment_id=torch.tensor([sample["embodiment_id"]], device=device),
                        )
                    predicted_codes = planner.sample(planning_tokens)
                    planner_acc.append(float((predicted_codes.cpu() == oracle_codes.cpu()).float().mean()))
                    future_embeds = adapter.code_embed(predicted_codes)
                    target_codes = predicted_codes
                else:
                    future_embeds = adapter.encode_local_clip(oracle_clip.unsqueeze(0).to(device))["embeds"]

            mean, log_std = idm(
                state_tokens=state_tokens,
                future_code_embeds=future_embeds,
                past_action_hist=past_actions.unsqueeze(0).to(device) if cfg["idm"].get("use_past_actions", True) else None,
                embodiment_id=torch.tensor([sample["embodiment_id"]], device=device),
            )
            action_chunk = mean[0]
            if use_verifier and verifier is not None and future_embeds is not None:
                candidates = []
                for _ in range(num_candidates):
                    candidates.append(mean + torch.randn_like(mean) * log_std.exp())
                candidate_actions = torch.stack(candidates, dim=1)
                reranked, _ = verifier.rerank(state_tokens, candidate_actions, target_codes)
                action_chunk = reranked[0]

            for action in action_chunk[:execute_per_replan].cpu():
                action = action.clamp(-action_scale, action_scale)
                executed_actions.append(action)
                velocity = action
                position = (position + dt * velocity).clamp(0.05, 0.95)
                new_frame = render_mock_frame(position, target, image_size)
                new_proprio = torch.cat([position, velocity])
                obs_hist = torch.cat([obs_hist[1:], new_frame.unsqueeze(0)], dim=0)
                proprio_hist = torch.cat([proprio_hist[1:], new_proprio.unsqueeze(0)], dim=0)
                past_actions = torch.cat([past_actions[1:], action.unsqueeze(0)], dim=0)

        executed_tensor = torch.stack(executed_actions).unsqueeze(0)
        successes.append(float(rollout_success(position.unsqueeze(0), target.unsqueeze(0)).cpu()))
        jerk_scores.append(float(jerk_metric(executed_tensor).cpu()))

    metrics = {
        "rollout_success": sum(successes) / max(len(successes), 1),
        "jerk": sum(jerk_scores) / max(len(jerk_scores), 1),
        "planner_code_accuracy": sum(planner_acc) / max(len(planner_acc), 1) if planner_acc else 0.0,
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    cfg = load_config(args.config)
    metrics = evaluate_closed_loop(cfg, checkpoint_path=args.checkpoint, device=resolve_device(args.device))
    print(metrics)
    if args.output_json:
        save_json(metrics, args.output_json)


if __name__ == "__main__":
    main()
