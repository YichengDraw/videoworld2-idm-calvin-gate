from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from videoworld2.robot_idm.eval.eval_offline_idm import load_policy_bundle
from videoworld2.robot_idm.models.dldm_local_adapter import DLDMLocalAdapter
from videoworld2.robot_idm.train.common import ensure_code_caches
from videoworld2.robot_idm.utils.config import load_config
from videoworld2.robot_idm.utils.factory import build_train_val_datasets, build_window_spec
from videoworld2.robot_idm.utils.metrics import ensure_finite_metrics, rollout_success
from videoworld2.robot_idm.utils.runtime import resolve_device, save_json


def _oracle_future_clip(sample: dict, horizon: int) -> torch.Tensor:
    from videoworld2.robot_idm.eval.eval_closed_loop import _oracle_future_clip as oracle_future_clip

    return oracle_future_clip(sample, horizon)


@torch.no_grad()
def collect_debug_stats(
    cfg: dict,
    checkpoint_path: str,
    split: str,
    max_rollouts: int,
    device: torch.device,
) -> dict[str, object]:
    adapter = DLDMLocalAdapter(cfg["adapter"]).to(device)
    ensure_code_caches(cfg, adapter=adapter, device=device)
    train_dataset, val_dataset = build_train_val_datasets(cfg, use_latent_cache=True)
    dataset = train_dataset if split == "train" else val_dataset
    sample = dataset[0]
    cfg["data"]["action_dim"] = int(sample["action_chunk"].size(-1))
    adapter, state_encoder, idm, planner_encoder, planner, verifier_encoder, verifier = load_policy_bundle(cfg, checkpoint_path, device)
    if cfg["idm"].get("code_source", "gt") == "predicted" and planner is None:
        raise ValueError("Predicted-code debug stats require a planner checkpoint.")

    data_actions = []
    rollout_actions = []
    first_action_errors = []
    clip_count = 0
    clip_total = 0
    successes = []
    planner_acc = []

    horizon = int(cfg.get("evaluation", {}).get("rollout_horizon", 16))
    execute_per_replan = int(cfg.get("evaluation", {}).get("execute_per_replan", cfg["data"].get("action_chunk", 8) // 2))
    if max_rollouts <= 0:
        raise ValueError("Debug rollout stats require max_rollouts > 0.")
    if execute_per_replan <= 0:
        raise ValueError("Debug rollout stats require execute_per_replan > 0.")
    if horizon <= 0:
        raise ValueError("Debug rollout stats require rollout_horizon > 0.")
    future_horizon = build_window_spec(cfg).future_video_horizon
    progress_curves = []

    for idx in range(min(max_rollouts, len(dataset))):
        sample = dataset[idx]
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
        start_distance = float((position - target).norm().clamp_min(1e-6))
        sample_progress = []
        first_action_error = None

        data_actions.append(sample["action_chunk"])

        while len(sample_progress) < horizon:
            oracle_clip = _oracle_future_clip(sample | {"proprio_hist": proprio_hist, "rgb_hist": obs_hist}, future_horizon)
            oracle_codes = adapter.encode_local_clip(oracle_clip.unsqueeze(0).to(device))["codes"]
            state_tokens, _ = state_encoder(
                rgb_hist=obs_hist.unsqueeze(0).to(device),
                proprio_hist=proprio_hist.unsqueeze(0).to(device),
                lang_texts=[sample["lang"]],
                embodiment_id=torch.tensor([sample["embodiment_id"]], device=device),
            )

            future_embeds = None
            if cfg["idm"].get("use_future_codes", True):
                if cfg["idm"].get("code_source", "gt") == "predicted" and planner is not None:
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
                else:
                    future_embeds = adapter.encode_local_clip(oracle_clip.unsqueeze(0).to(device))["embeds"]

            mean, log_std = idm(
                state_tokens=state_tokens,
                future_code_embeds=future_embeds,
                past_action_hist=past_actions.unsqueeze(0).to(device) if cfg["idm"].get("use_past_actions", True) else None,
                embodiment_id=torch.tensor([sample["embodiment_id"]], device=device),
            )
            predicted = mean[0]
            clip_count += int((predicted.abs() > action_scale).sum().item())
            clip_total += int(predicted.numel())
            executed_chunk = predicted.clamp(-action_scale, action_scale)
            if first_action_error is None:
                first_action_error = float(torch.mean((executed_chunk[0].cpu() - sample["action_chunk"][0]) ** 2).item())

            remaining_horizon = horizon - len(sample_progress)
            for action in executed_chunk[: min(execute_per_replan, remaining_horizon)].cpu():
                rollout_actions.append(action)
                velocity = action
                position = (position + dt * velocity).clamp(0.05, 0.95)
                from videoworld2.robot_idm.utils.mock_data import render_mock_frame

                new_frame = render_mock_frame(position, target, image_size)
                new_proprio = torch.cat([position, velocity])
                obs_hist = torch.cat([obs_hist[1:], new_frame.unsqueeze(0)], dim=0)
                proprio_hist = torch.cat([proprio_hist[1:], new_proprio.unsqueeze(0)], dim=0)
                past_actions = torch.cat([past_actions[1:], action.unsqueeze(0)], dim=0)
                current_distance = float((position - target).norm())
                sample_progress.append((start_distance - current_distance) / start_distance)

        successes.append(float(rollout_success(position.unsqueeze(0), target.unsqueeze(0)).item()))
        if first_action_error is not None:
            first_action_errors.append(first_action_error)
        progress_curves.append(sample_progress)

    if not successes:
        raise ValueError("Debug rollout stats produced no rollouts.")
    dataset_actions = torch.cat([chunk.reshape(-1, chunk.size(-1)) for chunk in data_actions], dim=0)
    if not rollout_actions:
        raise ValueError("Debug rollout stats produced no executed actions.")
    rollout_actions_tensor = torch.stack(rollout_actions)
    if clip_total == 0:
        raise ValueError("Debug rollout stats produced no actions for clip statistics.")
    max_steps = max((len(curve) for curve in progress_curves), default=0)
    mean_progress = []
    for step in range(max_steps):
        values = [curve[step] for curve in progress_curves if step < len(curve)]
        mean_progress.append(sum(values) / len(values))

    return ensure_finite_metrics({
        "split": split,
        "episodes": min(max_rollouts, len(dataset)),
        "dataset_action_mean": dataset_actions.mean(dim=0).tolist(),
        "dataset_action_std": dataset_actions.std(dim=0, unbiased=False).tolist(),
        "rollout_action_mean": rollout_actions_tensor.mean(dim=0).tolist(),
        "rollout_action_std": rollout_actions_tensor.std(dim=0, unbiased=False).tolist(),
        "first_action_mse": sum(first_action_errors) / len(first_action_errors),
        "clip_count": clip_count,
        "clip_fraction": clip_count / clip_total,
        "mean_progress_per_step": mean_progress,
        "rollout_success": sum(successes) / len(successes),
        "planner_code_accuracy": sum(planner_acc) / len(planner_acc) if planner_acc else None,
    }, context="debug action stats")


def _write_plot(stats: dict[str, object], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    dims = list(range(len(stats["dataset_action_mean"])))
    axes[0].plot(dims, stats["dataset_action_mean"], label="dataset_mean")
    axes[0].plot(dims, stats["rollout_action_mean"], label="rollout_mean")
    axes[0].fill_between(
        dims,
        [m - s for m, s in zip(stats["dataset_action_mean"], stats["dataset_action_std"])],
        [m + s for m, s in zip(stats["dataset_action_mean"], stats["dataset_action_std"])],
        alpha=0.2,
    )
    axes[0].fill_between(
        dims,
        [m - s for m, s in zip(stats["rollout_action_mean"], stats["rollout_action_std"])],
        [m + s for m, s in zip(stats["rollout_action_mean"], stats["rollout_action_std"])],
        alpha=0.2,
    )
    axes[0].set_title("Action Stats")
    axes[0].set_xlabel("Action dim")
    axes[0].legend()

    axes[1].plot(stats["mean_progress_per_step"])
    axes[1].set_title("Mean Task Progress")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Normalized progress")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=("train", "val"))
    parser.add_argument("--max-rollouts", type=int, default=8)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--output-plot", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    cfg = load_config(args.config)
    stats = collect_debug_stats(
        cfg,
        checkpoint_path=args.checkpoint,
        split=args.split,
        max_rollouts=args.max_rollouts,
        device=resolve_device(args.device),
    )
    print(
        {
            "split": stats["split"],
            "episodes": stats["episodes"],
            "first_action_mse": round(float(stats["first_action_mse"]), 6),
            "clip_fraction": round(float(stats["clip_fraction"]), 6),
            "rollout_success": round(float(stats["rollout_success"]), 4),
            "planner_code_accuracy": "n/a" if stats["planner_code_accuracy"] is None else round(float(stats["planner_code_accuracy"]), 4),
        }
    )
    if args.output_json:
        save_json(stats, args.output_json)
    if args.output_plot:
        _write_plot(stats, Path(args.output_plot))


if __name__ == "__main__":
    main()
