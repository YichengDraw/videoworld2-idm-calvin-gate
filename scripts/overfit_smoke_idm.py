from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from videoworld2.robot_idm.eval.eval_closed_loop import evaluate_closed_loop
from videoworld2.robot_idm.eval.eval_offline_idm import evaluate_offline
from videoworld2.robot_idm.utils.config import load_config
from videoworld2.robot_idm.utils.phase0 import (
    prepare_phase0_overfit_cfg,
    resolve_best_checkpoint,
    summarize_metric_history,
    write_runtime_config,
)
from videoworld2.robot_idm.utils.runtime import resolve_device, save_json


def run_overfit(config_path: str, episodes: int, max_epochs: int, device: str) -> dict[str, object]:
    base_cfg = load_config(config_path)
    run_name = f"phase0_history_gt_overfit_e{episodes}"
    cfg = prepare_phase0_overfit_cfg(base_cfg, run_name=run_name, episodes=episodes, max_epochs=max_epochs)
    runtime_config_path = write_runtime_config(cfg, cfg["training"]["output_dir"])

    subprocess.run(
        [sys.executable, "-m", "videoworld2.robot_idm.train.train_idm", str(runtime_config_path), "--device", device],
        check=True,
    )

    best_checkpoint = resolve_best_checkpoint(cfg["training"]["output_dir"])
    runtime_cfg = load_config(runtime_config_path)
    device_obj = resolve_device(device)
    offline = evaluate_offline(runtime_cfg, checkpoint_path=str(best_checkpoint), device=device_obj)
    closed_loop = evaluate_closed_loop(runtime_cfg, checkpoint_path=str(best_checkpoint), device=device_obj)
    history = summarize_metric_history(cfg["training"]["output_dir"])
    summary = {
        "run_name": run_name,
        "runtime_config": str(runtime_config_path),
        "checkpoint": str(best_checkpoint),
        "offline": offline,
        "closed_loop": closed_loop,
        "history": history,
    }
    save_json(summary, Path(cfg["training"]["output_dir"]) / "phase0_summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/vw2_idm/exp_gt_code_idm.yaml")
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    summary = run_overfit(args.config, episodes=args.episodes, max_epochs=args.max_epochs, device=args.device)
    print(
        {
            "run_name": summary["run_name"],
            "best_train_action_mse": round(float(summary["history"].get("best_train_action_mse", -1.0)), 6),
            "best_val_action_mse": round(float(summary["history"].get("best_val_action_mse", -1.0)), 6),
            "closed_loop_success": round(float(summary["closed_loop"]["rollout_success"]), 4),
            "offline_action_mse": round(float(summary["offline"]["action_mse"]), 6),
        }
    )


if __name__ == "__main__":
    main()
