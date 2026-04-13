from __future__ import annotations

import argparse
from pathlib import Path

from videoworld2.robot_idm.eval.eval_closed_loop import evaluate_closed_loop
from videoworld2.robot_idm.eval.eval_offline_idm import evaluate_offline
from videoworld2.robot_idm.utils.config import load_config
from videoworld2.robot_idm.utils.runtime import resolve_device, save_json


def render_report(results: list[dict], report_path: Path) -> None:
    gt_success = None
    predicted_success = None
    best_offline = min(results, key=lambda item: item["offline"]["action_mse"])
    best_rollout = max(results, key=lambda item: item["closed_loop"]["rollout_success"])
    for item in results:
        if item["name"] == "history_gt":
            gt_success = item["closed_loop"]["rollout_success"]
        if item["name"] == "history_pred":
            predicted_success = item["closed_loop"]["rollout_success"]

    lines = [
        "# Initial VW2-IDM Ablation",
        "",
        "Dataset: mock static-camera smoke subset standing in for CALVIN/LIBERO because no real robot benchmark is available in this workspace.",
        "Evaluator: closed-loop chunked replanning with `execute_per_replan = 1` on the mock rollout environment.",
        "",
        "| Variant | Offline NLL | Offline MSE | Rollout Success | Jerk | Planner Acc |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in results:
        offline = item["offline"]
        closed = item["closed_loop"]
        lines.append(
            f"| {item['label']} | {offline['action_nll']:.4f} | {offline['action_mse']:.4f} | {closed['rollout_success']:.4f} | "
            f"{closed['jerk']:.4f} | {closed.get('planner_code_accuracy', 0.0):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- BC uses state-only decoding.",
            "- Pair-IDM uses GT future codes without past action history.",
            "- History-IDM uses GT future codes plus past action history.",
            "- History-IDM + predicted codes uses the frozen planner checkpoint.",
            f"- Best offline action MSE: {best_offline['label']} at {best_offline['offline']['action_mse']:.4f}.",
            f"- Best rollout success: {best_rollout['label']} at {best_rollout['closed_loop']['rollout_success']:.4f}.",
            "- This smoke run validates the training and evaluation stack end-to-end, but the closed-loop policy is still weak on the mock benchmark.",
        ]
    )
    if gt_success is not None and predicted_success is not None:
        lines.extend(
            [
                "",
                "## GT vs Pred Gap",
                "",
                f"- GT-code vs predicted-code success gap: {gt_success - predicted_success:.4f}",
            ]
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(args.device)
    config_dir = Path(cfg["_meta"]["config_path"]).parent
    results = []
    for experiment in cfg["ablation"]["experiments"]:
        exp_config_path = Path(experiment["config"])
        if not exp_config_path.is_absolute():
            exp_config_path = (config_dir / exp_config_path).resolve()
        exp_cfg = load_config(exp_config_path)
        checkpoint_path = Path(experiment["checkpoint"])
        if not checkpoint_path.is_absolute():
            checkpoint_path = (config_dir / checkpoint_path).resolve()
        if "checkpoint" in experiment:
            exp_cfg["idm"]["planner_checkpoint"] = experiment.get("planner_checkpoint", exp_cfg.get("idm", {}).get("planner_checkpoint"))
        offline = evaluate_offline(exp_cfg, checkpoint_path=str(checkpoint_path), device=device)
        closed_loop = evaluate_closed_loop(exp_cfg, checkpoint_path=str(checkpoint_path), device=device)
        results.append(
            {
                "name": experiment["name"],
                "label": experiment["label"],
                "offline": offline,
                "closed_loop": closed_loop,
            }
        )

    output_json = Path(cfg["ablation"]["output_json"])
    if not output_json.is_absolute():
        output_json = (Path(cfg["_meta"]["config_path"]).parent / output_json).resolve()
    save_json({"results": results}, output_json)

    report_path = Path(cfg["ablation"]["report_path"])
    if not report_path.is_absolute():
        report_path = (Path(cfg["_meta"]["config_path"]).parent / report_path).resolve()
    render_report(results, report_path)


if __name__ == "__main__":
    main()
