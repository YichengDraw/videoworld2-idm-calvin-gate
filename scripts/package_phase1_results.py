from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


METRIC_KEYS = ("action_nll", "action_mse", "jerk", "planner_code_accuracy")
FLAG_KEYS = (
    "privileged_future_codes",
    "deployable_without_future_labels",
    "conditioning",
    "result_role",
    "real_calvin_closed_loop",
    "fresh_local_regeneration_blocked",
    "metric_origin",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def package_metrics(metrics: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    packaged: dict[str, Any] = {}
    for controller, values in metrics.items():
        if controller not in metadata:
            raise ValueError(f"Missing controller metadata for {controller}")
        row = {key: float(values[key]) for key in METRIC_KEYS}
        row.update(metadata[controller])
        row["result_role"] = "privileged_upper_bound" if row["privileged_future_codes"] else "deployable_policy"
        row["real_calvin_closed_loop"] = False
        row["fresh_local_regeneration_blocked"] = True
        row["metric_origin"] = "rescued_offline_summary"
        packaged[controller] = row
    extra = sorted(set(metadata) - set(metrics))
    if extra:
        raise ValueError(f"Controller metadata has no matching metrics: {extra[:5]}")
    return packaged


def write_csv(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ("controller",) + METRIC_KEYS + FLAG_KEYS
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for controller, values in metrics.items():
            row = {"controller": controller}
            for key in fieldnames:
                if key == "controller":
                    continue
                value = values[key]
                row[key] = str(value).lower() if isinstance(value, bool) else value
            writer.writerow(row)


def provenance() -> dict[str, Any]:
    return {
        "not_a_real_calvin_closed_loop_verdict": True,
        "fresh_local_regeneration_blocked": True,
        "safe_interpretation": "offline recovered metrics only; GT-code rows are privileged upper-bound diagnostics; no predicted-code claim is published",
        "phase1_offline_metrics": {
            "table_files": [
                "results/phase1_offline_metrics.json",
                "results/phase1_offline_metrics.csv",
            ],
            "status": "committed offline summary recovered from the rescued Phase 1 run",
            "single_file_safety": "metrics JSON/CSV duplicate privileged_future_codes, deployable_without_future_labels, conditioning, result_role, real_calvin_closed_loop, fresh_local_regeneration_blocked, and metric_origin flags",
            "not_a_real_calvin_closed_loop_verdict": True,
            "fresh_local_regeneration_blocked": True,
            "blockers": [
                "raw CALVIN RGB/proprio/action frame files are not committed",
                "rescued manifests reference the original remote Linux roots",
                "the locally observed ldm_tokenizer_training_init_weights.pt does not cover all official tokenizer encode-path parameters",
                "rescued latent caches predate the current strict metadata guard",
            ],
            "safe_interpretation": "offline recovered metrics only; GT-code rows are privileged upper-bound diagnostics; no predicted-code claim is published",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate the committed Phase 1 result table schema and provenance flags.")
    parser.add_argument("--metrics-json", type=Path, default=Path("results/phase1_offline_metrics.json"))
    parser.add_argument("--controller-metadata", type=Path, default=Path("results/phase1_controller_metadata.json"))
    parser.add_argument("--output-json", type=Path, default=Path("results/phase1_offline_metrics.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("results/phase1_offline_metrics.csv"))
    parser.add_argument("--provenance-json", type=Path, default=Path("results/phase1_result_provenance.json"))
    args = parser.parse_args()

    packaged = package_metrics(load_json(args.metrics_json), load_json(args.controller_metadata))
    write_json(args.output_json, packaged)
    write_csv(args.output_csv, packaged)
    write_json(args.provenance_json, provenance())


if __name__ == "__main__":
    main()
