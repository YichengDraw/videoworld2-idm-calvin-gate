from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


NUMERIC_METRIC_KEYS = ("action_nll", "action_mse", "jerk")
OPTIONAL_METRIC_KEYS = ("planner_code_accuracy",)
METRIC_KEYS = NUMERIC_METRIC_KEYS + OPTIONAL_METRIC_KEYS
FLAG_KEYS = (
    "privileged_future_codes",
    "deployable_without_future_labels",
    "conditioning",
    "result_role",
    "real_calvin_closed_loop",
    "fresh_local_regeneration_blocked",
    "metric_origin",
)
RESCUED_OFFLINE_EVALS = {
    "bc_vis": "bc_vis_calvin_4090/offline_eval.json",
    "bc_vis_proprio": "bc_vis_proprio_calvin_4090/offline_eval.json",
    "pair_idm_gtcode": "pair_idm_calvin_4090/offline_eval.json",
    "history_idm_gtcode": "history_gt_calvin_4090/offline_eval.json",
    "vw2_hidden_mlp_action_head": "vw2_hidden_mlp_action_head_calvin_4090/offline_eval.json",
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _audit_hashes(audit_path: Path) -> dict[str, str]:
    audit = load_json(audit_path)
    source_files = audit.get("source_files", [])
    hashes: dict[str, str] = {}
    for record in source_files:
        path = str(record.get("path", ""))
        if path.endswith("/offline_eval.json"):
            hashes[path.removeprefix("models/")] = str(record["sha256"])
    return hashes


def load_rescued_offline_metrics(models_dir: Path, audit_json: Path | None = None) -> dict[str, Any]:
    expected_hashes = _audit_hashes(audit_json) if audit_json is not None else {}
    metrics = {}
    for controller, rel_path in RESCUED_OFFLINE_EVALS.items():
        source = models_dir / rel_path
        if not source.exists():
            raise FileNotFoundError(f"Missing rescued offline eval for {controller}: {source}")
        expected_hash = expected_hashes.get(rel_path)
        if expected_hash is not None:
            actual_hash = _file_sha256(source)
            if actual_hash != expected_hash:
                raise ValueError(
                    f"Rescued offline eval hash mismatch for {controller}: "
                    f"{source} sha256={actual_hash}, expected={expected_hash}"
                )
        metrics[controller] = load_json(source)
    missing_hashes = sorted(set(RESCUED_OFFLINE_EVALS.values()) - set(expected_hashes)) if audit_json is not None else []
    if missing_hashes:
        raise ValueError(f"Audit file is missing offline_eval hashes: {missing_hashes[:5]}")
    return metrics


def _planner_accuracy_applies(controller_metadata: dict[str, Any]) -> bool:
    conditioning = str(controller_metadata.get("conditioning", ""))
    return "predicted" in conditioning


def package_metrics(metrics: dict[str, Any], metadata: dict[str, Any], metric_origin: str) -> dict[str, Any]:
    packaged: dict[str, Any] = {}
    for controller, values in metrics.items():
        if controller not in metadata:
            raise ValueError(f"Missing controller metadata for {controller}")
        row = {key: float(values[key]) for key in NUMERIC_METRIC_KEYS}
        row.update(metadata[controller])
        if _planner_accuracy_applies(row):
            if values.get("planner_code_accuracy") is None:
                raise ValueError(f"Missing planner_code_accuracy for predicted-code controller {controller}")
            row["planner_code_accuracy"] = float(values["planner_code_accuracy"])
        else:
            row["planner_code_accuracy"] = None
        row["result_role"] = "privileged_upper_bound" if row["privileged_future_codes"] else "deployable_policy"
        row["real_calvin_closed_loop"] = False
        row["fresh_local_regeneration_blocked"] = True
        row["metric_origin"] = metric_origin
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
                if value is None:
                    row[key] = ""
                else:
                    row[key] = str(value).lower() if isinstance(value, bool) else value
            writer.writerow(row)


def provenance() -> dict[str, Any]:
    return {
        "not_a_real_calvin_closed_loop_verdict": True,
        "fresh_local_regeneration_blocked": True,
        "safe_interpretation": "offline_eval.json metrics recovered from the non-versioned rescue bundle as annotation-window-weighted 1024-record validation-subset metrics; GT-code rows are privileged upper-bound diagnostics; no predicted-code claim is published",
        "phase1_offline_metrics": {
            "table_files": [
                "results/phase1_offline_metrics.json",
                "results/phase1_offline_metrics.csv",
            ],
            "artifact_audit_file": "results/rescued_artifact_audit.json",
            "status": "committed offline_eval.json values recovered from the rescued Phase 1 run",
            "single_file_safety": "metrics JSON/CSV duplicate privileged_future_codes, deployable_without_future_labels, conditioning, result_role, real_calvin_closed_loop, fresh_local_regeneration_blocked, and metric_origin flags",
            "planner_code_accuracy": "null/blank means no predicted-code planner was evaluated for that controller",
            "validation_window_basis": "configured annotation-window subset: 1024 validation latent-cache records from a 12881-window validation index",
            "action_metric_basis": "action_nll and action_mse are raw-vector losses over the stored CALVIN action representation; no standalone action normalizer was recovered",
            "jerk_metric_basis": "offline jerk is computed on the full predicted action chunk, not on a real closed-loop executed action stream",
            "future_planner_accuracy_note": "planner training accuracy is teacher-forced; predicted-code evaluation accuracy must be labeled separately as autoregressive sample accuracy",
            "not_a_real_calvin_closed_loop_verdict": True,
            "fresh_local_regeneration_blocked": True,
            "blockers": [
                "raw CALVIN RGB/proprio/action frame files are not committed",
                "rescued manifests reference the original remote Linux roots",
                "the locally observed ldm_tokenizer_training_init_weights.pt does not cover all official tokenizer encode-path parameters",
                "rescued window indexes predate the current strict metadata guard, although their payloads match current window generation",
                "rescued latent caches predate the current strict metadata guard and cover the configured limited evaluation windows only",
            ],
            "safe_interpretation": "offline_eval.json metrics recovered from the non-versioned rescue bundle as annotation-window-weighted 1024-record validation-subset metrics; GT-code rows are privileged upper-bound diagnostics; no predicted-code claim is published",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate the committed Phase 1 result table schema and provenance flags.")
    parser.add_argument("--rescued-models-dir", type=Path, required=True, help="Rescue-bundle models directory containing */offline_eval.json files.")
    parser.add_argument("--controller-metadata", type=Path, default=Path("results/phase1_controller_metadata.json"))
    parser.add_argument("--output-json", type=Path, default=Path("results/phase1_offline_metrics.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("results/phase1_offline_metrics.csv"))
    parser.add_argument("--provenance-json", type=Path, default=Path("results/phase1_result_provenance.json"))
    parser.add_argument("--audit-json", type=Path, default=Path("results/rescued_artifact_audit.json"), help="Audit JSON containing SHA256 hashes for the intended rescued offline_eval.json files.")
    parser.add_argument("--allow-unaudited-rescue", action="store_true", help="Package from a rescue directory without checking offline_eval.json hashes.")
    args = parser.parse_args()

    metric_origin = "rescued_offline_eval_json"
    audit_json = None if args.allow_unaudited_rescue else args.audit_json
    metrics = load_rescued_offline_metrics(args.rescued_models_dir, audit_json=audit_json)
    packaged = package_metrics(metrics, load_json(args.controller_metadata), metric_origin)
    write_json(args.output_json, packaged)
    write_csv(args.output_csv, packaged)
    write_json(args.provenance_json, provenance())


if __name__ == "__main__":
    main()
