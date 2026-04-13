#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/vw2_idm/exp_history_ablation.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" -m videoworld2.robot_idm.eval.eval_ablation "${CONFIG_PATH}"
