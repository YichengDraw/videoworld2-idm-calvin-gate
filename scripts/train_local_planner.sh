#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/vw2_idm/planner_small.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" -m videoworld2.robot_idm.train.extract_local_codes "${CONFIG_PATH}"
"${PYTHON_BIN}" -m videoworld2.robot_idm.train.train_planner "${CONFIG_PATH}"
