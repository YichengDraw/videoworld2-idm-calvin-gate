#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/vw2_idm/exp_distill.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" -m videoworld2.robot_idm.train.distill_policy "${CONFIG_PATH}"
