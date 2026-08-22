#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${ROBOTWIN_ROOT:-}" ]]; then
    echo "Set ROBOTWIN_ROOT to the RoboTwin repository root." >&2
    exit 1
fi

TASK_NAME="${1:-beat_block_hammer}"
TASK_CONFIG="${2:-demo_clean}"
SEED="${3:-0}"

cd "${ROBOTWIN_ROOT}"
export PYTHONPATH="${ROBOTWIN_ROOT}/policy/PoseVLA/robotwin:${ROBOTWIN_ROOT}:${PYTHONPATH:-}"

python script/eval_policy.py \
    --config policy/PoseVLA/pi05_baseline/robotwin_eval/deploy_policy.yml \
    --overrides \
    --task_name "${TASK_NAME}" \
    --task_config "${TASK_CONFIG}" \
    --seed "${SEED}" \
    --action_type pi05
