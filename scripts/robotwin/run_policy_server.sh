#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
POLICY_RUNTIME_ROOT="${TURBOVLA_POLICY_RUNTIME_ROOT:-${REPO_ROOT}}"

export PYTHONPATH="${POLICY_RUNTIME_ROOT}:${POLICY_RUNTIME_ROOT}/third_party/starvla_runtime:${PYTHONPATH:-}"

if [[ $# -lt 1 ]]; then
    echo "Usage: bash scripts/robotwin/run_policy_server.sh <ckpt_path> [gpu_id] [port]" >&2
    exit 1
fi

your_ckpt="$1"
gpu_id="${2:-${ROBOTWIN_SERVER_GPU:-0}}"
port="${3:-${ROBOTWIN_SERVER_PORT:-5694}}"
star_vla_python="${STARVLA_PYTHON:-${star_vla_python:-python}}"

use_bf16_flag=()
if [[ "${ROBOTWIN_USE_BF16:-1}" != "0" ]]; then
    use_bf16_flag+=(--use_bf16)
fi
config_flags=()
if [[ "${ROBOTWIN_LEGACY_RELEASED:-0}" == "1" ]]; then
    : "${TURBOVLA_DINOV3_PATH:?Legacy released runtime requires DINOv3 path}"
    : "${TURBOVLA_BERT_PATH:?Legacy released runtime requires BERT path}"
    config_flags+=(
        --cfg-option "framework.dinov3.model_path=${TURBOVLA_DINOV3_PATH}"
        --cfg-option "framework.dinov3.local_files_only=true"
        --cfg-option "framework.dinov3.attn_implementation=null"
        --cfg-option "framework.text.bert_path=${TURBOVLA_BERT_PATH}"
        --cfg-option "framework.text.local_files_only=true"
        --cfg-option "framework.text.attn_implementation=null"
    )
fi

echo "[INFO] Starting RoboTwin policy server"
echo "[INFO] checkpoint: ${your_ckpt}"
echo "[INFO] gpu: ${gpu_id}"
echo "[INFO] port: ${port}"

CUDA_VISIBLE_DEVICES="${gpu_id}" "${star_vla_python}" "${POLICY_RUNTIME_ROOT}/third_party/starvla_runtime/deployment/model_server/server_policy.py" \
    --ckpt_path "${your_ckpt}" \
    --port "${port}" \
    "${use_bf16_flag[@]}" \
    "${config_flags[@]}"
