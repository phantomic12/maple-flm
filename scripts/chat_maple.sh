#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export FLM_XCLBIN_PATH="${ROOT_DIR}/FastFlowLM/src/xclbins"
export FLM_CONFIG_PATH="${ROOT_DIR}/FastFlowLM/src/model_list.json"
export FLM_MODELS_PATH="${ROOT_DIR}/models"

MODEL_TAG="${1:-maple:20b}"

echo "================================================================="
echo "=== Starting FastFlowLM Interactive Chat: ${MODEL_TAG} ==="
echo "================================================================="
echo "Hardware Backend: AMD Ryzen AI NPU (/dev/accel/accel0)"
echo "Type '/exit' or Ctrl+C to quit."
echo "================================================================="

exec "${ROOT_DIR}/FastFlowLM/src/build/flm" run "${MODEL_TAG}"
