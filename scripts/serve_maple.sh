#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export FLM_XCLBIN_PATH="${ROOT_DIR}/FastFlowLM/src/xclbins"
export FLM_CONFIG_PATH="${ROOT_DIR}/FastFlowLM/src/model_list.json"
export FLM_MODELS_PATH="${ROOT_DIR}/models"

PORT="${1:-8080}"
HOST="${2:-0.0.0.0}"

echo "================================================================="
echo "=== FastFlowLM OpenAI-Compatible REST Server: Maple-20B ==="
echo "================================================================="
echo "Listening on: http://${HOST}:${PORT}"
echo "Endpoints:"
echo "  - POST /v1/chat/completions"
echo "  - GET  /v1/models"
echo "Hardware Backend: AMD Ryzen AI NPU"
echo "================================================================="

exec "${ROOT_DIR}/FastFlowLM/src/build/flm" serve --port "${PORT}" --host "${HOST}"
