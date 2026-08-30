#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# FastFlowLM Hugging Face Hub Model Uploader
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

REPO_ID="${1:-phantomic12/maple-preview-20b-q4nx}"
MODEL_DIR="${2:-${ROOT_DIR}/models/maple-preview-20b}"
COMMIT_MSG="${3:-feat: upload FastFlowLM Maple-Preview 20B Q4NX quantized weights}"

echo "================================================================="
echo "=== FastFlowLM Hugging Face Hub Model Uploader ==="
echo "================================================================="
echo "Target Repository ID : ${REPO_ID}"
echo "Local Model Directory : ${MODEL_DIR}"
echo "Commit Message        : ${COMMIT_MSG}"
echo "================================================================="

if ! command -v hf &>/dev/null; then
    echo "[ERROR] Hugging Face CLI ('hf') is not installed in PATH."
    echo "Install via: pip install -U huggingface_hub[cli]"
    exit 1
fi

echo -e "\n[1/3] Verifying Hugging Face Authentication..."
hf auth whoami

echo -e "\n[2/3] Verifying Model Files in ${MODEL_DIR}..."
ls -lh "${MODEL_DIR}"

echo -e "\n[3/3] Uploading Model Checkpoint to Hugging Face Hub (${REPO_ID})..."
hf upload "${REPO_ID}" "${MODEL_DIR}" . \
    --commit-message "${COMMIT_MSG}" \
    --commit-description "Quantized ternary Q4NX & GGUF weights for FastFlowLM on AMD Ryzen AI NPU"

echo -e "\n================================================================="
echo "=== Upload Complete! View your model at:"
echo "=== https://huggingface.co/${REPO_ID}"
echo "================================================================="
