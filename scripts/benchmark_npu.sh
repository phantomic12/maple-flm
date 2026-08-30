#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "================================================================="
echo "=== FastFlowLM AMD Ryzen AI NPU Hardware Benchmark Runner ==="
echo "================================================================="

export FLM_XCLBIN_PATH="${ROOT_DIR}/FastFlowLM/src/xclbins"
export FLM_CONFIG_PATH="${ROOT_DIR}/FastFlowLM/src/model_list.json"

if [ ! -f "${ROOT_DIR}/FastFlowLM/src/build/test_maple_npu_bench" ]; then
    echo "Building test_maple_npu_bench binary..."
    ninja -C "${ROOT_DIR}/FastFlowLM/src/build" test_maple_npu_bench
fi

MODEL_DIR="${ROOT_DIR}/models/maple-preview-20b"
if [ ! -f "${MODEL_DIR}/model.q4nx" ]; then
    echo "Preparing Q4NX checkpoint in ${MODEL_DIR}..."
    python3 "${ROOT_DIR}/convert_maple.py" --generate-synthetic --num-layers 24 --num-experts 16 --out-dir "${MODEL_DIR}"
fi

echo -e "\n--> 1. Running Short-Prompt NPU Benchmark (32 tokens)..."
"${ROOT_DIR}/FastFlowLM/src/build/test_maple_npu_bench" \
    --model-path "${MODEL_DIR}" \
    --gen-tokens 32 \
    --prompt "Explain MoE Top-8 routing."

echo -e "\n--> 2. Running Medium-Prompt NPU Benchmark (128 tokens)..."
"${ROOT_DIR}/FastFlowLM/src/build/test_maple_npu_bench" \
    --model-path "${MODEL_DIR}" \
    --gen-tokens 128 \
    --max-len 8192 \
    --prompt "Provide an overview of Mixture of Experts architecture, focusing on router gate activations, Top-8 expert selection, and SwiGLU clamping bounds."

echo -e "\n--> 3. Running Long-Context NPU Benchmark (256 tokens)..."
"${ROOT_DIR}/FastFlowLM/src/build/test_maple_npu_bench" \
    --model-path "${MODEL_DIR}" \
    --gen-tokens 256 \
    --max-len 16384 \
    --prompt "Analyze the mathematical foundation of ternary weight representation {-1, 0, +1} in large language models."

echo -e "\n================================================================="
echo "=== All NPU Benchmarks Completed Successfully! ==="
echo "================================================================="
