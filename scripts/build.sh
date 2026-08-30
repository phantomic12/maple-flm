#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "=== Building FastFlowLM with Maple support ==="
cd "${ROOT_DIR}/FastFlowLM/src"

cmake -S . -B build \
    -DFLM_VERSION=1.0.3 \
    -DNPU_VERSION=32.0.203.304 \
    -DCMAKE_BUILD_TYPE=Release \
    -G Ninja

ninja -C build flm test_maple_integration

echo "[OK] Build completed successfully."
echo "Binary: ${ROOT_DIR}/FastFlowLM/src/build/flm"
echo "Test:   ${ROOT_DIR}/FastFlowLM/src/build/test_maple_integration"
