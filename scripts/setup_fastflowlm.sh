#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "=== Setting up FastFlowLM repository ==="
cd "${ROOT_DIR}"

if [ ! -d "FastFlowLM" ]; then
    echo "Cloning ROCm/FastFlowLM..."
    git clone https://github.com/ROCm/FastFlowLM.git FastFlowLM
fi

cd FastFlowLM
echo "Updating submodules..."
git submodule update --init --recursive

echo "Copying Maple port files into FastFlowLM source tree..."
mkdir -p src/include/models/maple src/common/models/maple src/test

cp -v "${ROOT_DIR}/flm_maple/include/AutoModel/modeling_maple.hpp" src/include/AutoModel/
cp -v "${ROOT_DIR}/flm_maple/include/models/maple/maple_npu.hpp" src/include/models/maple/
cp -v "${ROOT_DIR}/flm_maple/common/AutoModel/modeling_maple.cpp" src/common/AutoModel/
cp -v "${ROOT_DIR}/flm_maple/common/models/maple/maple_npu.cpp" src/common/models/maple/
cp -v "${ROOT_DIR}/flm_maple/test/test_maple_integration.cpp" src/test/
cp -v "${ROOT_DIR}/convert_maple.py" .

echo "[OK] FastFlowLM setup and Maple files copied."
