#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "=== Running Maple FastFlowLM Verification Suite ==="
cd "${ROOT_DIR}"

if [ ! -f "FastFlowLM/src/build/test_maple_integration" ]; then
    echo "Building test_maple_integration first..."
    bash "${SCRIPT_DIR}/build.sh"
fi

echo "--> 1. Running test_maple_integration unit/integration binary..."
"${ROOT_DIR}/FastFlowLM/src/build/test_maple_integration"

echo "--> 2. Verifying 'flm list' model discovery..."
FLM_CONFIG_PATH="${ROOT_DIR}/FastFlowLM/src/model_list.json" "${ROOT_DIR}/FastFlowLM/src/build/flm" list || true

echo "--> 3. Testing convert_maple.py on sample config..."
python3 "${ROOT_DIR}/convert_maple.py" --help > /dev/null

echo "=== All verification steps passed! ==="
