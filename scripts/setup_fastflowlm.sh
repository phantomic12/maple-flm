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
cp -v "${ROOT_DIR}/flm_maple/test/"*.cpp src/test/
cp -v "${ROOT_DIR}/convert_maple.py" .

echo "Injecting Maple manifests into model_list.json and model_info.json..."
python3 - << 'EOF'
import json
from pathlib import Path

# 1. Update model_list.json
model_list_path = Path("src/model_list.json")
if model_list_path.exists():
    with open(model_list_path, "r") as f:
        ml = json.load(f)
    with open("../flm_maple/manifests/model_list_entry.json", "r") as f:
        mle = json.load(f)
    for k, v in mle.items():
        ml.setdefault("models", {})[k] = v
    with open(model_list_path, "w") as f:
        json.dump(ml, f, indent=4)

# 2. Update model_info.json
model_info_path = Path("src/model_info.json")
if model_info_path.exists():
    with open(model_info_path, "r") as f:
        mi = json.load(f)
    with open("../flm_maple/manifests/model_info_entry.json", "r") as f:
        mie = json.load(f)
    for k, v in mie.items():
        mi[k] = v
    with open(model_info_path, "w") as f:
        json.dump(mi, f, indent=4)
print("[OK] Manifests merged successfully.")
EOF

echo "[OK] FastFlowLM setup and Maple files copied."
