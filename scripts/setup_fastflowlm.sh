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

if [ -d "${ROOT_DIR}/flm_maple/include/xrt_headers" ]; then
    echo "Providing portable XRT headers to FastFlowLM include tree..."
    cp -rn "${ROOT_DIR}/flm_maple/include/xrt_headers/"* src/include/ || true
fi

echo "Injecting Maple manifests, CMake targets, and AutoModel bindings..."
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

# 3. Patch CMakeLists.txt with Maple test and benchmark targets
cm_path = Path("src/CMakeLists.txt")
targets_cmake_path = Path("../flm_maple/cmake/maple_targets.cmake")
if cm_path.exists() and targets_cmake_path.exists():
    cm_content = cm_path.read_text()
    if "test_maple_integration" not in cm_content:
        targets_cmake = targets_cmake_path.read_text()
        cm_path.write_text(cm_content + "\n\n" + targets_cmake)
        print("[OK] CMake targets injected into CMakeLists.txt")

# 4. Patch all_models.hpp
am_path = Path("src/include/AutoModel/all_models.hpp")
if am_path.exists():
    am_content = am_path.read_text()
    if "modeling_maple.hpp" not in am_content:
        am_content = '#include "modeling_maple.hpp"\n' + am_content
    if "maple," not in am_content:
        am_content = am_content.replace("enum class SupportedModelFamily {", "enum class SupportedModelFamily {\n    maple,")
    if '"maple"' not in am_content:
        map_needle = 'static const std::unordered_map<std::string, SupportedModelFamily> model_family_map = {'
        am_content = am_content.replace(map_needle, map_needle + '\n        {"maple", SupportedModelFamily::maple},\n        {"maple-preview", SupportedModelFamily::maple},')
    if "SupportedModelFamily::maple" not in am_content:
        switch_needle = 'switch (model_family) {'
        am_content = am_content.replace(switch_needle, switch_needle + '\n        case SupportedModelFamily::maple:\n            auto_chat_engine = std::make_unique<Maple>(npu_device_inst);\n            break;')
    am_path.write_text(am_content)
    print("[OK] all_models.hpp verified")

# 5. Patch automodel.hpp
atm_path = Path("src/include/AutoModel/automodel.hpp")
if atm_path.exists():
    atm_content = atm_path.read_text()
    if 'models/maple/maple_npu.hpp' not in atm_content:
        atm_content = '#include "models/maple/maple_npu.hpp"\n' + atm_content
        atm_path.write_text(atm_content)
        print("[OK] automodel.hpp verified")

# 6. Patch model_downloader.cpp
md_path = Path("src/pull/model_downloader.cpp")
if md_path.exists():
    md_content = md_path.read_text()
    if "FLM_MAPLE_HF_REPO" not in md_content:
        target = "if (!file_exists(local_path)) {\n                std::string url;"
        replacement = """if (!file_exists(local_path)) {
                std::string effective_base_url = base_url;
                if (new_model_tag.find("maple") != std::string::npos) {
                    const char* custom_repo = std::getenv("FLM_MAPLE_HF_REPO");
                    if (custom_repo && std::strlen(custom_repo) > 0) {
                        std::string repo_str(custom_repo);
                        if (repo_str.rfind("http", 0) == 0) {
                            effective_base_url = repo_str;
                        } else {
                            effective_base_url = "https://huggingface.co/" + repo_str;
                        }
                    }
                }

                std::string url;"""
        if target in md_content:
            md_content = md_content.replace(target, replacement)
            if "#include <cstring>" not in md_content:
                md_content = "#include <cstring>\n" + md_content
            md_path.write_text(md_content)
            print("[OK] model_downloader.cpp patched with FLM_MAPLE_HF_REPO")

print("[OK] All FastFlowLM files, targets, and manifests configured successfully.")
EOF

echo "[OK] FastFlowLM setup and Maple files copied."
