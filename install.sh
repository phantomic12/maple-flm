#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# maple-flm: One-Click Installer & FastFlowLM NPU Runtime Setup
# Port of DeepGrove Maple-Preview 20B MoE for AMD Ryzen AI NPUs (XDNA 2 / Strix)
# ==============================================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

BOLD="\033[1m"
GREEN="\033[0;32m"
BLUE="\033[0;34m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
NC="\033[0m"

echo -e "${BOLD}${BLUE}=================================================================${NC}"
echo -e "${BOLD}${BLUE}===  maple-flm: FastFlowLM AMD Ryzen AI NPU Setup & Installer ===${NC}"
echo -e "${BOLD}${BLUE}=================================================================${NC}"

echo -e "\n${BOLD}[1/5] Checking System Dependencies...${NC}"
MISSING_PKGS=()
for cmd in cmake ninja g++ git python3; do
    if ! command -v "$cmd" &>/dev/null; then
        MISSING_PKGS+=("$cmd")
    fi
done

if [ ${#MISSING_PKGS[@]} -ne 0 ]; then
    echo -e "${RED}[ERROR] Missing required build tools: ${MISSING_PKGS[*]}${NC}"
    echo "Please install them via your package manager (e.g. 'sudo pacman -S cmake ninja gcc git python' or 'sudo apt install cmake ninja-build build-essential git python3')"
    exit 1
fi
echo -e "  ${GREEN}✓ Build toolchain verified (cmake, ninja, g++, git, python3)${NC}"

# Check NPU driver node
if [ -e "/dev/accel/accel0" ]; then
    echo -e "  ${GREEN}✓ AMD Ryzen AI NPU device node detected (/dev/accel/accel0)${NC}"
else
    echo -e "  ${YELLOW}⚠ Warning: /dev/accel/accel0 not found. NPU hardware acceleration requires the amdxdna driver.${NC}"
fi

echo -e "\n${BOLD}[2/5] Initializing FastFlowLM Submodule & Port Sources...${NC}"
bash "${ROOT_DIR}/scripts/setup_fastflowlm.sh"

echo -e "\n${BOLD}[3/5] Compiling FastFlowLM with AVX2 & NPU Kernels...${NC}"
bash "${ROOT_DIR}/scripts/build.sh"

echo -e "\n${BOLD}[4/5] Running Hardware Diagnostic & Sanity Tests...${NC}"
bash "${ROOT_DIR}/scripts/test_maple.sh"

echo -e "\n${BOLD}[5/5] Generating Environment Configuration (env.sh)...${NC}"
cat << EOF > "${ROOT_DIR}/env.sh"
# Source this file to configure environment for FastFlowLM Maple
export FLM_XCLBIN_PATH="${ROOT_DIR}/FastFlowLM/src/xclbins"
export FLM_CONFIG_PATH="${ROOT_DIR}/FastFlowLM/src/model_list.json"
export FLM_MODELS_PATH="${ROOT_DIR}/models"
export PATH="${ROOT_DIR}/FastFlowLM/src/build:\$PATH"
EOF

chmod +x "${ROOT_DIR}/env.sh"

echo -e "\n${BOLD}${GREEN}=================================================================${NC}"
echo -e "${BOLD}${GREEN}===  maple-flm Installation & Build Completed Successfully!   ===${NC}"
echo -e "${BOLD}${GREEN}=================================================================${NC}"
echo -e "To start using maple-flm:"
echo -e "  1. Activate environment:     ${BOLD}source env.sh${NC}"
echo -e "  2. Interactive Chat:         ${BOLD}bash scripts/chat_maple.sh${NC}"
echo -e "  3. OpenAI REST Server:       ${BOLD}bash scripts/serve_maple.sh${NC}"
echo -e "  4. Run NPU Benchmarks:       ${BOLD}bash scripts/benchmark_npu.sh${NC}"
echo -e "  5. Hardware Diagnostics:     ${BOLD}bash scripts/diagnose_npu.sh${NC}"
echo -e "=================================================================\n"
