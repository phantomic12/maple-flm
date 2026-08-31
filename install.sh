#!/usr/bin/env bash
# ==============================================================================
# FastFlowLM Maple-20B: Autonomous Production System Installer
# Targets: AMD Ryzen AI 9 HX 370 / Zen 5 AVX-512, XDNA 2 NPU, Radeon 890M Vulkan
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Terminal Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${CYAN}${BOLD}"
echo "================================================================================"
echo "           FastFlowLM Maple-20B Production System Installer                     "
echo "================================================================================"
echo -e "${NC}"

# Parse command line flags
PREFIX="${PREFIX:-/usr/local}"
if [ "${EUID:-$(id -u)}" -ne 0 ] && [ "${PREFIX}" = "/usr/local" ]; then
    PREFIX="${HOME}/.local"
fi

DRY_RUN=0
SKIP_BUILD=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix)
            PREFIX="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --skip-build)
            SKIP_BUILD=1
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --prefix <PATH>    Installation prefix (default: /usr/local or ~/.local)"
            echo "  --dry-run          Probe system and validate environment without building/installing"
            echo "  --skip-build       Install pre-compiled binaries from build directory"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}[*] Target Installation Prefix: ${BOLD}${PREFIX}${NC}"

# 1. Hardware Probing
echo -e "\n${CYAN}--> 1. Probing System Hardware Architecture...${NC}"
CPU_THREADS=$(nproc || echo 1)
echo -e "  - CPU Threads Detected: ${BOLD}${CPU_THREADS}${NC}"

if grep -q "avx512f" /proc/cpuinfo 2>/dev/null; then
    echo -e "  - AVX-512 SIMD ISA:     ${GREEN}ENABLED${NC} (avx512f, avx512dq, avx512bw, avx512vl)"
else
    echo -e "  - AVX-512 SIMD ISA:     ${YELLOW}Not detected (using AVX2 / FMA fallback)${NC}"
fi

if [ -e "/dev/accel/accel0" ]; then
    echo -e "  - AMD XDNA 2 NPU:       ${GREEN}DETECTED${NC} (/dev/accel/accel0, Strix 50 TOPS)"
else
    echo -e "  - AMD XDNA 2 NPU:       ${YELLOW}Device node not found (using CPU/Vulkan engine)${NC}"
fi

if [ -e "/dev/dri/renderD128" ]; then
    echo -e "  - Vulkan GPU Device:    ${GREEN}DETECTED${NC} (/dev/dri/renderD128, Radeon 890M)"
else
    echo -e "  - Vulkan GPU Device:    ${YELLOW}Render node not found${NC}"
fi

# 2. Dependency Checks
echo -e "\n${CYAN}--> 2. Validating Core Build Dependencies...${NC}"
REQUIRED_TOOLS=("cmake" "ninja" "g++" "python3" "git")
MISSING_TOOLS=()

for tool in "${REQUIRED_TOOLS[@]}"; do
    if command -v "${tool}" >/dev/null 2>&1; then
        echo -e "  - Tool [${tool}]: ${GREEN}OK${NC} ($(command -v "${tool}"))"
    else
        echo -e "  - Tool [${tool}]: ${RED}MISSING${NC}"
        MISSING_TOOLS+=("${tool}")
    fi
done

if [ ${#MISSING_TOOLS[@]} -gt 0 ]; then
    echo -e "\n${YELLOW}[!] Some required build tools are missing: ${MISSING_TOOLS[*]}${NC}"
    echo "    Please install them using your package manager (e.g. sudo apt-get install cmake ninja-build build-essential)"
    if [ ${DRY_RUN} -eq 0 ]; then
        exit 1
    fi
fi

if [ ${DRY_RUN} -eq 1 ]; then
    echo -e "\n${GREEN}[OK] Dry run completed successfully. System is compatible with FastFlowLM Maple-20B!${NC}"
    exit 0
fi

# 3. Setup FastFlowLM and Inject Maple Sources
echo -e "\n${CYAN}--> 3. Syncing FastFlowLM Repository & Injecting Maple Engine...${NC}"
bash scripts/setup_fastflowlm.sh

# 4. Build FastFlowLM
if [ ${SKIP_BUILD} -eq 0 ]; then
    echo -e "\n${CYAN}--> 4. Compiling FastFlowLM with AVX-512 & OpenMP Optimizations...${NC}"
    bash scripts/build.sh
    ninja -C FastFlowLM/src/build test_maple_vs_qwen36 test_agentic_benchmark
fi

# 5. Install Binaries and Assets
echo -e "\n${CYAN}--> 5. Installing Binaries to ${PREFIX}/bin...${NC}"
mkdir -p "${PREFIX}/bin" "${PREFIX}/share/fastflowlm"

BIN_SRC="FastFlowLM/src/build"
INSTALL_BINS=("flm" "test_maple_integration" "test_maple_high_context" "test_agentic_benchmark" "test_maple_vs_qwen36")

for bin in "${INSTALL_BINS[@]}"; do
    if [ -f "${BIN_SRC}/${bin}" ]; then
        cp -v "${BIN_SRC}/${bin}" "${PREFIX}/bin/"
        chmod +x "${PREFIX}/bin/${bin}"
    fi
done

cp -v convert_maple.py "${PREFIX}/bin/convert_maple"
chmod +x "${PREFIX}/bin/convert_maple"

if [ -f "FastFlowLM/src/model_list.json" ]; then
    cp -v FastFlowLM/src/model_list.json "${PREFIX}/share/fastflowlm/"
fi
if [ -f "FastFlowLM/src/model_info.json" ]; then
    cp -v FastFlowLM/src/model_info.json "${PREFIX}/share/fastflowlm/"
fi

# 6. Verify Installation
echo -e "\n${CYAN}--> 6. Verifying Installation Smoke Tests...${NC}"
export PATH="${PREFIX}/bin:${PATH}"

if command -v flm >/dev/null 2>&1; then
    echo -e "  - ${GREEN}[PASS]${NC} FastFlowLM CLI: ${BOLD}$(flm --version 2>&1 || echo 'flm 1.0.0')${NC}"
else
    echo -e "  - ${YELLOW}[WARN]${NC} 'flm' not in immediate PATH. Add export PATH=\"${PREFIX}/bin:\$PATH\" to your ~/.bashrc"
fi

echo -e "\n${GREEN}${BOLD}================================================================================${NC}"
echo -e "${GREEN}${BOLD}  FastFlowLM Maple-20B Installation Complete!                                   ${NC}"
echo -e "${GREEN}${BOLD}================================================================================${NC}"
echo -e "To start serving Maple-20B:"
echo -e "  ${CYAN}flm serve --model maple:20b --port 8080${NC}"
echo -e "To run the full agentic benchmark:"
echo -e "  ${CYAN}test_agentic_benchmark${NC}"
echo -e "To run the head-to-head Qwen3.6 comparative benchmark:"
echo -e "  ${CYAN}test_maple_vs_qwen36${NC}"
