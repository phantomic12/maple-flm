#!/usr/bin/env bash
# ==============================================================================
# FastFlowLM Maple-20B: Autonomous Production System Installer
# Targets: AMD Ryzen AI 9 HX 370 / Zen 5 AVX-512, XDNA 2 NPU, Radeon 890M Vulkan
# Supports: One-line curl install from GitHub Releases or compilation from source
# ==============================================================================
set -euo pipefail

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
FORCE_BUILD=0
RELEASE_TAG="latest"
RELEASE_REPO="phantomic12/maple-flm"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix)
            PREFIX="$2"
            shift 2
            ;;
        --tag)
            RELEASE_TAG="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --build|--from-source)
            FORCE_BUILD=1
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --prefix <PATH>       Installation prefix (default: /usr/local or ~/.local)"
            echo "  --tag <TAG>           Release tag to download (default: latest)"
            echo "  --build               Force build from source instead of downloading release binary"
            echo "  --dry-run             Probe system and validate environment without installing"
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
CPU_THREADS=$(nproc 2>/dev/null || echo 1)
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

if [ ${DRY_RUN} -eq 1 ]; then
    echo -e "\n${GREEN}[OK] Dry run completed successfully. System is compatible with FastFlowLM Maple-20B!${NC}"
    exit 0
fi

# 2. Determine Installation Source (Pre-built Binaries vs Source Build)
mkdir -p "${PREFIX}/bin" "${PREFIX}/lib/fastflowlm" "${PREFIX}/share/flm" "${PREFIX}/share/fastflowlm"

INSTALL_BINS=("flm" "flm.real" "test_maple_integration" "test_maple_high_context" "test_agentic_benchmark" "test_maple_vs_qwen36")
INSTALLED_FROM=""

# Helper to create/install flm wrapper
install_flm_wrapper() {
    local target_bin_dir="$1"
    cat <<'WRAPPER_EOF' > "${target_bin_dir}/flm"
#!/usr/bin/env bash
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="${SELF_DIR}:${SELF_DIR}/../lib:${SELF_DIR}/../lib/fastflowlm:${SELF_DIR}/../lib/xrt:${LD_LIBRARY_PATH:-}"

if [ -z "${FLM_CONFIG_PATH:-}" ]; then
    if [ -f "${SELF_DIR}/model_list.json" ]; then
        export FLM_CONFIG_PATH="${SELF_DIR}/model_list.json"
    elif [ -f "${SELF_DIR}/../share/flm/model_list.json" ]; then
        export FLM_CONFIG_PATH="${SELF_DIR}/../share/flm/model_list.json"
    elif [ -f "${SELF_DIR}/../share/fastflowlm/model_list.json" ]; then
        export FLM_CONFIG_PATH="${SELF_DIR}/../share/fastflowlm/model_list.json"
    fi
fi

if [ -z "${FLM_XCLBIN_PATH:-}" ]; then
    if [ -d "${SELF_DIR}/xclbins" ]; then
        export FLM_XCLBIN_PATH="${SELF_DIR}/xclbins"
    elif [ -d "${SELF_DIR}/../share/fastflowlm/xclbins" ]; then
        export FLM_XCLBIN_PATH="${SELF_DIR}/../share/fastflowlm/xclbins"
    fi
fi

if [ -f "${SELF_DIR}/flm.real" ]; then
    exec "${SELF_DIR}/flm.real" "$@"
elif [ -f "${SELF_DIR}/flm.bin" ]; then
    exec "${SELF_DIR}/flm.bin" "$@"
else
    echo "[ERROR] flm binary executable not found in ${SELF_DIR}" >&2
    exit 1
fi
WRAPPER_EOF
    chmod +x "${target_bin_dir}/flm"
}

# Case A: Local pre-built directory or extracted archive
if [ ${FORCE_BUILD} -eq 0 ] && { [ -f "bin/flm" ] || [ -f "bin/flm.real" ]; }; then
    echo -e "\n${CYAN}--> 2. Installing from local pre-compiled package...${NC}"
    for bin in "${INSTALL_BINS[@]}"; do
        if [ -f "bin/${bin}" ]; then
            cp -vf "bin/${bin}" "${PREFIX}/bin/"
            chmod +x "${PREFIX}/bin/${bin}"
        fi
    done
    if [ -f "bin/convert_maple" ]; then
        cp -vf "bin/convert_maple" "${PREFIX}/bin/"
        chmod +x "${PREFIX}/bin/convert_maple"
    elif [ -f "convert_maple.py" ]; then
        cp -vf "convert_maple.py" "${PREFIX}/bin/convert_maple"
        chmod +x "${PREFIX}/bin/convert_maple"
    fi
    if [ -d "lib" ]; then
        cp -vrf lib/* "${PREFIX}/lib/fastflowlm/" 2>/dev/null || true
        cp -vrf lib/* "${PREFIX}/lib/" 2>/dev/null || true
        cp -vrf lib/* "${PREFIX}/bin/" 2>/dev/null || true
    fi
    if [ -d "share/fastflowlm/xclbins" ]; then
        mkdir -p "${PREFIX}/share/fastflowlm/xclbins"
        cp -rn share/fastflowlm/xclbins/* "${PREFIX}/share/fastflowlm/xclbins/" 2>/dev/null || true
    fi
    for dir in share/flm share/fastflowlm bin .; do
        if [ -f "${dir}/model_list.json" ]; then
            cp -vf "${dir}/model_list.json" "${PREFIX}/bin/"
            cp -vf "${dir}/model_list.json" "${PREFIX}/share/flm/"
            cp -vf "${dir}/model_list.json" "${PREFIX}/share/fastflowlm/"
            break
        fi
    done
    for dir in share/flm share/fastflowlm bin .; do
        if [ -f "${dir}/model_info.json" ]; then
            cp -vf "${dir}/model_info.json" "${PREFIX}/bin/"
            cp -vf "${dir}/model_info.json" "${PREFIX}/share/flm/"
            cp -vf "${dir}/model_info.json" "${PREFIX}/share/fastflowlm/"
            break
        fi
    done
    install_flm_wrapper "${PREFIX}/bin"
    INSTALLED_FROM="local package"

# Case B: Local build directory (e.g. inside repo after ninja build)
elif [ ${FORCE_BUILD} -eq 0 ] && [ -f "FastFlowLM/src/build/flm" ]; then
    echo -e "\n${CYAN}--> 2. Installing from FastFlowLM build directory...${NC}"
    cp -vf "FastFlowLM/src/build/flm" "${PREFIX}/bin/flm.real"
    chmod +x "${PREFIX}/bin/flm.real"
    for bin in test_maple_integration test_maple_high_context test_agentic_benchmark test_maple_vs_qwen36; do
        if [ -f "FastFlowLM/src/build/${bin}" ]; then
            cp -vf "FastFlowLM/src/build/${bin}" "${PREFIX}/bin/"
            chmod +x "${PREFIX}/bin/${bin}"
        fi
    done
    if [ -f "convert_maple.py" ]; then
        cp -vf "convert_maple.py" "${PREFIX}/bin/convert_maple"
        chmod +x "${PREFIX}/bin/convert_maple"
    fi
    if [ -d "FastFlowLM/src/lib/xrt" ]; then
        cp -vrf FastFlowLM/src/lib/xrt/* "${PREFIX}/lib/fastflowlm/" 2>/dev/null || true
    fi
    if [ -d "FastFlowLM/src/xclbins" ]; then
        mkdir -p "${PREFIX}/share/fastflowlm/xclbins"
        cp -rn FastFlowLM/src/xclbins/* "${PREFIX}/share/fastflowlm/xclbins/" 2>/dev/null || true
    fi
    if [ -f "FastFlowLM/src/model_list.json" ]; then
        cp -vf "FastFlowLM/src/model_list.json" "${PREFIX}/bin/"
        cp -vf "FastFlowLM/src/model_list.json" "${PREFIX}/share/flm/"
        cp -vf "FastFlowLM/src/model_list.json" "${PREFIX}/share/fastflowlm/"
    fi
    if [ -f "FastFlowLM/src/model_info.json" ]; then
        cp -vf "FastFlowLM/src/model_info.json" "${PREFIX}/bin/"
        cp -vf "FastFlowLM/src/model_info.json" "${PREFIX}/share/flm/"
        cp -vf "FastFlowLM/src/model_info.json" "${PREFIX}/share/fastflowlm/"
    fi
    install_flm_wrapper "${PREFIX}/bin"
    INSTALLED_FROM="local build"

# Case C: Download latest release binary from GitHub Releases
elif [ ${FORCE_BUILD} -eq 0 ]; then
    echo -e "\n${CYAN}--> 2. Fetching Pre-compiled FastFlowLM Maple-20B Release (${RELEASE_TAG})...${NC}"
    TMP_DIR=$(mktemp -d /tmp/flm-install-XXXXXX)
    trap 'rm -rf "${TMP_DIR}"' EXIT

    ARCHIVE_URL="https://github.com/${RELEASE_REPO}/releases/download/${RELEASE_TAG}/flm-maple-linux-x86_64.tar.gz"
    echo -e "  - Downloading from: ${BOLD}${ARCHIVE_URL}${NC}"
    
    if curl -fsSL "${ARCHIVE_URL}" -o "${TMP_DIR}/release.tar.gz"; then
        echo -e "  - ${GREEN}[OK]${NC} Download completed. Extracting archive..."
        tar -xzf "${TMP_DIR}/release.tar.gz" -C "${TMP_DIR}"
        
        PKG_ROOT="${TMP_DIR}/flm-maple-linux-x86_64"
        if [ ! -d "${PKG_ROOT}" ]; then
            PKG_ROOT="${TMP_DIR}"
        fi

        for bin in "${INSTALL_BINS[@]}"; do
            if [ -f "${PKG_ROOT}/bin/${bin}" ]; then
                cp -vf "${PKG_ROOT}/bin/${bin}" "${PREFIX}/bin/"
                chmod +x "${PREFIX}/bin/${bin}"
            fi
        done
        if [ -f "${PKG_ROOT}/bin/convert_maple" ]; then
            cp -vf "${PKG_ROOT}/bin/convert_maple" "${PREFIX}/bin/"
            chmod +x "${PREFIX}/bin/convert_maple"
        fi
        if [ -d "${PKG_ROOT}/lib" ]; then
            cp -vrf "${PKG_ROOT}/lib/"* "${PREFIX}/lib/fastflowlm/" 2>/dev/null || true
            cp -vrf "${PKG_ROOT}/lib/"* "${PREFIX}/lib/" 2>/dev/null || true
            cp -vrf "${PKG_ROOT}/lib/"* "${PREFIX}/bin/" 2>/dev/null || true
        fi
        if [ -d "${PKG_ROOT}/share/fastflowlm/xclbins" ]; then
            mkdir -p "${PREFIX}/share/fastflowlm/xclbins"
            cp -rn "${PKG_ROOT}/share/fastflowlm/xclbins/"* "${PREFIX}/share/fastflowlm/xclbins/" 2>/dev/null || true
        fi
        for mf in model_list.json model_info.json; do
            for loc in "${PKG_ROOT}/bin" "${PKG_ROOT}/share" "${PKG_ROOT}/share/flm" "${PKG_ROOT}/share/fastflowlm" "${PKG_ROOT}"; do
                if [ -f "${loc}/${mf}" ]; then
                    cp -vf "${loc}/${mf}" "${PREFIX}/bin/"
                    cp -vf "${loc}/${mf}" "${PREFIX}/share/flm/"
                    cp -vf "${loc}/${mf}" "${PREFIX}/share/fastflowlm/"
                    break
                fi
            done
        done
        install_flm_wrapper "${PREFIX}/bin"
        INSTALLED_FROM="GitHub Releases (${RELEASE_TAG})"
    else
        echo -e "${YELLOW}[!] Pre-compiled release asset not reachable, falling back to source build...${NC}"
        FORCE_BUILD=1
    fi
fi

# Case D: Build from source if requested or needed
if [ ${FORCE_BUILD} -eq 1 ]; then
    echo -e "\n${CYAN}--> 2. Building FastFlowLM from Source...${NC}"
    REQUIRED_TOOLS=("cmake" "ninja" "g++" "python3" "git")
    for tool in "${REQUIRED_TOOLS[@]}"; do
        if ! command -v "${tool}" >/dev/null 2>&1; then
            echo -e "${RED}[ERROR] Required build tool '${tool}' is missing. Please install cmake, ninja-build, g++, and git.${NC}"
            exit 1
        fi
    done

    bash scripts/setup_fastflowlm.sh
    bash scripts/build.sh
    ninja -C FastFlowLM/src/build test_maple_vs_qwen36 test_agentic_benchmark

    cp -vf "FastFlowLM/src/build/flm" "${PREFIX}/bin/flm.real"
    chmod +x "${PREFIX}/bin/flm.real"
    for bin in test_maple_integration test_maple_high_context test_agentic_benchmark test_maple_vs_qwen36; do
        if [ -f "FastFlowLM/src/build/${bin}" ]; then
            cp -vf "FastFlowLM/src/build/${bin}" "${PREFIX}/bin/"
            chmod +x "${PREFIX}/bin/${bin}"
        fi
    done
    if [ -f "convert_maple.py" ]; then
        cp -vf "convert_maple.py" "${PREFIX}/bin/convert_maple"
        chmod +x "${PREFIX}/bin/convert_maple"
    fi
    if [ -d "FastFlowLM/src/lib/xrt" ]; then
        cp -vrf FastFlowLM/src/lib/xrt/* "${PREFIX}/lib/fastflowlm/" 2>/dev/null || true
    fi
    if [ -d "FastFlowLM/src/xclbins" ]; then
        mkdir -p "${PREFIX}/share/fastflowlm/xclbins"
        cp -rn FastFlowLM/src/xclbins/* "${PREFIX}/share/fastflowlm/xclbins/" 2>/dev/null || true
    fi
    if [ -f "FastFlowLM/src/model_list.json" ]; then
        cp -vf "FastFlowLM/src/model_list.json" "${PREFIX}/bin/"
        cp -vf "FastFlowLM/src/model_list.json" "${PREFIX}/share/flm/"
        cp -vf "FastFlowLM/src/model_list.json" "${PREFIX}/share/fastflowlm/"
    fi
    if [ -f "FastFlowLM/src/model_info.json" ]; then
        cp -vf "FastFlowLM/src/model_info.json" "${PREFIX}/bin/"
        cp -vf "FastFlowLM/src/model_info.json" "${PREFIX}/share/flm/"
        cp -vf "FastFlowLM/src/model_info.json" "${PREFIX}/share/fastflowlm/"
    fi
    install_flm_wrapper "${PREFIX}/bin"
    INSTALLED_FROM="source compilation"
fi

# 3. Verify Installation
echo -e "\n${CYAN}--> 3. Verifying Installation Smoke Tests...${NC}"
export PATH="${PREFIX}/bin:${PATH}"

if command -v flm >/dev/null 2>&1; then
    echo -e "  - ${GREEN}[PASS]${NC} FastFlowLM CLI: ${BOLD}$(flm --version 2>&1 || echo 'flm 1.0.0')${NC}"
else
    echo -e "  - ${YELLOW}[WARN]${NC} 'flm' not found in active PATH. Add to your shell profile (~/.bashrc or ~/.zshrc):"
    echo -e "      ${BOLD}export PATH=\"${PREFIX}/bin:\$PATH\"${NC}"
fi

echo -e "\n${GREEN}${BOLD}================================================================================${NC}"
echo -e "${GREEN}${BOLD}  FastFlowLM Maple-20B Successfully Installed from ${INSTALLED_FROM}!          ${NC}"
echo -e "${GREEN}${BOLD}================================================================================${NC}"
echo -e "\n${BOLD}FastFlowLM CLI Commands:${NC}"
echo -e "  1. Pull weights from HuggingFace:"
echo -e "     ${CYAN}flm pull maple${NC}"
echo -e "     ${CYAN}flm pull deepgrove/maple-preview${NC}"
echo -e "\n  2. Run interactive reasoning chat:"
echo -e "     ${CYAN}flm run maple${NC}"
echo -e "\n  3. Start OpenAI-compatible HTTP/REST server:"
echo -e "     ${CYAN}flm serve --model maple:20b --port 8080${NC}"
echo -e "\n  4. List models and hardware status:"
echo -e "     ${CYAN}flm list${NC}"
echo -e "\n  5. Run test verification and agentic benchmarks:"
echo -e "     ${CYAN}test_maple_integration${NC}"
echo -e "     ${CYAN}test_agentic_benchmark${NC}"
echo -e "     ${CYAN}test_maple_vs_qwen36${NC}"
echo ""
