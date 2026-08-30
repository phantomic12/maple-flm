#!/usr/bin/env bash
# =================================================================
# FastFlowLM Maple-20B Power & Performance Mode Switcher
# =================================================================
# Usage:
#   bash scripts/switch_power_mode.sh performance   # 24 Threads + NPU LMHead + Max Throughput
#   bash scripts/switch_power_mode.sh battery       # Low-Power Zen 5c + NPU Full Offload + 0W GPU
# =================================================================

set -e

MODE="${1:-performance}"

case "$MODE" in
    battery|efficiency|eco)
        echo "================================================================="
        echo "🔋 Switching to BATTERY EFFICIENCY MODE (Ultra-Low Power)"
        echo "================================================================="
        echo "  - Hardware Profile: AMD Ryzen AI NPU (Full Offload) + Zen 5c cores"
        echo "  - GPU State:        Radeon 890M iGPU Powered Down (0W draw)"
        echo "  - Thread Budget:    4 Low-Power Cores"
        echo "  - Package Power:    ~5W - 8W (Max battery runtime)"
        echo "================================================================="
        export FLM_POWER_MODE=battery
        export FLM_MODE=battery
        export OMP_NUM_THREADS=4
        ;;
    perf|performance|max)
        echo "================================================================="
        echo "⚡ Switching to MAX PERFORMANCE MODE (AC / High-Throughput)"
        echo "================================================================="
        echo "  - Hardware Profile: 24 Zen 5 Threads + NPU LMHead + Batched GEMM"
        echo "  - GPU State:        Radeon 890M Available for High-Batch Compute"
        echo "  - Thread Budget:    24 Hardware Threads"
        echo "  - Target Speed:     ~200+ tok/s prefill, ~185-190 tok/s decode"
        echo "================================================================="
        export FLM_POWER_MODE=performance
        export FLM_MODE=performance
        export OMP_NUM_THREADS=24
        ;;
    *)
        echo "Usage: bash scripts/switch_power_mode.sh [performance|battery]"
        exit 1
        ;;
esac

echo ""
echo "To apply this in your current shell, run:"
echo "  export FLM_POWER_MODE=$FLM_POWER_MODE"
echo "  export OMP_NUM_THREADS=$OMP_NUM_THREADS"
