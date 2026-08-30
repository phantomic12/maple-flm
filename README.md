# maple-flm: Porting `deepgrove/maple-preview` to FastFlowLM

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![Model: deepgrove/maple-preview](https://img.shields.io/badge/Model-deepgrove%2Fmaple--preview-blue)](https://huggingface.co/deepgrove/maple-preview)
[![Runtime: FastFlowLM](https://img.shields.io/badge/Runtime-FastFlowLM-purple)](https://github.com/ROCm/FastFlowLM)

**maple-flm** is the port and optimization repository for running DeepGrove's **Maple-Preview** (20B-A1B ternary-weight reasoning MoE) on AMD Ryzen AI NPUs via [FastFlowLM (ROCm/FastFlowLM)](https://github.com/ROCm/FastFlowLM).

---

## 🚀 Key Features

- **Hybrid 3:1 Attention Pattern**: Seamlessly handles 18 sliding window attention layers (window size 512) and 6 full global attention layers across 24 Transformer decoder layers.
- **Partial RoPE with No-PE on Global Layers**: Correctly implements RoPE on the first 64 head dimensions for sliding attention layers, with rotary embeddings bypassed for full global attention layers.
- **QK Normalization**: RMSNorm applied to Query and Key projections per head prior to attention.
- **256-Expert Top-8 Clamped SwiGLU MoE**: Fast Top-8 router selection with SwiGLU clamped activation `silu(clamp(gate, max=7.0)) * clamp(up, min=-7.0, max=7.0)` routed to 256 compact ternary experts.
- **Thinking & Tool-Calling Support**: Native streaming `<think> ... </think>` parsing and OpenAI-compatible reasoning streaming.
- **Agent-Ready & Modular**: Portable architecture, comprehensive automated test suite, and detailed documentation for multi-agent collaboration.

---

## 📂 Repository Layout

| Path | Description |
|---|---|
| [`README.md`](README.md) | Master documentation and quickstart |
| [`TASK_BRIEF.md`](TASK_BRIEF.md) | Self-contained, delegate-able task specifications |
| [`AGENTS.md`](AGENTS.md) | Instructions and rules for AI agents |
| [`CLAUDE.md`](CLAUDE.md) | Assistant and developer cheat sheet |
| [`convert_maple.py`](convert_maple.py) | Standalone Hugging Face checkpoint converter |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Deep technical architecture specifications |
| [`docs/PLAN.md`](docs/PLAN.md) | Phased roadmap and remaining milestones |
| [`docs/VERIFY.md`](docs/VERIFY.md) | Verification and validation procedures |
| [`scripts/`](scripts/) | Setup, build, and test scripts |
| [`flm_maple/`](flm_maple/) | Self-contained C++ source files, headers, tests, and manifests |

---

## 🛠️ Quickstart

### 1. Setup and Build
```bash
# Clone dependencies and sync Maple files into FastFlowLM
bash scripts/setup_fastflowlm.sh

# Build FastFlowLM binary and test suite
bash scripts/build.sh
```

### 2. Run Verification Suite
```bash
bash scripts/test_maple.sh
```

Expected output:
```
=== Running FastFlowLM Maple Integration Verification ===
[PASS] Model registry recognizes maple:20b and maple-preview:20b
[PASS] Model details and family mapped correctly: "maple"
[PASS] AutoModel successfully instantiates Maple instance: Maple
[PASS] Parameter configuration (enable_think, reasoning_effort) works
[PASS] Non-stream reasoning parsing verified
[PASS] maple_npu causal_lm engine instantiated and verified

>>> ALL MAPLE-PREVIEW PORT INTEGRATION TESTS PASSED! <<<
```

### 3. Convert Model Checkpoint
```bash
python3 convert_maple.py --src-dir /path/to/deepgrove/maple-preview --out-dir /path/to/flm/models/maple
```

---

## 📜 License
This project is licensed under the [MIT License](LICENSE).
