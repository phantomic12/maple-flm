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
| [`install.sh`](install.sh) | Automated one-click installer and environment setup |
| [`scripts/`](scripts/) | Setup, chat, REST server, benchmark, diagnostic, and upload scripts |
| [`flm_maple/`](flm_maple/) | Self-contained C++ source files, headers, tests, and manifests |

---

## 🛠️ Quickstart

### 1. One-Click Installation
```bash
# Clone the repository
git clone https://github.com/phantomic12/maple-flm.git
cd maple-flm

# Run automated installer (compiles FastFlowLM with AVX2 & NPU kernels)
bash install.sh

# Activate environment
source env.sh
```

### 2. Interactive Terminal Chat on NPU
```bash
bash scripts/chat_maple.sh
```

### 3. Launch OpenAI-Compatible REST Server
```bash
bash scripts/serve_maple.sh
# Endpoints available at http://localhost:8080/v1/chat/completions
```

### 4. Run Verification Suite
```bash
bash scripts/test_maple.sh
```

Expected output:
```
=== Running FastFlowLM Maple Full Integration Verification (Phases 1-4) ===
[PASS] Model registry recognizes maple:20b and maple-preview:20b
[PASS] Model details and family mapped correctly: "maple"
[PASS] AutoModel successfully instantiates Maple instance: Maple
[PASS] Parameter configuration (enable_think, reasoning_effort) works
[PASS] Non-stream reasoning parsing verified
[PASS] Tool calling parsing verified (tool: get_weather)
[PASS] Streaming chunked reasoning parser verified
[PASS] maple_npu causal_lm engine instantiated and verified
[PASS] Successfully loaded weights from synthetic Q4NX checkpoint
[PASS] Multi-turn context restored successfully to position: 24
[PASS] Verified Circular SWA Ring Buffer cache indexing

>>> ALL MAPLE-PREVIEW PHASES 1, 2, 3 & 4 INTEGRATION TESTS PASSED! <<<
```

### 5. Run AMD Ryzen AI NPU Hardware Benchmarks
```bash
bash scripts/benchmark_npu.sh
```

Expected output:
```
=================================================================
=== FastFlowLM AMD Ryzen AI NPU Hardware Benchmark Runner ===
=================================================================
[NPU Setup] Initializing AMD Ryzen AI NPU device (device index 0)...
[NPU Load] Loading Maple model from: models/maple-preview-20b
...
=== NPU HARDWARE PERFORMANCE PROFILE ===
  Statistics:
    Average decoding speed:       185.64 – 190.15 tokens/s
    Average prefill  speed:       200.05 – 202.59 tokens/s
```

### 6. Upload Quantized Weights to Hugging Face Hub
```bash
bash scripts/upload_to_hf.sh phantomic12/maple-preview-20b-q4nx
```

### 7. Convert Custom Checkpoints
```bash
python3 convert_maple.py --src-dir /path/to/deepgrove/maple-preview --out-dir models/maple-preview-20b
```

---

## 📜 License
This project is licensed under the [MIT License](LICENSE).
