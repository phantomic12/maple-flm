# Task Brief: Port `deepgrove/maple-preview` to FastFlowLM

## Goal
Enable seamless local inference of DeepGrove's **Maple-Preview** 20B reasoning MoE model on AMD Ryzen AI NPUs using **FastFlowLM** (FLM).

---

## Technical Specifications
- **Model Source**: [https://huggingface.co/deepgrove/maple-preview](https://huggingface.co/deepgrove/maple-preview)
- **Architecture**: 20B-A1B Mixture-of-Experts (256 experts, top-8 active) with ternary weights.
- **Layers**: 24 Transformer layers (3:1 sliding-window attention of 512 tokens to full global attention).
- **Attention**: GQA (16 query heads, 4 KV heads), QK-RMSNorm, Partial RoPE on SWA layers, No-PE on global layers.
- **MLP**: Clamped SwiGLU `silu(clamp(gate, max=7.0)) * clamp(up, min=-7.0, max=7.0)` routed to top-8 experts.
- **Reasoning**: Native `<think> ... </think>` token emission and streaming.

---

## Deliverables Checklist
- [x] FastFlowLM AutoModel class (`Maple`) in `modeling_maple.hpp` and `modeling_maple.cpp`
- [x] FastFlowLM Causal LM engine (`maple_npu`) in `maple_npu.hpp` and `maple_npu.cpp`
- [x] Model registry entries in `model_list.json` and `model_info.json`
- [x] Checkpoint converter utility in `convert_maple.py`
- [x] CMakeLists build integration and integration tests in `test_maple_integration.cpp`
- [x] Automated test runner in `scripts/test_maple.sh`
- [x] 100K Agentic Ultra-Long Context Scaling Benchmark (`test_agentic_benchmark.cpp`)
- [x] Head-to-Head Comparative Benchmark Suite against Qwen3.6 MoE (`test_maple_vs_qwen36.cpp`)
- [x] Production GitHub Actions CI/CD Autobuilder with automated releases (`.github/workflows/ci.yml`)
- [x] One-line universal installer script (`install.sh`) supporting pre-compiled release downloads via curl
