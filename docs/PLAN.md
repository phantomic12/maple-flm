# Maple-Preview FastFlowLM Port Roadmap & Execution Plan

This document outlines the phased roadmap for porting and optimizing **Maple-Preview** on **FastFlowLM**.

---

## Phase 1: Baseline Architecture & Interface Porting (Status: Completed ✅)

- [x] **Architecture Mapping**: Detailed mapping of 24 layers, GQA 16:4, 3:1 SWA:Global attention, QK-norm, partial RoPE, and 256-expert Top-8 clamped SwiGLU.
- [x] **AutoModel Class**: Implemented `Maple` class in `modeling_maple.hpp` and `modeling_maple.cpp` with `<think>` and `<tool_call>` streaming parsers.
- [x] **Causal LM Engine**: Implemented `maple_npu` in `maple_npu.hpp` and `maple_npu.cpp` executing full forward and prefill routines with KV-cache management.
- [x] **Model Registry**: Registered `maple:20b` and `maple-preview:20b` in `model_list.json` and `model_info.json`.
- [x] **Build System**: Integrated into CMake and Ninja build system.
- [x] **Integration Tests**: Implemented `test_maple_integration.cpp` verifying model resolution, parameter configuration, stream parsing, and engine state.
- [x] **Conversion Tool**: Created `convert_maple.py` for transforming Hugging Face checkpoints to FastFlowLM format.

---

## Phase 2: Checkpoint Conversion & Q4NX Packing

- [ ] **Weight Packing**: Download full Hugging Face checkpoint shards (`model-00001-of-00009.safetensors` through `model-00009-of-00009.safetensors`).
- [ ] **Ternary to Q4NX / Q4_K Mapping**: Pack ternary (-1, 0, 1) expert weights into compact 2-bit/4-bit block-quantized memory layouts matching NPU memory tile strides.
- [ ] **Embedding & Head Quantization**: Quantize `word_embeddings` and `lm_head` with Q4_K / BF16 quantization.
- [ ] **Manifest & Hash Verification**: Compute SHA256 hashes and register packed weights into FastFlowLM repository.

---

## Phase 3: NPU Kernel Compilation & Optimization

- [ ] **Grouped MoE Kernel**: Dispatch all 8 active experts per token via unified batched GEMM kernel instead of 8 individual GEMM dispatches.
- [ ] **Sliding Window KV Cache Ring Buffer**: Implement fixed 512-slot circular buffer for sliding window layers to reduce memory bandwidth by 85%.
- [ ] **AVX-512 / XDNA2 Native Dispatch**: Verify zero-copy DMA transfers between host memory and NPU column tiles.
- [ ] **Throughput Benchmarking**: Benchmark prefill (tok/s) and decode (tok/s) on target AMD Ryzen AI NPU hardware.

---

## Phase 4: Serving & Tool-Calling Integration

- [ ] **OpenAI-Compatible Streaming**: Verify `flm serve maple:20b` with streaming reasoning tokens (`delta.reasoning_content`) and function call schemas.
- [ ] **Multi-Turn Chat History**: Verify prefix-matching cache reuse across multi-turn reasoning conversations.
- [ ] **Continuous Integration**: Add automated test runs into GitHub Actions workflows.
