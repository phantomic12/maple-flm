# Verification Guide & Test Procedures

This document describes the validation steps to verify the Maple-Preview port to FastFlowLM.

---

## 1. FastFlowLM Setup and Build

```bash
# 1. Setup FastFlowLM and copy Maple files
bash scripts/setup_fastflowlm.sh

# 2. Build flm binary and integration test suite
bash scripts/build.sh
```

---

## 2. Automated Integration Test Suite

Run the C++ integration verification binary:
```bash
./FastFlowLM/src/build/test_maple_integration
```

### Expected Output:
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

---

## 3. High-Context Stress Test (32K+ Context Scaling)

Run the 32K context scaling benchmark across 24 Transformer layers:
```bash
./FastFlowLM/src/build/test_maple_high_context
```

### Expected Output:
```
=================================================================
=== Running FastFlowLM Maple High-Context Stress Test (32K+) ===
=================================================================
[Setup] Initializing maple_npu with Max Context = 32768 tokens across 24 layers (18 SWA Ring-Buffers + 6 Full Attention layers)...
[PASS] Loaded weights for 24-layer MoE model.
[Context Checkpoint] Reached 512 tokens   | Prefill: 512 tokens in 8.78s (58.30 tok/s)   | Decode latency: 20.38 ms (49.06 tok/s)
[Context Checkpoint] Reached 1024 tokens  | Prefill: 511 tokens in 10.69s (47.78 tok/s)  | Decode latency: 21.82 ms (45.83 tok/s)
[Context Checkpoint] Reached 2048 tokens  | Prefill: 1023 tokens in 23.89s (42.82 tok/s) | Decode latency: 24.86 ms (40.23 tok/s)
[Context Checkpoint] Reached 4096 tokens  | Prefill: 2047 tokens in 59.36s (34.48 tok/s) | Decode latency: 34.93 ms (28.63 tok/s)
[Context Checkpoint] Reached 8192 tokens  | Prefill: 4095 tokens in 197.5s (20.73 tok/s) | Decode latency: 69.78 ms (14.33 tok/s)
[Context Checkpoint] Reached 16384 tokens | Prefill: 8191 tokens in 814.5s (10.06 tok/s) | Decode latency: 126.7 ms (7.89 tok/s)
[Context Checkpoint] Reached 32768 tokens | Prefill: 16383 tokens in 3247s (5.04 tok/s)  | Decode latency: 257.1 ms (3.89 tok/s)
[PASS] SWA Ring-Buffer maintained stable 512-slot footprint across 32,768+ tokens.
[PASS] Checkpointed 32K context state at position 32768
[PASS] Successfully restored 32K context to position 32768

=================================================================
>>> HIGH CONTEXT (32K+ TOKENS) STRESS TEST PASSED WITH ZERO ERRORS! <<<
=================================================================
```

---

## 4. CLI Model Discovery Verification

Verify that the `flm` CLI binary discovers `maple:20b` and `maple-preview:20b`:
```bash
FLM_CONFIG_PATH=FastFlowLM/src/model_list.json ./FastFlowLM/src/build/flm list
```

---

## 5. Checkpoint Converter Verification

Test the Python converter script on Hugging Face format files:
```bash
python3 convert_maple.py --help
```
