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

## 3. CLI Model Discovery Verification

Verify that the `flm` CLI binary discovers `maple:20b` and `maple-preview:20b`:
```bash
FLM_CONFIG_PATH=FastFlowLM/src/model_list.json ./FastFlowLM/src/build/flm list
```

### Expected Output:
```
Models:
  ...
  - maple:20b ⏬
  - maple-preview:20b ⏬
  ...
```

---

## 4. Checkpoint Converter Verification

Test the Python converter script on Hugging Face format files:
```bash
python3 convert_maple.py --help
```
