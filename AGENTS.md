# Agent Collaboration Guide (`AGENTS.md`)

Welcome, autonomous AI coding agent! This document contains all necessary environment context, file pointers, conventions, and procedures to work seamlessly on this repository.

---

## 1. Project Context
- **Project Name**: `maple-flm` (Maple-Preview 20B MoE port to FastFlowLM)
- **Upstream Model**: `deepgrove/maple-preview` (20B-A1B ternary reasoning MoE)
- **Target Runtime**: `FastFlowLM` (ROCm NPU runtime)

---

## 2. Directory Layout

```
.
├── README.md                 # Master overview and quickstart
├── TASK_BRIEF.md             # Self-contained task brief
├── AGENTS.md                 # This file (agent instructions)
├── CLAUDE.md                 # Developer cheat sheet & build commands
├── convert_maple.py          # Standalone checkpoint & config converter
├── docs/
│   ├── ARCHITECTURE.md       # Deep technical architecture details
│   ├── PLAN.md               # Phased roadmap & future tasks
│   └── VERIFY.md             # Ordered test and validation steps
├── scripts/
│   ├── setup_fastflowlm.sh   # Sets up FastFlowLM and injects Maple files
│   ├── build.sh              # Configures CMake and compiles with Ninja
│   └── test_maple.sh         # Runs full integration test suite
└── flm_maple/                # Portable source files for FastFlowLM integration
    ├── include/
    │   ├── AutoModel/
    │   │   └── modeling_maple.hpp
    │   └── models/maple/
    │       └── maple_npu.hpp
    ├── common/
    │   ├── AutoModel/
    │   │   └── modeling_maple.cpp
    │   └── models/maple/
    │       └── maple_npu.cpp
    ├── test/
    │   └── test_maple_integration.cpp
    └── manifests/
        ├── model_list_entry.json
        └── model_info_entry.json
```

---

## 3. FastFlowLM Setup & Build Commands

```bash
# 1. Initialize FastFlowLM and sync Maple sources
bash scripts/setup_fastflowlm.sh

# 2. Build the project
bash scripts/build.sh

# 3. Run verification suite
bash scripts/test_maple.sh
```

---

## 4. Key Implementation Rules for Agents
1. **Never break existing models**: Keep all changes to `all_models.hpp`, `automodel.hpp`, and `CMakeLists.txt` additive and backwards-compatible.
2. **Precision & Memory Layout**:
   - Head dimension is 128 (16 Q heads = 2048, 4 KV heads = 512).
   - Partial RoPE factor is 0.5 (rotary dimension 64, pass-through dimension 64).
   - Rotary embeddings **must only be applied on sliding window layers**; global attention layers must not apply RoPE (`nope_on_global_attention: true`).
   - Activation clamping: `silu(clamp(gate, max=7.0)) * clamp(up, min=-7.0, max=7.0)` before down projection.
3. **Always verify tests**: Before finishing your turn, ensure `bash scripts/test_maple.sh` exits with code 0 and all tests pass.
