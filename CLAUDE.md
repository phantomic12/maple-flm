# Developer & Assistant Guide (`CLAUDE.md`)

## Quick Commands

```bash
# Setup FastFlowLM and inject Maple port files
bash scripts/setup_fastflowlm.sh

# Build FastFlowLM and test binaries
bash scripts/build.sh

# Run test suite
bash scripts/test_maple.sh

# Convert Hugging Face model files
python3 convert_maple.py --src-dir <hf_dir> --out-dir <flm_dir>
```

## Key Code References
- **AutoModel Interface**: `flm_maple/include/AutoModel/modeling_maple.hpp`
- **AutoModel Implementation**: `flm_maple/common/AutoModel/modeling_maple.cpp`
- **Engine Interface**: `flm_maple/include/models/maple/maple_npu.hpp`
- **Engine Implementation**: `flm_maple/common/models/maple/maple_npu.cpp`
- **Integration Test**: `flm_maple/test/test_maple_integration.cpp`
- **Model Registry Entries**: `flm_maple/manifests/`
- **Architecture Details**: `docs/ARCHITECTURE.md`
- **Roadmap & Plan**: `docs/PLAN.md`
