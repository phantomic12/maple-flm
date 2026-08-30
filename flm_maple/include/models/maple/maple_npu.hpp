/// \file maple_npu.hpp
/// \brief maple_npu class for Maple-Preview 20B-A1B MoE reasoning model
/// \author FastFlowLM Team
/// \date 2026-08-29
/// \version 1.0.0
/// \note This is a header file for the maple_npu class
#pragma once

#include "lm_config.hpp"
#include "npu_utils/npu_utils.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "modules/embedding.hpp"
#include "modules/lm_head.hpp"
#include "modules/gemm.hpp"
#include "modules/dequant.hpp"
#include "tensor_2d.hpp"
#include "utils/utils.hpp"
#include "causal_lm.hpp"

#if USEAVX2
#include <immintrin.h>
#endif

class maple_npu : public causal_lm {
public:
    /// \brief initialize the maple_npu engine
    /// \param config the model configuration
    /// \param npu_instance the NPU manager instance
    /// \param MAX_L the maximum sequence context length
    maple_npu(LM_Config config, npu_xclbin_manager *npu_instance, int MAX_L = 4096);
    ~maple_npu() override;

    /// \brief forward pass for a single token
    /// \param ids input token id
    /// \return output logits buffer
    buffer<bf16> forward(int ids) override;

    /// \brief prefill pass for a sequence of tokens
    /// \param ids input token ids
    /// \param payload optional multimodal payload (unused for pure text Maple)
    /// \return output logits buffer for the last token
    buffer<bf16> prefill(std::vector<int>& ids, void* payload = nullptr) override;

    /// \brief set the context length
    /// \param L context length
    void set_context_length(int L) override;

    /// \brief load weights from Q4NX / SafeTensors format
    /// \param q4nx the weight loader
    void load_weights(Q4NX& q4nx) override;

    /// \brief clear KV cache and context
    void clear_context() override;

    /// \brief get key cache for specific layer and index
    buffer<bf16> get_k_cache(int layer_idx, int idx) override;

    /// \brief get value cache for specific layer and index
    buffer<bf16> get_v_cache(int layer_idx, int idx) override;

    /// \brief update the maximum context length
    void update_max_length(uint32_t MAX_L) override;

    /// \brief get the current context length
    int get_current_context_length() override;

    /// \brief create a checkpoint of the current KV cache and state
    int checkpoint() override;

    /// \brief restore KV cache and state from checkpoint
    int restore() override;

private:
    struct Impl;
    Impl* _impl;
};
