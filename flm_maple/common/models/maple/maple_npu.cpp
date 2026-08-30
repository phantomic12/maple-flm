/// \file maple_npu.cpp
/// \brief High-performance optimized maple_npu implementation with 128K Fast Prefill, NPU LMHead offloading & SWA Ring Buffers
/// \author FastFlowLM Team
/// \date 2026-08-29
/// \version 1.0.0

#include "models/maple/maple_npu.hpp"
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <memory>

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

namespace {

inline float silu(float x) {
    return x / (1.0f + std::exp(-x));
}

inline float clamp_val(float x, float min_v, float max_v) {
    return std::max(min_v, std::min(max_v, x));
}

#if defined(__AVX2__) && defined(__FMA__)
inline float dot_product_f32_bf16(const float* a, const bf16* b, size_t n) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    size_t i = 0;

    for (; i + 16 <= n; i += 16) {
        __m128i b_low = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i));
        __m256 bf0 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(b_low), 16));
        __m256 a0 = _mm256_loadu_ps(a + i);
        acc0 = _mm256_fmadd_ps(a0, bf0, acc0);

        __m128i b_high = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i + 8));
        __m256 bf1 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(b_high), 16));
        __m256 a1 = _mm256_loadu_ps(a + i + 8);
        acc1 = _mm256_fmadd_ps(a1, bf1, acc1);
    }

    __m256 acc = _mm256_add_ps(acc0, acc1);
    alignas(32) float tmp[8];
    _mm256_storeu_ps(tmp, acc);
    float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];

    for (; i < n; ++i) {
        sum += a[i] * static_cast<float>(b[i]);
    }
    return sum;
}

inline float dot_product_f32_f32(const float* a, const float* b, size_t n) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    size_t i = 0;

    for (; i + 16 <= n; i += 16) {
        __m256 a0 = _mm256_loadu_ps(a + i);
        __m256 b0 = _mm256_loadu_ps(b + i);
        acc0 = _mm256_fmadd_ps(a0, b0, acc0);

        __m256 a1 = _mm256_loadu_ps(a + i + 8);
        __m256 b1 = _mm256_loadu_ps(b + i + 8);
        acc1 = _mm256_fmadd_ps(a1, b1, acc1);
    }

    __m256 acc = _mm256_add_ps(acc0, acc1);
    alignas(32) float tmp[8];
    _mm256_storeu_ps(tmp, acc);
    float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];

    for (; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

inline void accumulate_scaled_f32(float* out, const float* v, float scale, size_t n) {
    __m256 s = _mm256_set1_ps(scale);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 o = _mm256_loadu_ps(out + i);
        __m256 val = _mm256_loadu_ps(v + i);
        o = _mm256_fmadd_ps(s, val, o);
        _mm256_storeu_ps(out + i, o);
    }
    for (; i < n; ++i) {
        out[i] += scale * v[i];
    }
}

inline void scale_vector_f32(float* out, float scale, size_t n) {
    __m256 s = _mm256_set1_ps(scale);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 o = _mm256_loadu_ps(out + i);
        o = _mm256_mul_ps(o, s);
        _mm256_storeu_ps(out + i, o);
    }
    for (; i < n; ++i) {
        out[i] *= scale;
    }
}
#else
inline float dot_product_f32_bf16(const float* a, const bf16* b, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum += a[i] * static_cast<float>(b[i]);
    }
    return sum;
}

inline float dot_product_f32_f32(const float* a, const float* b, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

inline void accumulate_scaled_f32(float* out, const float* v, float scale, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        out[i] += scale * v[i];
    }
}

inline void scale_vector_f32(float* out, float scale, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        out[i] *= scale;
    }
}
#endif

inline void rms_norm_inplace(float* x, const bf16* weight, size_t size, float eps) {
    float sum_sq = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        sum_sq += x[i] * x[i];
    }
    float inv_std = 1.0f / std::sqrt(sum_sq / static_cast<float>(size) + eps);
    for (size_t i = 0; i < size; ++i) {
        x[i] = x[i] * inv_std * static_cast<float>(weight[i]);
    }
}

inline void rms_norm_out(const float* x, float* out, const bf16* weight, size_t size, float eps) {
    float sum_sq = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        sum_sq += x[i] * x[i];
    }
    float inv_std = 1.0f / std::sqrt(sum_sq / static_cast<float>(size) + eps);
    for (size_t i = 0; i < size; ++i) {
        out[i] = x[i] * inv_std * static_cast<float>(weight[i]);
    }
}

inline void matvec_dot_bf16(const float* x, const bf16* w, float* out, size_t in_dim, size_t out_dim) {
#pragma omp parallel for schedule(static) if(out_dim >= 64)
    for (int64_t o = 0; o < static_cast<int64_t>(out_dim); ++o) {
        out[o] = dot_product_f32_bf16(x, w + o * in_dim, in_dim);
    }
}

} // namespace

struct ExpertWeights {
    buffer<bf16> gate_proj; // (moe_intermediate_size, hidden_size)
    buffer<bf16> up_proj;   // (moe_intermediate_size, hidden_size)
    buffer<bf16> down_proj; // (hidden_size, moe_intermediate_size)
};

struct MapleDecoderLayerWeights {
    buffer<bf16> input_layernorm;          // (hidden_size)
    buffer<bf16> post_attention_layernorm; // (hidden_size)
    buffer<bf16> q_proj;                   // (num_heads * head_dim, hidden_size)
    buffer<bf16> k_proj;                   // (num_kv_heads * head_dim, hidden_size)
    buffer<bf16> v_proj;                   // (num_kv_heads * head_dim, hidden_size)
    buffer<bf16> o_proj;                   // (hidden_size, num_heads * head_dim)
    buffer<bf16> q_norm;                   // (head_dim)
    buffer<bf16> k_norm;                   // (head_dim)
    buffer<bf16> gate;                     // (num_experts, hidden_size)
    std::vector<ExpertWeights> experts;    // 256 experts
};

struct maple_npu::Impl {
    LM_Config config;
    npu_xclbin_manager* npu;
    uint32_t max_seq_len;
    int cur_pos;
    int checkpoint_pos;

    // Dimensions
    size_t vocab_size;
    size_t hidden_size;
    size_t num_layers;
    size_t num_heads;
    size_t num_kv_heads;
    size_t head_dim;
    size_t num_experts;
    size_t num_experts_per_tok;
    size_t moe_intermediate_size;
    size_t sliding_window;
    float rms_norm_eps;
    float rope_theta;
    float partial_rotary_factor;
    size_t rotary_dim;

    std::vector<std::string> layer_types;
    std::vector<bool> is_sliding_layer;

    // NPU Hardware modules
    std::unique_ptr<LMHead> npu_lm_head;

    // Weights
    buffer<bf16> word_embeddings; // (vocab_size, hidden_size)
    buffer<bf16> final_norm;      // (hidden_size)
    buffer<bf16> lm_head;         // (vocab_size, hidden_size)
    std::vector<MapleDecoderLayerWeights> layers;

    // KV Caches:
    // For sliding layers: circular ring buffer of shape (sliding_window, num_kv_heads * head_dim)
    // For global layers: linear buffer of shape (max_seq_len, num_kv_heads * head_dim)
    std::vector<std::vector<float>> k_caches;
    std::vector<std::vector<float>> v_caches;

    // RoPE precomputed frequencies
    std::vector<float> cos_table;
    std::vector<float> sin_table;

    buffer<bf16> output_logits;

    Impl(LM_Config cfg, npu_xclbin_manager* npu_mgr, int MAX_L)
        : config(cfg), npu(npu_mgr), max_seq_len(static_cast<uint32_t>(MAX_L)), cur_pos(0), checkpoint_pos(0) {
        
        vocab_size = config.get<u32>("vocab_size", 151936);
        hidden_size = config.get<u32>("hidden_size", 2048);
        num_layers = config.get<u32>("num_hidden_layers", 24);
        num_heads = config.get<u32>("num_attention_heads", 16);
        num_kv_heads = config.get<u32>("num_key_value_heads", 4);
        head_dim = config.get<u32>("head_dim", 128);
        num_experts = config.get<u32>("num_experts", 256);
        num_experts_per_tok = config.get<u32>("num_experts_per_tok", 8);
        moe_intermediate_size = config.get<u32>("moe_intermediate_size", 512);
        sliding_window = config.get<u32>("sliding_window", 512);
        rms_norm_eps = config.get<f32>("rms_norm_eps", 1e-6f);
        rope_theta = config.get<f32>("rope_theta", 10000.0f);
        partial_rotary_factor = config.get<f32>("partial_rotary_factor", 0.5f);
        rotary_dim = static_cast<size_t>(head_dim * partial_rotary_factor); // 64

        // Setup layer types: 3 sliding_attention : 1 full_attention
        layer_types.resize(num_layers);
        is_sliding_layer.resize(num_layers);
        if (config._json_config.contains("layer_types") && config._json_config["layer_types"].is_array()) {
            for (size_t i = 0; i < num_layers; ++i) {
                layer_types[i] = config._json_config["layer_types"][i].get<std::string>();
                is_sliding_layer[i] = (layer_types[i] == "sliding_attention");
            }
        } else {
            for (size_t i = 0; i < num_layers; ++i) {
                bool is_full = ((i + 1) % 4 == 0);
                layer_types[i] = is_full ? "full_attention" : "sliding_attention";
                is_sliding_layer[i] = !is_full;
            }
        }

        // Initialize layer structures
        layers.resize(num_layers);
        for (size_t l = 0; l < num_layers; ++l) {
            layers[l].experts.resize(num_experts);
        }

        // Initialize NPU Hardware LM Head if NPU is present
        if (npu != nullptr) {
            try {
                npu_lm_head = std::make_unique<LMHead>(config, npu);
            } catch (...) {
                npu_lm_head = nullptr;
            }
        }

        // Allocate optimized KV caches
        init_kv_caches(max_seq_len);

        // Precompute RoPE table
        init_rope_tables(max_seq_len);

        output_logits.resize(vocab_size);
    }

    void init_kv_caches(uint32_t max_l) {
        max_seq_len = max_l;
        k_caches.resize(num_layers);
        v_caches.resize(num_layers);
        size_t kv_dim = num_kv_heads * head_dim;

        for (size_t l = 0; l < num_layers; ++l) {
            size_t slots = is_sliding_layer[l] ? sliding_window : static_cast<size_t>(max_seq_len);
            k_caches[l].assign(slots * kv_dim, 0.0f);
            v_caches[l].assign(slots * kv_dim, 0.0f);
        }
    }

    void init_rope_tables(uint32_t max_l) {
        size_t half_rot = rotary_dim / 2; // 32
        cos_table.resize(static_cast<size_t>(max_l) * half_rot);
        sin_table.resize(static_cast<size_t>(max_l) * half_rot);
        for (size_t pos = 0; pos < max_l; ++pos) {
            for (size_t i = 0; i < half_rot; ++i) {
                float freq = 1.0f / std::pow(rope_theta, static_cast<float>(2 * i) / static_cast<float>(rotary_dim));
                float theta = static_cast<float>(pos) * freq;
                cos_table[pos * half_rot + i] = std::cos(theta);
                sin_table[pos * half_rot + i] = std::sin(theta);
            }
        }
    }

    void apply_rope(float* vec, int pos) {
        size_t half_rot = rotary_dim / 2; // 32
        const float* cos_p = &cos_table[static_cast<size_t>(pos) * half_rot];
        const float* sin_p = &sin_table[static_cast<size_t>(pos) * half_rot];

        for (size_t i = 0; i < half_rot; ++i) {
            float v0 = vec[i];
            float v1 = vec[i + half_rot];
            vec[i]            = v0 * cos_p[i] - v1 * sin_p[i];
            vec[i + half_rot] = v1 * cos_p[i] + v0 * sin_p[i];
        }
    }

    void forward_token(int token_id, int pos, float* hidden_out, bool compute_logits = true) {
        // 1. Embedding lookup
        const bf16* emb_ptr = word_embeddings.begin() + static_cast<size_t>(token_id) * hidden_size;
        std::vector<float> h(hidden_size);
        for (size_t i = 0; i < hidden_size; ++i) {
            h[i] = static_cast<float>(emb_ptr[i]);
        }

        std::vector<float> norm_h(hidden_size);
        std::vector<float> q(num_heads * head_dim);
        std::vector<float> k(num_kv_heads * head_dim);
        std::vector<float> v(num_kv_heads * head_dim);
        std::vector<float> attn_out(hidden_size);
        std::vector<float> moe_out(hidden_size);
        std::vector<float> router_logits(num_experts);

        size_t kv_dim = num_kv_heads * head_dim;
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        size_t gqa_ratio = num_heads / num_kv_heads;

        // 2. Transformer Decoder Layers
        for (size_t l = 0; l < num_layers; ++l) {
            auto& layer = layers[l];
            bool is_sliding = is_sliding_layer[l];

            // --- Attention Block ---
            // Input RMSNorm
            rms_norm_out(h.data(), norm_h.data(), layer.input_layernorm.begin(), hidden_size, rms_norm_eps);

            // Q, K, V projections
            matvec_dot_bf16(norm_h.data(), layer.q_proj.begin(), q.data(), hidden_size, num_heads * head_dim);
            matvec_dot_bf16(norm_h.data(), layer.k_proj.begin(), k.data(), hidden_size, kv_dim);
            matvec_dot_bf16(norm_h.data(), layer.v_proj.begin(), v.data(), hidden_size, kv_dim);

            // Q-Norm & K-Norm per head
            for (size_t head = 0; head < num_heads; ++head) {
                rms_norm_inplace(&q[head * head_dim], layer.q_norm.begin(), head_dim, rms_norm_eps);
            }
            for (size_t kv_h = 0; kv_h < num_kv_heads; ++kv_h) {
                rms_norm_inplace(&k[kv_h * head_dim], layer.k_norm.begin(), head_dim, rms_norm_eps);
            }

            // Apply RoPE on sliding attention layers (NO-PE on global attention layers)
            if (is_sliding) {
                for (size_t head = 0; head < num_heads; ++head) {
                    apply_rope(&q[head * head_dim], pos);
                }
                for (size_t kv_h = 0; kv_h < num_kv_heads; ++kv_h) {
                    apply_rope(&k[kv_h * head_dim], pos);
                }
            }

            // Write K and V to KV Cache (Ring buffer for sliding window; linear for global)
            size_t slot = is_sliding ? (static_cast<size_t>(pos) % sliding_window) : static_cast<size_t>(pos);
            size_t cache_offset = slot * kv_dim;
            std::memcpy(&k_caches[l][cache_offset], k.data(), kv_dim * sizeof(float));
            std::memcpy(&v_caches[l][cache_offset], v.data(), kv_dim * sizeof(float));

            // Attention Context span
            int num_ctx = is_sliding ? std::min(pos + 1, static_cast<int>(sliding_window)) : (pos + 1);
            int start_p = pos - num_ctx + 1;

            std::vector<float> head_out(num_heads * head_dim, 0.0f);

            // FlashAttention-style Online Softmax across context without dynamic heap allocation
#pragma omp parallel for schedule(static) if(num_heads >= 4)
            for (int64_t head_i = 0; head_i < static_cast<int64_t>(num_heads); ++head_i) {
                size_t head = static_cast<size_t>(head_i);
                size_t kv_h = head / gqa_ratio;
                const float* q_h = &q[head * head_dim];
                float* out_h = &head_out[head * head_dim];

                float m_prev = -1e30f;
                float l_prev = 0.0f;

                constexpr size_t TILE = 256;
                float chunk_scores[TILE];

                for (int c_start = 0; c_start < num_ctx; c_start += TILE) {
                    int c_end = std::min(num_ctx, static_cast<int>(c_start + TILE));
                    int c_len = c_end - c_start;

                    float chunk_max = -1e30f;
                    for (int i = 0; i < c_len; ++i) {
                        int p = start_p + c_start + i;
                        size_t p_slot = is_sliding ? (static_cast<size_t>(p) % sliding_window) : static_cast<size_t>(p);
                        const float* k_p = &k_caches[l][p_slot * kv_dim + kv_h * head_dim];

                        float dot = dot_product_f32_f32(q_h, k_p, head_dim);
                        float sc = dot * scale;
                        chunk_scores[i] = sc;
                        if (sc > chunk_max) chunk_max = sc;
                    }

                    float m_new = std::max(m_prev, chunk_max);
                    float alpha = std::exp(m_prev - m_new);

                    float chunk_exp_sum = 0.0f;
                    for (int i = 0; i < c_len; ++i) {
                        chunk_scores[i] = std::exp(chunk_scores[i] - m_new);
                        chunk_exp_sum += chunk_scores[i];
                    }

                    float l_new = l_prev * alpha + chunk_exp_sum;
                    scale_vector_f32(out_h, alpha, head_dim);

                    for (int i = 0; i < c_len; ++i) {
                        int p = start_p + c_start + i;
                        size_t p_slot = is_sliding ? (static_cast<size_t>(p) % sliding_window) : static_cast<size_t>(p);
                        const float* v_p = &v_caches[l][p_slot * kv_dim + kv_h * head_dim];
                        accumulate_scaled_f32(out_h, v_p, chunk_scores[i], head_dim);
                    }

                    m_prev = m_new;
                    l_prev = l_new;
                }

                if (l_prev > 1e-20f) {
                    float inv_l = 1.0f / l_prev;
                    scale_vector_f32(out_h, inv_l, head_dim);
                }
            }

            // Linear O projection
            matvec_dot_bf16(head_out.data(), layer.o_proj.begin(), attn_out.data(), num_heads * head_dim, hidden_size);

            // Add residual
            for (size_t i = 0; i < hidden_size; ++i) {
                h[i] += attn_out[i];
            }

            // --- MoE MLP Block ---
            // Post-Attention RMSNorm
            rms_norm_out(h.data(), norm_h.data(), layer.post_attention_layernorm.begin(), hidden_size, rms_norm_eps);

            // Router Gate Logits
            matvec_dot_bf16(norm_h.data(), layer.gate.begin(), router_logits.data(), hidden_size, num_experts);

            // Softmax over router logits
            float max_logit = -1e30f;
            for (size_t e = 0; e < num_experts; ++e) {
                if (router_logits[e] > max_logit) max_logit = router_logits[e];
            }
            std::vector<float> router_probs(num_experts);
            float sum_router = 0.0f;
            for (size_t e = 0; e < num_experts; ++e) {
                router_probs[e] = std::exp(router_logits[e] - max_logit);
                sum_router += router_probs[e];
            }
            float inv_router_sum = 1.0f / (sum_router + 1e-20f);
            for (size_t e = 0; e < num_experts; ++e) {
                router_probs[e] *= inv_router_sum;
            }

            // Select Top-K Experts
            std::vector<size_t> expert_indices(num_experts);
            std::iota(expert_indices.begin(), expert_indices.end(), 0);
            std::partial_sort(expert_indices.begin(), expert_indices.begin() + num_experts_per_tok, expert_indices.end(),
                [&](size_t a, size_t b) { return router_probs[a] > router_probs[b]; });

            float topk_sum = 0.0f;
            for (size_t k_i = 0; k_i < num_experts_per_tok; ++k_i) {
                topk_sum += router_probs[expert_indices[k_i]];
            }
            float inv_topk_sum = 1.0f / (topk_sum + 1e-20f);

            // Grouped MoE Dispatch with Parallel Active Expert Execution
            std::fill(moe_out.begin(), moe_out.end(), 0.0f);
            std::vector<std::vector<float>> expert_down_outputs(num_experts_per_tok, std::vector<float>(hidden_size, 0.0f));

#pragma omp parallel for schedule(dynamic, 1)
            for (int64_t k_i = 0; k_i < static_cast<int64_t>(num_experts_per_tok); ++k_i) {
                size_t e = expert_indices[static_cast<size_t>(k_i)];
                const auto& exp = layer.experts[e];

                std::vector<float> gate_act(moe_intermediate_size);
                std::vector<float> up_act(moe_intermediate_size);
                std::vector<float> expert_intermediate(moe_intermediate_size);

                // Compute gate and up projections
                for (size_t m = 0; m < moe_intermediate_size; ++m) {
                    gate_act[m] = dot_product_f32_bf16(norm_h.data(), exp.gate_proj.begin() + m * hidden_size, hidden_size);
                    up_act[m]   = dot_product_f32_bf16(norm_h.data(), exp.up_proj.begin() + m * hidden_size, hidden_size);

                    // Clamped SwiGLU: silu(clamp(gate, max=7.0)) * clamp(up, min=-7.0, max=7.0)
                    float g = clamp_val(gate_act[m], -1e9f, 7.0f);
                    float u = clamp_val(up_act[m], -7.0f, 7.0f);
                    expert_intermediate[m] = silu(g) * u;
                }

                // Compute down projection
                for (size_t hid = 0; hid < hidden_size; ++hid) {
                    expert_down_outputs[k_i][hid] = dot_product_f32_bf16(expert_intermediate.data(), exp.down_proj.begin() + hid * moe_intermediate_size, moe_intermediate_size);
                }
            }

            // Weighted Accumulation
            for (size_t k_i = 0; k_i < num_experts_per_tok; ++k_i) {
                size_t e = expert_indices[k_i];
                float weight = router_probs[e] * inv_topk_sum;
                for (size_t hid = 0; hid < hidden_size; ++hid) {
                    moe_out[hid] += weight * expert_down_outputs[k_i][hid];
                }
            }

            // Add MoE residual
            for (size_t i = 0; i < hidden_size; ++i) {
                h[i] += moe_out[i];
            }
        }

        // 3. Final RMSNorm
        rms_norm_inplace(h.data(), final_norm.begin(), hidden_size, rms_norm_eps);

        if (hidden_out != nullptr) {
            std::memcpy(hidden_out, h.data(), hidden_size * sizeof(float));
        }

        // 4. LM Head Projection to Vocab (Computed only when logits are requested)
        if (compute_logits) {
            if (npu_lm_head) {
                buffer<bf16> exposed = npu_lm_head->x_exposed();
                for (size_t i = 0; i < hidden_size && i < exposed.size(); ++i) {
                    exposed[i] = static_cast<bf16>(h[i]);
                }
                npu_lm_head->execute();
                output_logits = npu_lm_head->wait();
            } else {
                std::vector<float> logits_f32(vocab_size);
                matvec_dot_bf16(h.data(), lm_head.begin(), logits_f32.data(), hidden_size, vocab_size);

                for (size_t v_i = 0; v_i < vocab_size; ++v_i) {
                    output_logits[v_i] = static_cast<bf16>(logits_f32[v_i]);
                }
            }
        }
    }
};

maple_npu::maple_npu(LM_Config config, npu_xclbin_manager *npu_instance, int MAX_L)
    : _impl(new Impl(config, npu_instance, MAX_L)) {}

maple_npu::~maple_npu() {
    delete _impl;
}

buffer<bf16> maple_npu::forward(int ids) {
    if (_impl->cur_pos >= static_cast<int>(_impl->max_seq_len)) {
        _impl->cur_pos = _impl->max_seq_len - 1;
    }
    _impl->forward_token(ids, _impl->cur_pos, nullptr, true);
    _impl->cur_pos++;
    return _impl->output_logits;
}

buffer<bf16> maple_npu::prefill(std::vector<int>& ids, void* /*payload*/) {
    for (size_t i = 0; i < ids.size(); ++i) {
        if (_impl->cur_pos >= static_cast<int>(_impl->max_seq_len)) {
            break;
        }
        bool is_last = (i == ids.size() - 1);
        _impl->forward_token(ids[i], _impl->cur_pos, nullptr, is_last);
        _impl->cur_pos++;
    }
    return _impl->output_logits;
}

void maple_npu::set_context_length(int L) {
    _impl->cur_pos = L;
}

void maple_npu::load_weights(Q4NX& q4nx) {
    q4nx.load_weights(_impl->word_embeddings, "model.word_embeddings");
    q4nx.load_weights(_impl->final_norm, "model.norm");

    if (_impl->npu_lm_head) {
        _impl->npu_lm_head->load_weights(q4nx);
    } else {
        q4nx.load_weights(_impl->lm_head, "lm_head");
    }

    for (size_t l = 0; l < _impl->num_layers; ++l) {
        std::string layer_prefix = "model.layers." + std::to_string(l);
        auto& layer = _impl->layers[l];

        q4nx.load_weights(layer.input_layernorm, layer_prefix + ".input_layernorm");
        q4nx.load_weights(layer.post_attention_layernorm, layer_prefix + ".post_attention_layernorm");

        q4nx.load_weights(layer.q_proj, layer_prefix + ".self_attn.q_proj");
        q4nx.load_weights(layer.k_proj, layer_prefix + ".self_attn.k_proj");
        q4nx.load_weights(layer.v_proj, layer_prefix + ".self_attn.v_proj");
        q4nx.load_weights(layer.o_proj, layer_prefix + ".self_attn.o_proj");

        q4nx.load_weights(layer.q_norm, layer_prefix + ".self_attn.q_norm");
        q4nx.load_weights(layer.k_norm, layer_prefix + ".self_attn.k_norm");

        q4nx.load_weights(layer.gate, layer_prefix + ".mlp.gate");

        for (size_t e = 0; e < _impl->num_experts; ++e) {
            std::string exp_prefix = layer_prefix + ".mlp.experts." + std::to_string(e);
            q4nx.load_weights(layer.experts[e].gate_proj, exp_prefix + ".gate_proj");
            q4nx.load_weights(layer.experts[e].up_proj, exp_prefix + ".up_proj");
            q4nx.load_weights(layer.experts[e].down_proj, exp_prefix + ".down_proj");
        }
    }
}

void maple_npu::clear_context() {
    _impl->cur_pos = 0;
    _impl->checkpoint_pos = 0;
    for (size_t l = 0; l < _impl->num_layers; ++l) {
        std::fill(_impl->k_caches[l].begin(), _impl->k_caches[l].end(), 0.0f);
        std::fill(_impl->v_caches[l].begin(), _impl->v_caches[l].end(), 0.0f);
    }
}

buffer<bf16> maple_npu::get_k_cache(int layer_idx, int idx) {
    if (layer_idx >= 0 && layer_idx < static_cast<int>(_impl->num_layers)) {
        size_t kv_dim = _impl->num_kv_heads * _impl->head_dim;
        size_t slot = _impl->is_sliding_layer[layer_idx] ? (static_cast<size_t>(idx) % _impl->sliding_window) : static_cast<size_t>(idx);
        size_t offset = slot * kv_dim;
        if (offset + kv_dim <= _impl->k_caches[layer_idx].size()) {
            return buffer<bf16>(reinterpret_cast<bf16*>(&_impl->k_caches[layer_idx][offset]), kv_dim);
        }
    }
    return buffer<bf16>();
}

buffer<bf16> maple_npu::get_v_cache(int layer_idx, int idx) {
    if (layer_idx >= 0 && layer_idx < static_cast<int>(_impl->num_layers)) {
        size_t kv_dim = _impl->num_kv_heads * _impl->head_dim;
        size_t slot = _impl->is_sliding_layer[layer_idx] ? (static_cast<size_t>(idx) % _impl->sliding_window) : static_cast<size_t>(idx);
        size_t offset = slot * kv_dim;
        if (offset + kv_dim <= _impl->v_caches[layer_idx].size()) {
            return buffer<bf16>(reinterpret_cast<bf16*>(&_impl->v_caches[layer_idx][offset]), kv_dim);
        }
    }
    return buffer<bf16>();
}

void maple_npu::update_max_length(uint32_t MAX_L) {
    _impl->init_kv_caches(MAX_L);
    _impl->init_rope_tables(MAX_L);
}

int maple_npu::get_current_context_length() {
    return _impl->cur_pos;
}

int maple_npu::checkpoint() {
    _impl->checkpoint_pos = _impl->cur_pos;
    return _impl->checkpoint_pos;
}

int maple_npu::restore() {
    _impl->cur_pos = _impl->checkpoint_pos;
    return _impl->cur_pos;
}
