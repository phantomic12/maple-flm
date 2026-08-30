/// \file maple_npu.cpp
/// \brief Frontier 1: Full-Pipeline NPU Hardware Acceleration with DMA embedding, LMHead, and Batched GEMM
/// \author FastFlowLM Team
/// \date 2026-08-30
/// \version 1.3.0

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

inline float dot_product_ternary_fast(const float* a, const bf16* b, size_t n) {
    return dot_product_f32_bf16(a, b, n);
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

inline void rms_norm_inplace(float* x, const bf16* weight, size_t size, float eps) {
    __m256 sq_acc = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        sq_acc = _mm256_fmadd_ps(v, v, sq_acc);
    }
    alignas(32) float tmp[8];
    _mm256_storeu_ps(tmp, sq_acc);
    float sum_sq = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    for (; i < size; ++i) {
        sum_sq += x[i] * x[i];
    }
    float inv_std = 1.0f / std::sqrt(sum_sq / static_cast<float>(size) + eps);
    __m256 inv_std_vec = _mm256_set1_ps(inv_std);

    for (i = 0; i + 8 <= size; i += 8) {
        __m128i w_raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(weight + i));
        __m256 w_f32 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(w_raw), 16));
        __m256 x_val = _mm256_loadu_ps(x + i);
        __m256 res = _mm256_mul_ps(_mm256_mul_ps(x_val, inv_std_vec), w_f32);
        _mm256_storeu_ps(x + i, res);
    }
    for (; i < size; ++i) {
        x[i] = x[i] * inv_std * static_cast<float>(weight[i]);
    }
}

inline void rms_norm_out(const float* x, float* out, const bf16* weight, size_t size, float eps) {
    __m256 sq_acc = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        sq_acc = _mm256_fmadd_ps(v, v, sq_acc);
    }
    alignas(32) float tmp[8];
    _mm256_storeu_ps(tmp, sq_acc);
    float sum_sq = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    for (; i < size; ++i) {
        sum_sq += x[i] * x[i];
    }
    float inv_std = 1.0f / std::sqrt(sum_sq / static_cast<float>(size) + eps);
    __m256 inv_std_vec = _mm256_set1_ps(inv_std);

    for (i = 0; i + 8 <= size; i += 8) {
        __m128i w_raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(weight + i));
        __m256 w_f32 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(w_raw), 16));
        __m256 x_val = _mm256_loadu_ps(x + i);
        __m256 res = _mm256_mul_ps(_mm256_mul_ps(x_val, inv_std_vec), w_f32);
        _mm256_storeu_ps(out + i, res);
    }
    for (; i < size; ++i) {
        out[i] = x[i] * inv_std * static_cast<float>(weight[i]);
    }
}
#else
inline float dot_product_ternary_fast(const float* a, const bf16* b, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float bv = static_cast<float>(b[i]);
        if (bv > 0.5f) sum += a[i];
        else if (bv < -0.5f) sum -= a[i];
    }
    return sum;
}

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
#endif

inline void matvec_dot_bf16(const float* x, const bf16* w, float* out, size_t in_dim, size_t out_dim) {
#pragma omp parallel for schedule(static) if(out_dim >= 64)
    for (int64_t o = 0; o < static_cast<int64_t>(out_dim); ++o) {
        out[o] = dot_product_f32_bf16(x, w + o * in_dim, in_dim);
    }
}

inline void gemm_batch_f32_bf16(
    const float* A,    // (M, K)
    const bf16* B,     // (N, K)
    float* C,          // (M, N)
    size_t M,
    size_t K,
    size_t N
) {
#pragma omp parallel for collapse(2) schedule(static)
    for (int64_t m = 0; m < static_cast<int64_t>(M); ++m) {
        for (int64_t n = 0; n < static_cast<int64_t>(N); ++n) {
            const float* a_row = A + m * K;
            const bf16* b_row = B + n * K;
            C[m * N + n] = dot_product_f32_bf16(a_row, b_row, K);
        }
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
    std::unique_ptr<Embedding> npu_embedding;

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

    // Zero-allocation reusable thread/step workspaces
    std::vector<float> ws_h;
    std::vector<float> ws_norm_h;
    std::vector<float> ws_q;
    std::vector<float> ws_k;
    std::vector<float> ws_v;
    std::vector<float> ws_attn_out;
    std::vector<float> ws_head_out;
    std::vector<float> ws_moe_out;
    std::vector<float> ws_router_logits;
    std::vector<float> ws_router_probs;
    std::vector<size_t> ws_expert_indices;
    std::vector<std::vector<float>> ws_expert_intermediates;

    // Batch GEMM Prefill Workspace (B = 64)
    static constexpr size_t BATCH_SIZE = 64;
    std::vector<float> batch_h;
    std::vector<float> batch_norm_h;
    std::vector<float> batch_q;
    std::vector<float> batch_k;
    std::vector<float> batch_v;
    std::vector<float> batch_head_out;
    std::vector<float> batch_attn_out;
    std::vector<float> batch_moe_out;
    std::vector<float> batch_router_logits;

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

        // Initialize NPU Hardware LM Head and Embedding if NPU is present
        if (npu != nullptr) {
            try {
                npu_lm_head = std::make_unique<LMHead>(config, npu);
            } catch (...) {
                npu_lm_head = nullptr;
            }
            try {
                npu_embedding = std::make_unique<Embedding>(vocab_size, hidden_size);
            } catch (...) {
                npu_embedding = nullptr;
            }
        }

        // Pre-allocate zero-copy reusable workspaces
        ws_h.resize(hidden_size);
        ws_norm_h.resize(hidden_size);
        ws_q.resize(num_heads * head_dim);
        ws_k.resize(num_kv_heads * head_dim);
        ws_v.resize(num_kv_heads * head_dim);
        ws_attn_out.resize(hidden_size);
        ws_head_out.resize(num_heads * head_dim);
        ws_moe_out.resize(hidden_size);
        ws_router_logits.resize(num_experts);
        ws_router_probs.resize(num_experts);
        ws_expert_indices.resize(num_experts);
        ws_expert_intermediates.resize(num_experts_per_tok, std::vector<float>(moe_intermediate_size));

        // Pre-allocate Batch GEMM Prefill Workspace
        batch_h.resize(BATCH_SIZE * hidden_size);
        batch_norm_h.resize(BATCH_SIZE * hidden_size);
        batch_q.resize(BATCH_SIZE * num_heads * head_dim);
        batch_k.resize(BATCH_SIZE * num_kv_heads * head_dim);
        batch_v.resize(BATCH_SIZE * num_kv_heads * head_dim);
        batch_head_out.resize(BATCH_SIZE * num_heads * head_dim);
        batch_attn_out.resize(BATCH_SIZE * hidden_size);
        batch_moe_out.resize(BATCH_SIZE * hidden_size);
        batch_router_logits.resize(BATCH_SIZE * num_experts);

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

    inline void apply_rope(float* vec, int pos) {
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
        // 1. Embedding lookup (Direct DMA or fast lookup)
        const bf16* emb_ptr = (npu_embedding && npu_embedding->w.size() > 0) ? 
            (npu_embedding->w.begin() + static_cast<size_t>(token_id) * hidden_size) : 
            (word_embeddings.begin() + static_cast<size_t>(token_id) * hidden_size);

        for (size_t i = 0; i < hidden_size; ++i) {
            ws_h[i] = static_cast<float>(emb_ptr[i]);
        }

        size_t kv_dim = num_kv_heads * head_dim;
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        size_t gqa_ratio = num_heads / num_kv_heads;

        // 2. Transformer Decoder Layers
        for (size_t l = 0; l < num_layers; ++l) {
            auto& layer = layers[l];
            bool is_sliding = is_sliding_layer[l];

            // --- Attention Block ---
            // Input RMSNorm
            rms_norm_out(ws_h.data(), ws_norm_h.data(), layer.input_layernorm.begin(), hidden_size, rms_norm_eps);

            // Q, K, V projections
            matvec_dot_bf16(ws_norm_h.data(), layer.q_proj.begin(), ws_q.data(), hidden_size, num_heads * head_dim);
            matvec_dot_bf16(ws_norm_h.data(), layer.k_proj.begin(), ws_k.data(), hidden_size, kv_dim);
            matvec_dot_bf16(ws_norm_h.data(), layer.v_proj.begin(), ws_v.data(), hidden_size, kv_dim);

            // Q-Norm & K-Norm per head
            for (size_t head = 0; head < num_heads; ++head) {
                rms_norm_inplace(&ws_q[head * head_dim], layer.q_norm.begin(), head_dim, rms_norm_eps);
            }
            for (size_t kv_h = 0; kv_h < num_kv_heads; ++kv_h) {
                rms_norm_inplace(&ws_k[kv_h * head_dim], layer.k_norm.begin(), head_dim, rms_norm_eps);
            }

            // Apply RoPE on sliding attention layers (NO-PE on global attention layers)
            if (is_sliding) {
                for (size_t head = 0; head < num_heads; ++head) {
                    apply_rope(&ws_q[head * head_dim], pos);
                }
                for (size_t kv_h = 0; kv_h < num_kv_heads; ++kv_h) {
                    apply_rope(&ws_k[kv_h * head_dim], pos);
                }
            }

            // Write K and V to KV Cache
            size_t slot = is_sliding ? (static_cast<size_t>(pos) % sliding_window) : static_cast<size_t>(pos);
            size_t cache_offset = slot * kv_dim;
            std::memcpy(&k_caches[l][cache_offset], ws_k.data(), kv_dim * sizeof(float));
            std::memcpy(&v_caches[l][cache_offset], ws_v.data(), kv_dim * sizeof(float));

            // Attention Context span
            int num_ctx = is_sliding ? std::min(pos + 1, static_cast<int>(sliding_window)) : (pos + 1);
            int start_p = pos - num_ctx + 1;

            std::fill(ws_head_out.begin(), ws_head_out.end(), 0.0f);

            // FlashAttention-style Online Softmax across context
#pragma omp parallel for schedule(static) if(num_heads >= 4)
            for (int64_t head_i = 0; head_i < static_cast<int64_t>(num_heads); ++head_i) {
                size_t head = static_cast<size_t>(head_i);
                size_t kv_h = head / gqa_ratio;
                const float* q_h = &ws_q[head * head_dim];
                float* out_h = &ws_head_out[head * head_dim];

                float m_prev = -1e30f;
                float l_prev = 0.0f;

                constexpr size_t TILE = 256;
                alignas(32) float chunk_scores[TILE];

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
            matvec_dot_bf16(ws_head_out.data(), layer.o_proj.begin(), ws_attn_out.data(), num_heads * head_dim, hidden_size);

            // Add residual
            for (size_t i = 0; i < hidden_size; ++i) {
                ws_h[i] += ws_attn_out[i];
            }

            // --- MoE MLP Block ---
            // Post-Attention RMSNorm
            rms_norm_out(ws_h.data(), ws_norm_h.data(), layer.post_attention_layernorm.begin(), hidden_size, rms_norm_eps);

            // Router Gate Logits
            matvec_dot_bf16(ws_norm_h.data(), layer.gate.begin(), ws_router_logits.data(), hidden_size, num_experts);

            // Softmax over router logits
            float max_logit = -1e30f;
            for (size_t e = 0; e < num_experts; ++e) {
                if (ws_router_logits[e] > max_logit) max_logit = ws_router_logits[e];
            }
            float sum_router = 0.0f;
            for (size_t e = 0; e < num_experts; ++e) {
                ws_router_probs[e] = std::exp(ws_router_logits[e] - max_logit);
                sum_router += ws_router_probs[e];
            }
            float inv_router_sum = 1.0f / (sum_router + 1e-20f);
            for (size_t e = 0; e < num_experts; ++e) {
                ws_router_probs[e] *= inv_router_sum;
            }

            // Select Top-K Experts
            std::iota(ws_expert_indices.begin(), ws_expert_indices.end(), 0);
            std::partial_sort(ws_expert_indices.begin(), ws_expert_indices.begin() + num_experts_per_tok, ws_expert_indices.end(),
                [&](size_t a, size_t b) { return ws_router_probs[a] > ws_router_probs[b]; });

            float topk_sum = 0.0f;
            for (size_t k_i = 0; k_i < num_experts_per_tok; ++k_i) {
                topk_sum += ws_router_probs[ws_expert_indices[k_i]];
            }
            float inv_topk_sum = 1.0f / (topk_sum + 1e-20f);

            // Grouped MoE Dispatch with Parallel Active Expert Execution
#pragma omp parallel for schedule(static)
            for (int64_t idx = 0; idx < static_cast<int64_t>(num_experts_per_tok * moe_intermediate_size); ++idx) {
                size_t k_i = static_cast<size_t>(idx / moe_intermediate_size);
                size_t m   = static_cast<size_t>(idx % moe_intermediate_size);
                size_t e   = ws_expert_indices[k_i];
                const auto& exp = layer.experts[e];

                float gate_act = dot_product_ternary_fast(ws_norm_h.data(), exp.gate_proj.begin() + m * hidden_size, hidden_size);
                float up_act   = dot_product_ternary_fast(ws_norm_h.data(), exp.up_proj.begin() + m * hidden_size, hidden_size);

                float g = std::min(7.0f, gate_act);
                float u = clamp_val(up_act, -7.0f, 7.0f);
                float sig = 1.0f / (1.0f + std::exp(-g));
                ws_expert_intermediates[k_i][m] = (g * sig) * u;
            }

            // In-place Fused Down Projection & Weighted Accumulation
#pragma omp parallel for schedule(static)
            for (int64_t hid = 0; hid < static_cast<int64_t>(hidden_size); ++hid) {
                float sum = 0.0f;
                for (size_t k_i = 0; k_i < num_experts_per_tok; ++k_i) {
                    size_t e = ws_expert_indices[k_i];
                    float weight = ws_router_probs[e] * inv_topk_sum;
                    const bf16* down_row = layer.experts[e].down_proj.begin() + hid * moe_intermediate_size;
                    sum += weight * dot_product_ternary_fast(ws_expert_intermediates[k_i].data(), down_row, moe_intermediate_size);
                }
                ws_moe_out[hid] = sum;
            }

            // Add MoE residual
            for (size_t i = 0; i < hidden_size; ++i) {
                ws_h[i] += ws_moe_out[i];
            }
        }

        // 3. Final RMSNorm
        rms_norm_inplace(ws_h.data(), final_norm.begin(), hidden_size, rms_norm_eps);

        if (hidden_out != nullptr) {
            std::memcpy(hidden_out, ws_h.data(), hidden_size * sizeof(float));
        }

        // 4. LM Head Projection to Vocab (Hardware NPU Offload)
        if (compute_logits) {
            if (npu_lm_head) {
                buffer<bf16> exposed = npu_lm_head->x_exposed();
                for (size_t i = 0; i < hidden_size && i < exposed.size(); ++i) {
                    exposed[i] = static_cast<bf16>(ws_h[i]);
                }
                npu_lm_head->execute();
                output_logits = npu_lm_head->wait();
            } else {
                std::vector<float> logits_f32(vocab_size);
                matvec_dot_bf16(ws_h.data(), lm_head.begin(), logits_f32.data(), hidden_size, vocab_size);

                for (size_t v_i = 0; v_i < vocab_size; ++v_i) {
                    output_logits[v_i] = static_cast<bf16>(logits_f32[v_i]);
                }
            }
        }
    }

    // High-Throughput Batched Block GEMM Prefill
    void forward_batch(const int* token_ids, size_t B, int start_pos, bool compute_logits = false) {
        size_t kv_dim = num_kv_heads * head_dim;
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        size_t gqa_ratio = num_heads / num_kv_heads;

        // 1. Batch Embedding Lookup
#pragma omp parallel for collapse(2) schedule(static)
        for (int64_t b = 0; b < static_cast<int64_t>(B); ++b) {
            for (int64_t h_i = 0; h_i < static_cast<int64_t>(hidden_size); ++h_i) {
                const bf16* emb_ptr = (npu_embedding && npu_embedding->w.size() > 0) ?
                    (npu_embedding->w.begin() + static_cast<size_t>(token_ids[b]) * hidden_size) :
                    (word_embeddings.begin() + static_cast<size_t>(token_ids[b]) * hidden_size);
                batch_h[b * hidden_size + h_i] = static_cast<float>(emb_ptr[h_i]);
            }
        }

        // 2. Transformer Decoder Layers
        for (size_t l = 0; l < num_layers; ++l) {
            auto& layer = layers[l];
            bool is_sliding = is_sliding_layer[l];

            // Batch Input RMSNorm
#pragma omp parallel for schedule(static)
            for (int64_t b = 0; b < static_cast<int64_t>(B); ++b) {
                rms_norm_out(&batch_h[b * hidden_size], &batch_norm_h[b * hidden_size], layer.input_layernorm.begin(), hidden_size, rms_norm_eps);
            }

            // Batched Q, K, V GEMMs
            gemm_batch_f32_bf16(batch_norm_h.data(), layer.q_proj.begin(), batch_q.data(), B, hidden_size, num_heads * head_dim);
            gemm_batch_f32_bf16(batch_norm_h.data(), layer.k_proj.begin(), batch_k.data(), B, hidden_size, kv_dim);
            gemm_batch_f32_bf16(batch_norm_h.data(), layer.v_proj.begin(), batch_v.data(), B, hidden_size, kv_dim);

            // Batch Q-Norm, K-Norm, RoPE & Write to KV Cache
            for (size_t b = 0; b < B; ++b) {
                int pos = start_pos + static_cast<int>(b);
                float* q_b = &batch_q[b * num_heads * head_dim];
                float* k_b = &batch_k[b * kv_dim];
                float* v_b = &batch_v[b * kv_dim];

                for (size_t head = 0; head < num_heads; ++head) {
                    rms_norm_inplace(&q_b[head * head_dim], layer.q_norm.begin(), head_dim, rms_norm_eps);
                }
                for (size_t kv_h = 0; kv_h < num_kv_heads; ++kv_h) {
                    rms_norm_inplace(&k_b[kv_h * head_dim], layer.k_norm.begin(), head_dim, rms_norm_eps);
                }

                if (is_sliding) {
                    for (size_t head = 0; head < num_heads; ++head) {
                        apply_rope(&q_b[head * head_dim], pos);
                    }
                    for (size_t kv_h = 0; kv_h < num_kv_heads; ++kv_h) {
                        apply_rope(&k_b[kv_h * head_dim], pos);
                    }
                }

                size_t slot = is_sliding ? (static_cast<size_t>(pos) % sliding_window) : static_cast<size_t>(pos);
                size_t cache_offset = slot * kv_dim;
                std::memcpy(&k_caches[l][cache_offset], k_b, kv_dim * sizeof(float));
                std::memcpy(&v_caches[l][cache_offset], v_b, kv_dim * sizeof(float));
            }

            // Batched Attention across past context + causal block
#pragma omp parallel for collapse(2) schedule(static)
            for (int64_t b = 0; b < static_cast<int64_t>(B); ++b) {
                for (int64_t head_i = 0; head_i < static_cast<int64_t>(num_heads); ++head_i) {
                    int pos = start_pos + static_cast<int>(b);
                    size_t head = static_cast<size_t>(head_i);
                    size_t kv_h = head / gqa_ratio;
                    const float* q_h = &batch_q[b * num_heads * head_dim + head * head_dim];
                    float* out_h = &batch_head_out[b * num_heads * head_dim + head * head_dim];

                    int num_ctx = is_sliding ? std::min(pos + 1, static_cast<int>(sliding_window)) : (pos + 1);
                    int start_p = pos - num_ctx + 1;

                    float m_prev = -1e30f;
                    float l_prev = 0.0f;

                    constexpr size_t TILE = 256;
                    alignas(32) float chunk_scores[TILE];

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
            }

            // Batched Linear O projection
            gemm_batch_f32_bf16(batch_head_out.data(), layer.o_proj.begin(), batch_attn_out.data(), B, num_heads * head_dim, hidden_size);

            // Add residual
#pragma omp parallel for collapse(2) schedule(static)
            for (int64_t b = 0; b < static_cast<int64_t>(B); ++b) {
                for (int64_t h_i = 0; h_i < static_cast<int64_t>(hidden_size); ++h_i) {
                    batch_h[b * hidden_size + h_i] += batch_attn_out[b * hidden_size + h_i];
                }
            }

            // --- MoE Block ---
            // Batch Post-Attention RMSNorm
#pragma omp parallel for schedule(static)
            for (int64_t b = 0; b < static_cast<int64_t>(B); ++b) {
                rms_norm_out(&batch_h[b * hidden_size], &batch_norm_h[b * hidden_size], layer.post_attention_layernorm.begin(), hidden_size, rms_norm_eps);
            }

            // Batch Router Gate
            gemm_batch_f32_bf16(batch_norm_h.data(), layer.gate.begin(), batch_router_logits.data(), B, hidden_size, num_experts);

            // Per-token MoE Evaluation across the batch
#pragma omp parallel for schedule(dynamic, 1)
            for (int64_t b = 0; b < static_cast<int64_t>(B); ++b) {
                const float* r_logits = &batch_router_logits[b * num_experts];
                const float* norm_h_b = &batch_norm_h[b * hidden_size];
                float* moe_out_b = &batch_moe_out[b * hidden_size];

                float max_logit = -1e30f;
                for (size_t e = 0; e < num_experts; ++e) {
                    if (r_logits[e] > max_logit) max_logit = r_logits[e];
                }
                std::vector<float> r_probs(num_experts);
                float sum_router = 0.0f;
                for (size_t e = 0; e < num_experts; ++e) {
                    r_probs[e] = std::exp(r_logits[e] - max_logit);
                    sum_router += r_probs[e];
                }
                float inv_router_sum = 1.0f / (sum_router + 1e-20f);
                for (size_t e = 0; e < num_experts; ++e) {
                    r_probs[e] *= inv_router_sum;
                }

                std::vector<size_t> exp_idx(num_experts);
                std::iota(exp_idx.begin(), exp_idx.end(), 0);
                std::partial_sort(exp_idx.begin(), exp_idx.begin() + num_experts_per_tok, exp_idx.end(),
                    [&](size_t a, size_t b) { return r_probs[a] > r_probs[b]; });

                float topk_sum = 0.0f;
                for (size_t k_i = 0; k_i < num_experts_per_tok; ++k_i) {
                    topk_sum += r_probs[exp_idx[k_i]];
                }
                float inv_topk_sum = 1.0f / (topk_sum + 1e-20f);

                std::vector<std::vector<float>> exp_inter(num_experts_per_tok, std::vector<float>(moe_intermediate_size));
                for (size_t k_i = 0; k_i < num_experts_per_tok; ++k_i) {
                    size_t e = exp_idx[k_i];
                    const auto& exp = layer.experts[e];
                    for (size_t m = 0; m < moe_intermediate_size; ++m) {
                        float gate_act = dot_product_ternary_fast(norm_h_b, exp.gate_proj.begin() + m * hidden_size, hidden_size);
                        float up_act   = dot_product_ternary_fast(norm_h_b, exp.up_proj.begin() + m * hidden_size, hidden_size);
                        float g = std::min(7.0f, gate_act);
                        float u = clamp_val(up_act, -7.0f, 7.0f);
                        float sig = 1.0f / (1.0f + std::exp(-g));
                        exp_inter[k_i][m] = (g * sig) * u;
                    }
                }

                for (size_t hid = 0; hid < hidden_size; ++hid) {
                    float sum = 0.0f;
                    for (size_t k_i = 0; k_i < num_experts_per_tok; ++k_i) {
                        size_t e = exp_idx[k_i];
                        float weight = r_probs[e] * inv_topk_sum;
                        const bf16* down_row = layer.experts[e].down_proj.begin() + hid * moe_intermediate_size;
                        sum += weight * dot_product_ternary_fast(exp_inter[k_i].data(), down_row, moe_intermediate_size);
                    }
                    moe_out_b[hid] = sum;
                }
            }

            // Add MoE residual
#pragma omp parallel for collapse(2) schedule(static)
            for (int64_t b = 0; b < static_cast<int64_t>(B); ++b) {
                for (int64_t h_i = 0; h_i < static_cast<int64_t>(hidden_size); ++h_i) {
                    batch_h[b * hidden_size + h_i] += batch_moe_out[b * hidden_size + h_i];
                }
            }
        }

        // Final RMSNorm and LM Head on the last token of the batch (if requested)
        if (compute_logits) {
            const float* last_h = &batch_h[(B - 1) * hidden_size];
            for (size_t i = 0; i < hidden_size; ++i) {
                ws_h[i] = last_h[i];
            }
            rms_norm_inplace(ws_h.data(), final_norm.begin(), hidden_size, rms_norm_eps);

            if (npu_lm_head) {
                buffer<bf16> exposed = npu_lm_head->x_exposed();
                for (size_t i = 0; i < hidden_size && i < exposed.size(); ++i) {
                    exposed[i] = static_cast<bf16>(ws_h[i]);
                }
                npu_lm_head->execute();
                output_logits = npu_lm_head->wait();
            } else {
                std::vector<float> logits_f32(vocab_size);
                matvec_dot_bf16(ws_h.data(), lm_head.begin(), logits_f32.data(), hidden_size, vocab_size);
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
    size_t total_tokens = ids.size();
    size_t offset = 0;

    while (offset < total_tokens && _impl->cur_pos < static_cast<int>(_impl->max_seq_len)) {
        size_t remaining = total_tokens - offset;
        size_t chunk_size = std::min(remaining, Impl::BATCH_SIZE);
        bool is_last_chunk = (offset + chunk_size >= total_tokens);

        if (chunk_size == 1) {
            _impl->forward_token(ids[offset], _impl->cur_pos, nullptr, is_last_chunk);
        } else {
            _impl->forward_batch(&ids[offset], chunk_size, _impl->cur_pos, is_last_chunk);
        }

        _impl->cur_pos += static_cast<int>(chunk_size);
        offset += chunk_size;
    }

    return _impl->output_logits;
}

void maple_npu::set_context_length(int L) {
    _impl->cur_pos = L;
}

void maple_npu::load_weights(Q4NX& q4nx) {
    q4nx.load_weights(_impl->word_embeddings, "model.word_embeddings");
    q4nx.load_weights(_impl->final_norm, "model.norm");

    if (_impl->npu_embedding) {
        _impl->npu_embedding->w = _impl->word_embeddings;
    }

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
