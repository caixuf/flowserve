#pragma once

// ============================================================================
// flowserve::nn - 多头低秩潜空间注意力 (Multi-Head Latent Attention, MLA)
// 移植自 kun-cellular 项目。借鉴自 DeepSeek-V2 / DeepSeek-V3 架构
//
// 核心优势:
//   通过 W_DKV 将 Key/Value 压缩进低秩潜空间向量 c_t (如 1/4 ~ 1/8 维度)，
//   KV Cache 显存占用直接缩减至传统 MHA/GQA 的 1/4 ~ 1/8。
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

namespace flowserve::nn {

struct MLAConfig {
    size_t in_dim{32};          // 输入维度 D_in
    size_t latent_dim{8};       // KV 压缩潜空间维度 D_c (D_c << D_in)
    size_t q_latent_dim{8};     // Q 压缩潜空间维度 D_cq
    size_t num_heads{2};        // 多头注意力头数 H
    size_t head_dim{8};         // 每头特征维度 d_k = d_v
    size_t max_history_len{64}; // 最大时序历史缓存容量
    float scale{0.0f};          // 缩放因子 1 / sqrt(d_k)

    void validate() const {
        if (in_dim == 0 || latent_dim == 0 || num_heads == 0 || head_dim == 0 || max_history_len == 0) {
            throw std::invalid_argument("MLAConfig parameters must be positive.");
        }
    }

    size_t compression_ratio() const {
        // 传统 KV 每 step 存储: 2 * num_heads * head_dim
        // MLA 仅存储: latent_dim
        size_t traditional_kv = 2 * num_heads * head_dim;
        return (latent_dim > 0) ? std::max<size_t>(1, traditional_kv / latent_dim) : 1;
    }
};

struct MLAWeights {
    std::vector<float> w_dkv; // [latent_dim, in_dim]
    std::vector<float> w_dq;  // [q_latent_dim, in_dim]
    std::vector<float> w_uk;  // [num_heads * head_dim, latent_dim]
    std::vector<float> w_uv;  // [num_heads * head_dim, latent_dim]
    std::vector<float> w_uq;  // [num_heads * head_dim, q_latent_dim]
    std::vector<float> w_o;   // [in_dim, num_heads * head_dim]

    void init_xavier(const MLAConfig& cfg, uint32_t seed = 42) {
        std::mt19937 rng(seed);
        auto init_mat = [&](std::vector<float>& mat, size_t rows, size_t cols) {
            mat.resize(rows * cols);
            float limit = std::sqrt(6.0f / static_cast<float>(rows + cols));
            std::uniform_real_distribution<float> dist(-limit, limit);
            for (auto& val : mat) val = dist(rng);
        };

        init_mat(w_dkv, cfg.latent_dim, cfg.in_dim);
        init_mat(w_dq, cfg.q_latent_dim, cfg.in_dim);
        init_mat(w_uk, cfg.num_heads * cfg.head_dim, cfg.latent_dim);
        init_mat(w_uv, cfg.num_heads * cfg.head_dim, cfg.latent_dim);
        init_mat(w_uq, cfg.num_heads * cfg.head_dim, cfg.q_latent_dim);
        init_mat(w_o, cfg.in_dim, cfg.num_heads * cfg.head_dim);
    }
};

class MLAEngine {
public:
    MLAConfig config;
    MLAWeights weights;
    float inv_sqrt_dk{1.0f};

    // 低秩潜空间历史记忆缓存 (Latent KV Cache): [max_history_len, latent_dim]
    std::vector<float> latent_kv_cache;
    size_t current_history_len{0};

    explicit MLAEngine(const MLAConfig& cfg) : config(cfg) {
        config.validate();
        inv_sqrt_dk = (cfg.scale > 0.0f) ? cfg.scale : (1.0f / std::sqrt(static_cast<float>(cfg.head_dim)));
        weights.init_xavier(config);
        latent_kv_cache.assign(config.max_history_len * config.latent_dim, 0.0f);
        current_history_len = 0;
    }

    void reset_memory() {
        std::fill(latent_kv_cache.begin(), latent_kv_cache.end(), 0.0f);
        current_history_len = 0;
    }

    // Step 前向计算：将输入向量 x_t 经低秩压缩后存入潜缓存，并计算多头因果注意力输出
    void step(const float* x_t, float* out) {
        if (!x_t || !out) return;

        // 1. 压缩存储当前步的 KV 潜向量: c_t = W_DKV * x_t
        size_t t_idx = std::min(current_history_len, config.max_history_len - 1);
        float* c_t = &latent_kv_cache[t_idx * config.latent_dim];
        for (size_t i = 0; i < config.latent_dim; ++i) {
            float sum = 0.0f;
            for (size_t j = 0; j < config.in_dim; ++j) {
                sum += weights.w_dkv[i * config.in_dim + j] * x_t[j];
            }
            c_t[i] = sum;
        }
        if (current_history_len < config.max_history_len) {
            current_history_len++;
        }

        // 2. 压缩与多头投射 Query: c_q = W_DQ * x_t, Q_h = W_UQ * c_q
        std::vector<float> c_q(config.q_latent_dim, 0.0f);
        for (size_t i = 0; i < config.q_latent_dim; ++i) {
            float sum = 0.0f;
            for (size_t j = 0; j < config.in_dim; ++j) {
                sum += weights.w_dq[i * config.in_dim + j] * x_t[j];
            }
            c_q[i] = sum;
        }

        const size_t total_head_dim = config.num_heads * config.head_dim;
        std::vector<float> q(total_head_dim, 0.0f);
        for (size_t i = 0; i < total_head_dim; ++i) {
            float sum = 0.0f;
            for (size_t j = 0; j < config.q_latent_dim; ++j) {
                sum += weights.w_uq[i * config.q_latent_dim + j] * c_q[j];
            }
            q[i] = sum;
        }

        // 3. 因果多头注意力检索与融合
        std::vector<float> head_outputs(total_head_dim, 0.0f);

        for (size_t h = 0; h < config.num_heads; ++h) {
            const float* q_h = &q[h * config.head_dim];
            std::vector<float> attn_scores(current_history_len, 0.0f);
            float max_score = -1e9f;

            // 对历史各步潜向量展开 Key 并点积
            for (size_t s = 0; s < current_history_len; ++s) {
                const float* c_s = &latent_kv_cache[s * config.latent_dim];
                float dot = 0.0f;
                for (size_t d = 0; d < config.head_dim; ++d) {
                    size_t w_row = h * config.head_dim + d;
                    float k_val = 0.0f;
                    for (size_t l = 0; l < config.latent_dim; ++l) {
                        k_val += weights.w_uk[w_row * config.latent_dim + l] * c_s[l];
                    }
                    dot += q_h[d] * k_val;
                }
                dot *= inv_sqrt_dk;
                attn_scores[s] = dot;
                if (dot > max_score) max_score = dot;
            }

            // Softmax
            float sum_exp = 0.0f;
            for (size_t s = 0; s < current_history_len; ++s) {
                attn_scores[s] = std::exp(attn_scores[s] - max_score);
                sum_exp += attn_scores[s];
            }
            float inv_sum = (sum_exp > 1e-7f) ? (1.0f / sum_exp) : 0.0f;
            for (size_t s = 0; s < current_history_len; ++s) {
                attn_scores[s] *= inv_sum;
            }

            // 加权聚合 Value
            float* h_out = &head_outputs[h * config.head_dim];
            for (size_t s = 0; s < current_history_len; ++s) {
                const float* c_s = &latent_kv_cache[s * config.latent_dim];
                float score = attn_scores[s];
                for (size_t d = 0; d < config.head_dim; ++d) {
                    size_t w_row = h * config.head_dim + d;
                    float v_val = 0.0f;
                    for (size_t l = 0; l < config.latent_dim; ++l) {
                        v_val += weights.w_uv[w_row * config.latent_dim + l] * c_s[l];
                    }
                    h_out[d] += score * v_val;
                }
            }
        }

        // 4. 输出全连接投射 W_O
        for (size_t i = 0; i < config.in_dim; ++i) {
            float sum = 0.0f;
            for (size_t j = 0; j < total_head_dim; ++j) {
                sum += weights.w_o[i * total_head_dim + j] * head_outputs[j];
            }
            out[i] = sum;
        }
    }
};

} // namespace flowserve::nn
