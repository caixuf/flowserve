#pragma once

// ============================================================================
// flowserve - 真实多层 MLA Transformer 前向执行器 (Model Executor)
//
// 架构体系:
//   Embedding -> N x [ RMSNorm -> MLA Attention -> Residual -> RMSNorm -> SwiGLU FFN -> Residual ]
//             -> Final RMSNorm -> LM Head -> Logits -> Sampler
//
// 核心特性:
//   1. 原生 MLA (Multi-Head Latent Attention): 低秩潜空间 KV 缓存，极大降低显存开销；
//   2. 真实线性代数: 逐层矩阵乘法、SwiGLU 非线性激活、RMSNorm 浮点标准化；
//   3. 原生支持加载 WeightLoader mmap 权重或内置初始化。
// ============================================================================

#include "flowserve/nn/mla.hpp"
#include "flowserve/nn/rope.hpp"
#include "flowserve/weight_loader.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace flowserve {

struct TransformerConfig {
    size_t vocab_size{259};     // 词表大小 (对接 Byte-level Tokenizer)
    size_t hidden_dim{32};      // 隐藏层特征维度
    size_t num_layers{2};       // Transformer 层数
    size_t num_heads{2};        // 注意力头数
    size_t head_dim{8};         // 每头特征维度
    size_t mla_latent_dim{8};   // MLA KV 潜空间维度
    size_t intermediate_dim{64};// FFN 中间维度 (约 2~3 倍 hidden_dim)
    size_t max_seq_len{256};    // 最大上下文长度
    float rms_norm_eps{1e-5f};  // RMSNorm epsilon
};

inline void rms_norm(const float* in, const float* weight, float* out, size_t dim, float eps = 1e-5f) {
    float sum_sq = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        sum_sq += in[i] * in[i];
    }
    float scale = 1.0f / std::sqrt((sum_sq / static_cast<float>(dim)) + eps);
    for (size_t i = 0; i < dim; ++i) {
        out[i] = in[i] * scale * (weight ? weight[i] : 1.0f);
    }
}

inline float silu(float x) {
    return x / (1.0f + std::exp(-x));
}

// 单层 Transformer Block 权重与状态
struct TransformerLayer {
    nn::MLAEngine mla_engine;
    std::vector<float> norm1_weight; // [hidden_dim]
    std::vector<float> norm2_weight; // [hidden_dim]

    // SwiGLU FFN 权重
    std::vector<float> w_gate; // [intermediate_dim, hidden_dim]
    std::vector<float> w_up;   // [intermediate_dim, hidden_dim]
    std::vector<float> w_down; // [hidden_dim, intermediate_dim]

    explicit TransformerLayer(const TransformerConfig& cfg)
        : mla_engine([&]() {
              nn::MLAConfig mcfg;
              mcfg.in_dim = cfg.hidden_dim;
              mcfg.latent_dim = cfg.mla_latent_dim;
              mcfg.q_latent_dim = cfg.mla_latent_dim;
              mcfg.num_heads = cfg.num_heads;
              mcfg.head_dim = cfg.head_dim;
              mcfg.max_history_len = cfg.max_seq_len;
              return mcfg;
          }()) {
        norm1_weight.assign(cfg.hidden_dim, 1.0f);
        norm2_weight.assign(cfg.hidden_dim, 1.0f);

        std::mt19937 rng(1337);
        auto init_w = [&](std::vector<float>& w, size_t rows, size_t cols) {
            w.resize(rows * cols);
            float limit = std::sqrt(2.0f / static_cast<float>(rows + cols));
            std::uniform_real_distribution<float> dist(-limit, limit);
            for (auto& v : w) v = dist(rng);
        };

        init_w(w_gate, cfg.intermediate_dim, cfg.hidden_dim);
        init_w(w_up, cfg.intermediate_dim, cfg.hidden_dim);
        init_w(w_down, cfg.hidden_dim, cfg.intermediate_dim);
    }

    void reset_state() {
        mla_engine.reset_memory();
    }
};

class TransformerModel {
public:
    TransformerConfig config;
    std::vector<float> token_embeddings; // [vocab_size, hidden_dim]
    std::vector<TransformerLayer> layers;
    std::vector<float> final_norm_weight;// [hidden_dim]
    std::vector<float> lm_head;          // [vocab_size, hidden_dim]

    explicit TransformerModel(const TransformerConfig& cfg)
        : config(cfg) {
        init_weights();
    }

    void load_from_weights(const WeightLoader& loader) {
        const auto& hdr = loader.header();
        config.num_layers = hdr.num_layers;
        config.hidden_dim = hdr.hidden_dim;
        config.num_heads = hdr.num_heads;
        config.head_dim = hdr.head_dim;
        config.mla_latent_dim = hdr.mla_latent_dim;
        config.intermediate_dim = hdr.intermediate_dim;
        config.vocab_size = hdr.vocab_size;

        const size_t H = config.hidden_dim;
        const size_t V = config.vocab_size;
        const size_t L = config.mla_latent_dim;
        const size_t heads = config.num_heads;
        const size_t d = config.head_dim;
        const size_t total_head_dim = heads * d;
        const size_t M = config.intermediate_dim;

        token_embeddings.assign(loader.token_embeddings(), loader.token_embeddings() + (V * H));
        final_norm_weight.assign(loader.final_norm(), loader.final_norm() + H);
        lm_head.assign(loader.lm_head(), loader.lm_head() + (V * H));

        layers.clear();
        layers.reserve(config.num_layers);
        for (size_t l = 0; l < config.num_layers; ++l) {
            layers.emplace_back(config);
            const auto& lw = loader.layer(l);

            layers[l].norm1_weight.assign(lw.norm1, lw.norm1 + H);
            layers[l].norm2_weight.assign(lw.norm2, lw.norm2 + H);

            layers[l].mla_engine.weights.w_dkv.assign(lw.w_dkv, lw.w_dkv + (L * H));
            layers[l].mla_engine.weights.w_dq.assign(lw.w_dq, lw.w_dq + (L * H));
            layers[l].mla_engine.weights.w_uk.assign(lw.w_uk, lw.w_uk + (total_head_dim * L));
            layers[l].mla_engine.weights.w_uv.assign(lw.w_uv, lw.w_uv + (total_head_dim * L));
            layers[l].mla_engine.weights.w_uq.assign(lw.w_uq, lw.w_uq + (total_head_dim * L));
            layers[l].mla_engine.weights.w_o.assign(lw.w_o, lw.w_o + (H * total_head_dim));

            layers[l].w_gate.assign(lw.w_gate, lw.w_gate + (M * H));
            layers[l].w_up.assign(lw.w_up, lw.w_up + (M * H));
            layers[l].w_down.assign(lw.w_down, lw.w_down + (H * M));
        }
    }

    void reset_state() {
        for (auto& l : layers) l.reset_state();
    }

    // 单步前向自回归推演：输入前一个 token，输出对下一个 token 的 Logits [vocab_size]
    void forward_step(int token_id, float* logits_out) {
        const size_t H = config.hidden_dim;
        const size_t V = config.vocab_size;

        if (token_id < 0 || static_cast<size_t>(token_id) >= V) {
            token_id = 0;
        }

        // 1. Embedding 查表取向量
        std::vector<float> h(H);
        const float* emb = &token_embeddings[static_cast<size_t>(token_id) * H];
        std::copy(emb, emb + H, h.begin());

        std::vector<float> norm_buf(H);
        std::vector<float> attn_out(H);
        std::vector<float> ffn_out(H);

        // 2. 逐层 Transformer 前向
        for (auto& layer : layers) {
            // (a) Pre-Norm
            rms_norm(h.data(), layer.norm1_weight.data(), norm_buf.data(), H, config.rms_norm_eps);

            // (b) MLA 潜空间多头注意力
            layer.mla_engine.step(norm_buf.data(), attn_out.data());

            // (c) 残差连接
            for (size_t i = 0; i < H; ++i) h[i] += attn_out[i];

            // (d) Pre-Norm 2
            rms_norm(h.data(), layer.norm2_weight.data(), norm_buf.data(), H, config.rms_norm_eps);

            // (e) SwiGLU FFN
            const size_t M = config.intermediate_dim;
            std::vector<float> gate(M, 0.0f);
            std::vector<float> up(M, 0.0f);
            for (size_t r = 0; r < M; ++r) {
                float sum_g = 0.0f, sum_u = 0.0f;
                for (size_t c = 0; c < H; ++c) {
                    sum_g += layer.w_gate[r * H + c] * norm_buf[c];
                    sum_u += layer.w_up[r * H + c] * norm_buf[c];
                }
                gate[r] = silu(sum_g);
                up[r] = sum_u;
            }

            // Down projection
            for (size_t i = 0; i < H; ++i) {
                float sum_d = 0.0f;
                for (size_t r = 0; r < M; ++r) {
                    sum_d += layer.w_down[i * M + r] * (gate[r] * up[r]);
                }
                ffn_out[i] = sum_d;
            }

            // (f) 残差连接 2
            for (size_t i = 0; i < H; ++i) h[i] += ffn_out[i];
        }

        // 3. Final RMSNorm
        std::vector<float> final_h(H);
        rms_norm(h.data(), final_norm_weight.data(), final_h.data(), H, config.rms_norm_eps);

        // 4. LM Head 输出投射
        for (size_t v = 0; v < V; ++v) {
            float sum = 0.0f;
            for (size_t c = 0; c < H; ++c) {
                sum += lm_head[v * H + c] * final_h[c];
            }
            logits_out[v] = sum;
        }
    }

    // 贪心采样预测下一个 token
    int sample_greedy(const float* logits) const {
        const size_t V = config.vocab_size;
        int best_tok = 0;
        float best_val = logits[0];
        for (size_t i = 1; i < V; ++i) {
            if (logits[i] > best_val) {
                best_val = logits[i];
                best_tok = static_cast<int>(i);
            }
        }
        return best_tok;
    }

    // 带重复惩罚的采样（仅对 ASCII 单字符/空格生效，保护 UTF-8 多字节汉字结构不被破坏）
    int sample_with_penalty(const float* logits, const std::vector<int>& recent_tokens, float penalty = 1.3f) const {
        const size_t V = config.vocab_size;
        std::vector<float> mod_logits(logits, logits + V);
        for (int tok : recent_tokens) {
            if (tok >= 0 && static_cast<size_t>(tok) < V) {
                // Tokenizer::BYTE_OFFSET = 3, 3..130 为 ASCII 字符。>=131 为 UTF-8 多字节分片，严禁盲目惩罚
                if (tok <= 130) {
                    if (mod_logits[tok] > 0.0f) mod_logits[tok] /= penalty;
                    else mod_logits[tok] *= penalty;
                }
            }
        }
        int best_tok = 0;
        float best_val = mod_logits[0];
        for (size_t i = 1; i < V; ++i) {
            if (mod_logits[i] > best_val) {
                best_val = mod_logits[i];
                best_tok = static_cast<int>(i);
            }
        }
        return best_tok;
    }

private:
    void init_weights() {
        std::mt19937 rng(42);
        auto init_mat = [&](std::vector<float>& mat, size_t size, float stddev) {
            mat.resize(size);
            std::normal_distribution<float> dist(0.0f, stddev);
            for (auto& val : mat) val = dist(rng);
        };

        const size_t H = config.hidden_dim;
        const size_t V = config.vocab_size;

        init_mat(token_embeddings, V * H, 0.02f);
        final_norm_weight.assign(H, 1.0f);
        init_mat(lm_head, V * H, 0.02f);

        layers.reserve(config.num_layers);
        for (size_t l = 0; l < config.num_layers; ++l) {
            layers.emplace_back(config);
        }
    }
};

} // namespace flowserve
