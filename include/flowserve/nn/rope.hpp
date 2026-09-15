#pragma once

// ============================================================================
// flowserve::nn - 旋转位置编码 (Rotary Position Embedding, RoPE)
// 移植自 kun-cellular 项目的核心前向算子
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace flowserve::nn {

struct RoPEConfig {
    size_t dim{8};              // 旋转嵌入维度 (必须为偶数)
    float base_freq{10000.0f};  // 几何级数基频 base
    size_t max_seq_len{512};    // 预计算最大相对时序窗口
    float scaling_factor{1.0f}; // 扩展上下文长度时的线性插值标度

    void validate() const {
        if (dim == 0 || (dim % 2) != 0) {
            throw std::invalid_argument("RoPE dimension must be a positive even integer.");
        }
        if (base_freq <= 0.0f) {
            throw std::invalid_argument("RoPE base frequency must be positive.");
        }
    }
};

class RotaryPhaseTable {
public:
    RoPEConfig config;
    std::vector<float> cos_table; // [max_seq_len, dim / 2]
    std::vector<float> sin_table; // [max_seq_len, dim / 2]
    std::vector<float> inv_freq;  // [dim / 2]

    explicit RotaryPhaseTable(const RoPEConfig& cfg) : config(cfg) {
        config.validate();
        init();
    }

    void init() {
        const size_t half_dim = config.dim / 2;
        inv_freq.resize(half_dim);

        for (size_t i = 0; i < half_dim; ++i) {
            double exponent = static_cast<double>(2 * i) / static_cast<double>(config.dim);
            inv_freq[i] = static_cast<float>(1.0 / std::pow(static_cast<double>(config.base_freq), exponent));
        }

        const size_t max_len = config.max_seq_len;
        cos_table.resize(max_len * half_dim);
        sin_table.resize(max_len * half_dim);

        for (size_t pos = 0; pos < max_len; ++pos) {
            float scaled_pos = static_cast<float>(pos) / config.scaling_factor;
            for (size_t i = 0; i < half_dim; ++i) {
                float angle = scaled_pos * inv_freq[i];
                cos_table[pos * half_dim + i] = std::cos(angle);
                sin_table[pos * half_dim + i] = std::sin(angle);
            }
        }
    }

    void apply(const float* in_vec, size_t pos, float* out_vec) const {
        if (!in_vec || !out_vec) return;
        size_t p = std::min(pos, config.max_seq_len - 1);
        const size_t half_dim = config.dim / 2;
        const float* cos_row = &cos_table[p * half_dim];
        const float* sin_row = &sin_table[p * half_dim];

        for (size_t i = 0; i < half_dim; ++i) {
            float u0 = in_vec[2 * i];
            float u1 = in_vec[2 * i + 1];
            float c = cos_row[i];
            float s = sin_row[i];
            out_vec[2 * i]     = u0 * c - u1 * s;
            out_vec[2 * i + 1] = u0 * s + u1 * c;
        }
    }
};

} // namespace flowserve::nn
