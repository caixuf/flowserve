#pragma once

// ============================================================================
// flowserve::nn - 投机推演决策仲裁机制 (Speculative Decoding & Verification Engine)
// 移植自 kun-cellular 项目的大模型前沿算子
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace flowserve::nn {

struct SpeculativeConfig {
    float acceptance_threshold{0.65f};  // 草稿被接纳的置信度下限
    float confidence_margin{0.15f};     // 草稿胜出动作与次优动作的概率间隔边界
    bool enable_soft_arbitration{false};// 是否采用软插值仲裁而非硬裁决
    float draft_weight{0.20f};          // 软仲裁时草稿所占权重
};

struct SpeculativeTelemetry {
    uint64_t total_decisions{0};       // 总决策次数
    uint64_t accepted_drafts{0};      // 草稿直接被接纳次数 (Hit)
    uint64_t rejected_drafts{0};      // 草稿被否决回退次数 (Miss)
    double acceptance_rate{0.0};       // 投机接纳率 (Alpha = Hit / Total)
    double speedup_ratio{1.0};         // 相比全量验证模型的有效加速比
};

struct SpeculativeDecision {
    int chosen_token{0};               // 最终采纳的 token
    bool draft_accepted{false};        // 本次是否直接采纳草稿决策
    float confidence{0.0f};            // 最终决策置信度
    std::vector<float> arbitrated_probs;// 仲裁后的归一化概率分布
};

inline std::vector<float> compute_softmax(const float* logits, size_t dim) {
    if (!logits || dim == 0) return {};
    float max_l = logits[0];
    for (size_t i = 1; i < dim; ++i) {
        if (logits[i] > max_l) max_l = logits[i];
    }
    std::vector<float> probs(dim);
    float sum_exp = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        probs[i] = std::exp(logits[i] - max_l);
        sum_exp += probs[i];
    }
    float inv_sum = (sum_exp > 1e-7f) ? (1.0f / sum_exp) : 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        probs[i] *= inv_sum;
    }
    return probs;
}

class SpeculativeEngine {
public:
    SpeculativeConfig config;
    SpeculativeTelemetry telemetry;

    explicit SpeculativeEngine(const SpeculativeConfig& cfg = SpeculativeConfig{})
        : config(cfg) {}

    void reset_telemetry() {
        telemetry = SpeculativeTelemetry{};
    }

    SpeculativeDecision arbitrate(const float* draft_logits,
                                  const float* verifier_logits,
                                  size_t vocab_size) {
        if (!draft_logits || !verifier_logits || vocab_size == 0) {
            throw std::invalid_argument("Invalid logits pointer or vocab_size.");
        }

        auto draft_probs = compute_softmax(draft_logits, vocab_size);
        auto verifier_probs = compute_softmax(verifier_logits, vocab_size);

        int draft_best = 0;
        float draft_best_p = draft_probs[0];
        int verifier_best = 0;
        float verifier_best_p = verifier_probs[0];

        for (size_t i = 1; i < vocab_size; ++i) {
            if (draft_probs[i] > draft_best_p) {
                draft_best_p = draft_probs[i];
                draft_best = static_cast<int>(i);
            }
            if (verifier_probs[i] > verifier_best_p) {
                verifier_best_p = verifier_probs[i];
                verifier_best = static_cast<int>(i);
            }
        }

        telemetry.total_decisions++;

        SpeculativeDecision decision;
        bool accept = false;

        // 仲裁准则：如果主模型最高概率 token 与草稿一致，或者草稿在主模型中的概率足够高
        if (draft_best == verifier_best && verifier_best_p >= config.acceptance_threshold) {
            accept = true;
        } else if (verifier_probs[draft_best] >= config.acceptance_threshold) {
            accept = true;
        }

        if (accept) {
            telemetry.accepted_drafts++;
            decision.chosen_token = draft_best;
            decision.draft_accepted = true;
            decision.confidence = verifier_probs[draft_best];
        } else {
            telemetry.rejected_drafts++;
            decision.chosen_token = verifier_best;
            decision.draft_accepted = false;
            decision.confidence = verifier_best_p;
        }

        decision.arbitrated_probs = std::move(verifier_probs);

        if (telemetry.total_decisions > 0) {
            telemetry.acceptance_rate = static_cast<double>(telemetry.accepted_drafts) /
                                       static_cast<double>(telemetry.total_decisions);
        }
        return decision;
    }
};

} // namespace flowserve::nn
