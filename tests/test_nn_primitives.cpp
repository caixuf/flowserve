#include "flowserve/nn/rope.hpp"
#include "flowserve/nn/speculative.hpp"
#include "flowserve/nn/mla.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace flowserve::nn;

static void test_rope() {
    RoPEConfig cfg;
    cfg.dim = 4;
    cfg.base_freq = 10000.0f;
    cfg.max_seq_len = 16;
    RotaryPhaseTable rpt(cfg);

    float in[4] = {1.0f, 0.0f, 0.5f, 0.5f};
    float out0[4];
    float out1[4];

    rpt.apply(in, 0, out0);
    // At pos=0, cos=1, sin=0, so out0 should equal in
    assert(std::fabs(out0[0] - in[0]) < 1e-5f);
    assert(std::fabs(out0[1] - in[1]) < 1e-5f);

    rpt.apply(in, 1, out1);
    // Norm should be preserved under orthogonal 2D rotation
    float norm_in = in[0] * in[0] + in[1] * in[1];
    float norm_out1 = out1[0] * out1[0] + out1[1] * out1[1];
    assert(std::fabs(norm_in - norm_out1) < 1e-5f);
    std::cout << "[PASS] test_rope passed.\n";
}

static void test_speculative() {
    SpeculativeConfig cfg;
    cfg.acceptance_threshold = 0.6f;
    SpeculativeEngine engine(cfg);

    // Case 1: Draft agrees with Verifier on token 2 with high confidence
    float draft_logits[4] = {0.1f, 0.2f, 2.5f, 0.1f};
    float verifier_logits[4] = {0.0f, 0.1f, 3.0f, 0.2f};

    auto dec1 = engine.arbitrate(draft_logits, verifier_logits, 4);
    assert(dec1.draft_accepted == true);
    assert(dec1.chosen_token == 2);
    assert(engine.telemetry.accepted_drafts == 1);

    // Case 2: Draft chooses token 1, but Verifier strongly prefers token 3
    float draft_logits2[4] = {0.1f, 2.8f, 0.1f, 0.2f};
    float verifier_logits2[4] = {0.1f, 0.1f, 0.2f, 4.0f};

    auto dec2 = engine.arbitrate(draft_logits2, verifier_logits2, 4);
    assert(dec2.draft_accepted == false);
    assert(dec2.chosen_token == 3);
    assert(engine.telemetry.rejected_drafts == 1);
    assert(engine.telemetry.acceptance_rate == 0.5);

    std::cout << "[PASS] test_speculative passed.\n";
}

static void test_mla() {
    MLAConfig cfg;
    cfg.in_dim = 16;
    cfg.latent_dim = 4;
    cfg.q_latent_dim = 4;
    cfg.num_heads = 2;
    cfg.head_dim = 4;
    cfg.max_history_len = 8;
    assert(cfg.compression_ratio() == 4); // (2 * 2 * 4) / 4 = 4x compression

    MLAEngine mla(cfg);
    std::vector<float> x1(16, 1.0f);
    std::vector<float> out1(16, 0.0f);

    mla.step(x1.data(), out1.data());
    assert(mla.current_history_len == 1);

    // Step 2
    std::vector<float> x2(16, 0.5f);
    std::vector<float> out2(16, 0.0f);
    mla.step(x2.data(), out2.data());
    assert(mla.current_history_len == 2);

    std::cout << "[PASS] test_mla passed.\n";
}

int main() {
    test_rope();
    test_speculative();
    test_mla();
    return 0;
}
