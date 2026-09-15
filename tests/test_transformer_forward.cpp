#include "flowserve/model.hpp"
#include "flowserve/tokenizer.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace flowserve;

int main() {
    TransformerConfig cfg;
    cfg.vocab_size = 259;
    cfg.hidden_dim = 16;
    cfg.num_layers = 2;
    cfg.num_heads = 2;
    cfg.head_dim = 4;
    cfg.mla_latent_dim = 4;
    cfg.intermediate_dim = 32;
    cfg.max_seq_len = 64;

    TransformerModel model(cfg);
    Tokenizer tok;

    std::string prompt = "Hi";
    auto prompt_tokens = tok.encode(prompt, true); // [BOS, 'H', 'i']
    assert(prompt_tokens.size() == 3);

    std::vector<float> logits(cfg.vocab_size);

    // 1. Prefill 阶段
    int last_token = prompt_tokens.back();
    for (int t : prompt_tokens) {
        model.forward_step(t, logits.data());
    }

    // 验证 logits 数值正常
    for (float v : logits) {
        assert(!std::isnan(v));
        assert(!std::isinf(v));
    }

    // 2. Decode 阶段：自回归生成 5 个真实计算的 token
    std::vector<int> generated_tokens;
    for (int step = 0; step < 5; ++step) {
        int next_token = model.sample_greedy(logits.data());
        assert(next_token >= 0 && static_cast<size_t>(next_token) < cfg.vocab_size);
        generated_tokens.push_back(next_token);
        model.forward_step(next_token, logits.data());
    }

    assert(generated_tokens.size() == 5);
    std::string text_out = tok.decode(generated_tokens);
    std::cout << "[PASS] test_transformer_forward passed! Generated "
              << generated_tokens.size() << " real tokens.\n";
    return 0;
}
