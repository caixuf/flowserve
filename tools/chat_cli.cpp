#include "flowserve/model.hpp"
#include "flowserve/tokenizer.hpp"
#include <chrono>
#include <iostream>
#include <string>
#include <thread>

using namespace flowserve;

int main(int argc, char** argv) {
    bool raw_mode = false;
    std::string prompt = "Hello, flowserve!";
    std::string weights_path = "";
    int max_gen = 48;

    std::vector<std::string> positional;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--raw") {
            raw_mode = true;
        } else {
            positional.push_back(arg);
        }
    }

    if (!positional.empty()) prompt = positional[0];
    if (positional.size() > 1) weights_path = positional[1];
    if (positional.size() > 2) max_gen = std::stoi(positional[2]);

    if (!raw_mode) {
        std::cout << "========================================================\n";
        std::cout << "  flowserve: Real Multi-Head Latent Attention (MLA) LLM \n";
        std::cout << "========================================================\n";
    }

    TransformerConfig cfg;
    cfg.vocab_size = 259;
    cfg.hidden_dim = 64;
    cfg.num_layers = 2;
    cfg.num_heads = 4;
    cfg.head_dim = 16;
    cfg.mla_latent_dim = 16; // 4x latent KV compression
    cfg.intermediate_dim = 128;
    cfg.max_seq_len = 128;

    TransformerModel model(cfg);
    Tokenizer tok;

    if (!weights_path.empty()) {
        if (!raw_mode) {
            std::cout << "Loading model weights from: " << weights_path << " (via mmap zero-copy)...\n";
        }
        WeightLoader loader = WeightLoader::open_mmap(weights_path);
        model.load_from_weights(loader);
        if (!raw_mode) {
            std::cout << "Weights loaded successfully! Layers=" << model.config.num_layers
                      << ", HiddenDim=" << model.config.hidden_dim << "\n";
        }
    }

    auto prompt_tokens = tok.encode(prompt, true);
    if (!raw_mode) {
        std::cout << "Prompt: \"" << prompt << "\"\n";
        std::cout << "Encoded tokens: [ ";
        for (int t : prompt_tokens) std::cout << t << " ";
        std::cout << "] (count=" << prompt_tokens.size() << ")\n\n";
        std::cout << "Model generation (Real MLA Forward Steps):\n> " << std::flush;
    }

    std::vector<float> logits(cfg.vocab_size);

    // 1. Prefill
    for (int t : prompt_tokens) {
        model.forward_step(t, logits.data());
    }

    // 2. Decode 流式输出
    for (int step = 0; step < max_gen; ++step) {
        int next_token = model.sample_greedy(logits.data());
        if (next_token == Tokenizer::EOS_TOKEN) break;

        std::string piece = tok.decode_token(next_token);
        std::cout << piece << std::flush;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));

        model.forward_step(next_token, logits.data());
    }

    if (!raw_mode) {
        std::cout << "\n\n[DONE] Real tensor forward computation complete!\n";
    }
    return 0;
}
