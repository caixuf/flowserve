#include "flowserve/model.hpp"
#include "flowserve/tokenizer.hpp"
#include <iostream>
#include <string>
#include <thread>
#include <chrono>
#include <vector>

using namespace flowserve;

static TransformerConfig tiny_cfg() {
    TransformerConfig cfg;
    cfg.vocab_size = 259;
    cfg.hidden_dim = 64;
    cfg.num_layers = 2;
    cfg.num_heads = 4;
    cfg.head_dim = 16;
    cfg.mla_latent_dim = 16;
    cfg.intermediate_dim = 128;
    cfg.max_seq_len = 128;
    return cfg;
}

static void generate_once(TransformerModel& model, Tokenizer& tok, const std::string& prompt,
                          int max_gen, bool raw) {
    model.reset_state();
    auto prompt_tokens = tok.encode(prompt, true);
    if (!raw) {
        std::cout << "Prompt: \"" << prompt << "\"\n";
        std::cout << "Encoded tokens: [ ";
        for (int t : prompt_tokens) std::cout << t << " ";
        std::cout << "] (count=" << prompt_tokens.size() << ")\n\n";
        std::cout << "Model generation (Real MLA Forward Steps):\n> " << std::flush;
    }

    std::vector<float> logits(model.config.vocab_size);
    for (int t : prompt_tokens) {
        model.forward_step(t, logits.data());
    }
    for (int step = 0; step < max_gen; ++step) {
        int next_token = model.sample_greedy(logits.data());
        if (next_token == Tokenizer::EOS_TOKEN) break;
        std::cout << tok.decode_token(next_token) << std::flush;
        if (!raw) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        model.forward_step(next_token, logits.data());
    }
}

static std::string unescape_prompt(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '\\' && i + 1 < s.size() && s[i + 1] == 'n') {
            out.push_back('\n');
            ++i;
        } else if (s[i] == '\\' && i + 1 < s.size() && s[i + 1] == 't') {
            out.push_back('\t');
            ++i;
        } else {
            out.push_back(s[i]);
        }
    }
    return out;
}

static int run_serve(TransformerModel& model, Tokenizer& tok) {
    std::ios::sync_with_stdio(false);
    std::string line;
    while (std::getline(std::cin, line)) {
        int max_gen = 48;
        std::string prompt = line;
        const auto tab = line.find('\t');
        if (tab != std::string::npos) {
            max_gen = std::stoi(line.substr(0, tab));
            prompt = line.substr(tab + 1);
        }
        prompt = unescape_prompt(prompt);
        generate_once(model, tok, prompt, max_gen, /*raw=*/true);
        std::cout.put('\0') << std::flush;
    }
    return 0;
}

int main(int argc, char** argv) {
    bool raw_mode = false;
    bool serve_mode = false;
    std::string prompt = "Hello, flowserve!";
    std::string weights_path;
    int max_gen = 48;

    std::vector<std::string> positional;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--raw") {
            raw_mode = true;
        } else if (arg == "--serve") {
            serve_mode = true;
        } else if (arg == "--weights" && i + 1 < argc) {
            weights_path = argv[++i];
        } else if (arg == "--prompt" && i + 1 < argc) {
            prompt = argv[++i];
        } else if ((arg == "--max-tokens" || arg == "--max-gen") && i + 1 < argc) {
            max_gen = std::stoi(argv[++i]);
        } else if (arg.rfind("--", 0) != 0) {
            positional.push_back(std::move(arg));
        }
    }

    if (weights_path.empty() && !positional.empty() && serve_mode) {
        weights_path = positional[0];
    } else if (!serve_mode) {
        if (!positional.empty() && prompt == "Hello, flowserve!") {
            prompt = positional[0];
            if (positional.size() > 1 && weights_path.empty()) weights_path = positional[1];
            if (positional.size() > 2) max_gen = std::stoi(positional[2]);
        }
    }

    if (!raw_mode && !serve_mode) {
        std::cout << "========================================================\n";
        std::cout << "  flowserve: Real Multi-Head Latent Attention (MLA) LLM \n";
        std::cout << "========================================================\n";
    }

    TransformerModel model(tiny_cfg());
    Tokenizer tok;
    if (!weights_path.empty()) {
        if (!raw_mode && !serve_mode) {
            std::cout << "Loading model weights from: " << weights_path << " (via mmap zero-copy)...\n";
        }
        WeightLoader loader = WeightLoader::open_mmap(weights_path);
        model.load_from_weights(loader);
        if (!raw_mode && !serve_mode) {
            std::cout << "Weights loaded successfully! Layers=" << model.config.num_layers
                      << ", HiddenDim=" << model.config.hidden_dim << "\n";
        }
    }

    if (serve_mode) {
        return run_serve(model, tok);
    }

    prompt = unescape_prompt(prompt);
    generate_once(model, tok, prompt, max_gen, raw_mode);
    if (!raw_mode) {
        std::cout << "\n\n[DONE] Real tensor forward computation complete!\n";
    }
    return 0;
}
