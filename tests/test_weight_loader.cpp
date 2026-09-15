#include "flowserve/weight_loader.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace flowserve;

int main() {
    ModelFileHeader hdr{};
    hdr.num_layers = 2;
    hdr.hidden_dim = 16;
    hdr.num_heads = 2;
    hdr.head_dim = 4;
    hdr.mla_latent_dim = 4;
    hdr.intermediate_dim = 32;
    hdr.vocab_size = 64;

    const size_t H = hdr.hidden_dim;
    const size_t L = hdr.mla_latent_dim;
    const size_t M = hdr.intermediate_dim;
    const size_t V = hdr.vocab_size;
    const size_t total_head_dim = hdr.num_heads * hdr.head_dim;

    const size_t per_layer_params = 2 * H + 2 * (L * H) + 3 * (total_head_dim * L) + (H * total_head_dim) + 2 * (M * H) + (H * M);
    const size_t total_params = (V * H) + (per_layer_params * hdr.num_layers) + H + (V * H);

    std::vector<float> dummy_params(total_params);
    for (size_t i = 0; i < total_params; ++i) {
        dummy_params[i] = static_cast<float>(i + 1) * 0.01f;
    }

    const std::string tmp_file = "/tmp/test_flowserve_model.bin";
    WeightLoader::dump_binary(tmp_file, hdr, dummy_params);

    WeightLoader loader = WeightLoader::open_mmap(tmp_file);
    assert(loader.header().num_layers == 2);
    assert(loader.header().hidden_dim == 16);
    assert(loader.layers().size() == 2);

    // Verify token embeddings
    assert(std::fabs(loader.token_embeddings()[0] - 0.01f) < 1e-6f);

    // Verify first layer Norm1
    size_t norm1_offset = V * H;
    assert(std::fabs(loader.layer(0).norm1[0] - dummy_params[norm1_offset]) < 1e-6f);

    // Verify move semantics
    WeightLoader moved_loader = std::move(loader);
    assert(moved_loader.layers().size() == 2);
    assert(std::fabs(moved_loader.layer(0).norm1[0] - dummy_params[norm1_offset]) < 1e-6f);

    ::unlink(tmp_file.c_str());
    std::cout << "[PASS] test_weight_loader passed.\n";
    return 0;
}
