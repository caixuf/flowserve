#include "flowserve/engine.hpp"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <iostream>
#include <vector>

using namespace flowserve;

static std::vector<WorkloadItem> make_load() {
    std::vector<WorkloadItem> items;
    items.reserve(24);
    for (int i = 0; i < 24; ++i) {
        WorkloadItem w;
        w.prompt.assign(32, (i % 4 == 0) ? 3 : (3 + i));
        if (i % 4 == 0) w.prompt.assign(32, 3);  // shared prefix group
        w.max_new_tokens = 8;
        w.arrival_tick = static_cast<uint64_t>(i * 80);
        items.push_back(std::move(w));
    }
    return items;
}

int main() {
    const auto load = make_load();

    ServingConfig full;
    full.block_size = 16;
    full.num_gpu_blocks = 128;
    full.max_num_seqs = 4;
    full.max_num_batched_tokens = 256;
    full.max_model_len = 64;
    full.continuous_batching = true;
    full.paged_kv = true;
    full.prefix_cache = true;

    ServingConfig naive = full;
    naive.paged_kv = false;
    naive.prefix_cache = false;

    ServingConfig static_batch = full;
    static_batch.continuous_batching = false;

    const RunReport r_full = Engine(full).run(load);
    const RunReport r_naive = Engine(naive).run(load);
    const RunReport r_static = Engine(static_batch).run(load);

    std::cout << "full_peak=" << r_full.peak_used_blocks
              << " naive_peak=" << r_naive.peak_used_blocks
              << " full_ttft=" << r_full.mean_ttft_us
              << " static_ttft=" << r_static.mean_ttft_us
              << " hits=" << r_full.prefix_block_hits << std::endl;
    assert(r_full.finished == 24);
    assert(r_naive.finished == 24);
    assert(r_static.finished == 24);
    assert(r_full.peak_used_blocks < r_naive.peak_used_blocks);
    assert(r_full.mean_ttft_us < r_static.mean_ttft_us);
    assert(r_full.prefix_block_hits > 0);

    std::cout << "PASS ablation"
              << " full_peak=" << r_full.peak_used_blocks
              << " naive_peak=" << r_naive.peak_used_blocks
              << " full_ttft=" << r_full.mean_ttft_us
              << " static_ttft=" << r_static.mean_ttft_us
              << " prefix_hits=" << r_full.prefix_block_hits
              << "\n";
    return 0;
}
