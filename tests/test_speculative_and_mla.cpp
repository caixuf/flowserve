#include "flowserve/engine.hpp"
#include <cassert>
#include <iostream>
#include <vector>

using namespace flowserve;

int main() {
    ServingConfig base_cfg;
    base_cfg.block_size = 16;
    base_cfg.num_gpu_blocks = 256;
    base_cfg.max_num_seqs = 8;
    base_cfg.max_num_batched_tokens = 512;
    base_cfg.max_model_len = 256;
    base_cfg.continuous_batching = true;
    base_cfg.paged_kv = true;
    base_cfg.prefix_cache = true;

    // Build synthetic workload: 32 requests
    std::vector<WorkloadItem> items;
    for (int i = 0; i < 32; ++i) {
        WorkloadItem it;
        it.prompt.assign(32, 100 + (i % 4)); // some shared prefixes
        it.max_new_tokens = 16;
        it.arrival_tick = i * 200;
        items.push_back(it);
    }

    // 1. Base run
    Engine base_engine(base_cfg);
    RunReport base_report = base_engine.run(items);

    // 2. MLA run (Multi-Head Latent Attention: 4x compression)
    ServingConfig mla_cfg = base_cfg;
    mla_cfg.enable_mla = true;
    mla_cfg.mla_latent_ratio = 4;
    Engine mla_engine(mla_cfg);
    RunReport mla_report = mla_engine.run(items);

    std::cout << "Base peak blocks: " << base_report.peak_used_blocks << "\n";
    std::cout << "MLA  peak blocks: " << mla_report.peak_used_blocks << "\n";
    // MLA should dramatically reduce peak block footprint
    assert(mla_report.peak_used_blocks < base_report.peak_used_blocks);
    assert(mla_report.peak_used_blocks <= (base_report.peak_used_blocks + 3) / 4 + 2);

    // 3. Speculative Decoding run
    ServingConfig spec_cfg = base_cfg;
    spec_cfg.enable_speculative = true;
    spec_cfg.speculative_draft_tokens = 3;
    spec_cfg.speculative_acceptance_rate = 0.8;
    spec_cfg.draft_us_per_token = 0.4; // very fast draft step
    Engine spec_engine(spec_cfg);
    RunReport spec_report = spec_engine.run(items);

    std::cout << "Base sim_us: " << base_report.simulated_us << ", mean TPOT: " << base_report.mean_tpot_us << "\n";
    std::cout << "Spec sim_us: " << spec_report.simulated_us << ", mean TPOT: " << spec_report.mean_tpot_us << "\n";
    std::cout << "Drafted: " << spec_report.speculative_drafted_tokens
              << ", Accepted: " << spec_report.speculative_accepted_tokens << "\n";

    // Speculative should produce speedup in simulated decode time and lower TPOT
    assert(spec_report.simulated_us < base_report.simulated_us);
    assert(spec_report.mean_tpot_us < base_report.mean_tpot_us);
    assert(spec_report.speculative_accepted_tokens > 0);

    std::cout << "[PASS] test_speculative_and_mla passed successfully!\n";
    return 0;
}
