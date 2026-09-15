#include "flowserve/engine.hpp"
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace flowserve;

static std::vector<WorkloadItem> make_load(int n, int prompt, int gen, uint64_t gap) {
    std::vector<WorkloadItem> items;
    items.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        WorkloadItem w;
        const int fill = (i % 5 == 0) ? 11 : (11 + (i % 17));
        w.prompt.assign(static_cast<size_t>(prompt), fill);
        if (i % 5 == 0) w.prompt.assign(static_cast<size_t>(prompt), 11);
        w.max_new_tokens = gen;
        w.arrival_tick = static_cast<uint64_t>(i) * gap;
        items.push_back(std::move(w));
    }
    return items;
}

static void print_row(const char* name, const RunReport& r) {
    std::cout << std::left << std::setw(28) << name
              << " finished=" << std::setw(4) << r.finished
              << " sim_us=" << std::setw(10) << static_cast<int>(r.simulated_us)
              << " peak_kv=" << std::setw(4) << r.peak_used_blocks
              << " ttft=" << std::setw(8) << static_cast<int>(r.mean_ttft_us)
              << " p95=" << std::setw(8) << static_cast<int>(r.p95_ttft_us)
              << " tpot=" << std::setw(6) << static_cast<int>(r.mean_tpot_us)
              << " hits=" << r.prefix_block_hits
              << " preempt=" << r.preemptions
              << "\n";
}

int main() {
    ServingConfig base;
    base.block_size = 16;
    base.num_gpu_blocks = 256;
    base.max_num_seqs = 8;
    base.max_num_batched_tokens = 512;
    base.max_model_len = 96;
    base.prefill_us_per_token = 2.0;
    base.decode_us_per_seq = 40.0;

    const auto load = make_load(64, 32, 16, 120);

    struct Arm {
        const char* name;
        ServingConfig cfg;
    };
    std::vector<Arm> arms;
    {
        ServingConfig c = base;
        c.continuous_batching = true;
        c.paged_kv = true;
        c.prefix_cache = true;
        arms.push_back({"full (paged+cont+prefix)", c});
    }
    {
        ServingConfig c = base;
        c.continuous_batching = true;
        c.paged_kv = true;
        c.prefix_cache = false;
        arms.push_back({"no_prefix", c});
    }
    {
        ServingConfig c = base;
        c.continuous_batching = false;
        c.paged_kv = true;
        c.prefix_cache = true;
        arms.push_back({"no_continuous_batch", c});
    }
    {
        ServingConfig c = base;
        c.continuous_batching = true;
        c.paged_kv = false;
        c.prefix_cache = false;
        arms.push_back({"no_paged (reserve max_len)", c});
    }
    {
        ServingConfig c = base;
        c.continuous_batching = true;
        c.paged_kv = true;
        c.prefix_cache = true;
        c.enable_mla = true;
        c.mla_latent_ratio = 4;
        arms.push_back({"+ mla_latent_kv (4x)", c});
    }
    {
        ServingConfig c = base;
        c.continuous_batching = true;
        c.paged_kv = true;
        c.prefix_cache = true;
        c.enable_speculative = true;
        c.speculative_draft_tokens = 3;
        c.speculative_acceptance_rate = 0.75;
        arms.push_back({"+ speculative_decoding", c});
    }
    {
        ServingConfig c = base;
        c.continuous_batching = true;
        c.paged_kv = true;
        c.prefix_cache = true;
        c.enable_mla = true;
        c.mla_latent_ratio = 4;
        c.enable_speculative = true;
        c.speculative_draft_tokens = 3;
        c.speculative_acceptance_rate = 0.75;
        arms.push_back({"+ modern_stack (mla+spec)", c});
    }

    std::cout << "flowserve bench n=64 prompt=32 gen=16 max_seqs=8 gap=120us\n";
    for (const auto& arm : arms) {
        print_row(arm.name, Engine(arm.cfg).run(load));
    }
    std::cout << "fake tokens + cost model; not a real LLM server.\n";
    return 0;
}
