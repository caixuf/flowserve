#pragma once

#include <cstdint>
#include <deque>
#include <vector>

namespace flowserve {

struct ServingConfig {
    int block_size{16};
    int num_gpu_blocks{256};
    int max_num_seqs{8};
    int max_num_batched_tokens{512};
    int max_model_len{256};
    bool continuous_batching{true};
    bool paged_kv{true};
    bool prefix_cache{true};
    // Fake GPU cost model (microseconds of simulated device time).
    double prefill_us_per_token{2.0};
    double decode_us_per_seq{40.0};
};

struct Sequence {
    uint64_t id{0};
    std::vector<int> token_ids;
    int prompt_len{0};
    int max_new_tokens{16};
    int num_computed{0};
    uint64_t arrival_tick{0};
    uint64_t first_token_tick{0};
    uint64_t finish_tick{0};
    std::vector<int32_t> block_table;
    bool finished{false};
};

inline int tokens_to_blocks(int tokens, int block_size) {
    if (tokens <= 0) return 0;
    return (tokens + block_size - 1) / block_size;
}

struct Schedule {
    std::vector<uint64_t> prefill;
    std::vector<uint64_t> decode;
};

struct RequestMetric {
    uint64_t id{0};
    double ttft_us{0};
    double tpot_us{0};
    int output_tokens{0};
    int prompt_tokens{0};
};

struct RunReport {
    double simulated_us{0};
    int peak_used_blocks{0};
    int prefix_block_hits{0};
    int prefix_block_misses{0};
    int preemptions{0};
    int finished{0};
    double mean_ttft_us{0};
    double mean_tpot_us{0};
    double p95_ttft_us{0};
    std::vector<RequestMetric> per_request;
};

} // namespace flowserve
