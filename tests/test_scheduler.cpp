#include "flowserve/scheduler.hpp"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <iostream>
#include <vector>

using namespace flowserve;

int main() {
    ServingConfig cfg;
    cfg.block_size = 16;
    cfg.num_gpu_blocks = 64;
    cfg.max_num_seqs = 2;
    cfg.max_num_batched_tokens = 512;
    cfg.max_model_len = 128;
    cfg.continuous_batching = true;
    cfg.paged_kv = true;
    cfg.prefix_cache = false;

    Scheduler sched(cfg);
    BlockManager bm(cfg.block_size, cfg.num_gpu_blocks, false);

    const std::vector<int> prompt(16, 1);
    sched.add_request(prompt, 4, 0);
    sched.add_request(prompt, 4, 0);

    const Schedule s0 = sched.schedule(bm);
    assert(s0.prefill.size() == 2);
    assert(s0.decode.empty());
    for (uint64_t id : s0.prefill) sched.seq(id).num_computed = sched.seq(id).prompt_len;

    const Schedule s1 = sched.schedule(bm);
    assert(s1.prefill.empty());
    assert(s1.decode.size() == 2);

    // Third request waits until a slot frees, but can still sit in waiting while others decode.
    sched.add_request(prompt, 4, 10);
    const Schedule s2 = sched.schedule(bm);
    assert(s2.decode.size() == 2);
    assert(s2.prefill.empty());
    assert(sched.waiting_size() == 1);

    std::cout << "PASS test_scheduler running=" << sched.running_size()
              << " waiting=" << sched.waiting_size() << "\n";
    return 0;
}
