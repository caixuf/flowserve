#pragma once

#include "flowserve/block_manager.hpp"
#include "flowserve/types.hpp"
#include <cstdint>
#include <deque>
#include <unordered_map>
#include <vector>

namespace flowserve {

class Scheduler {
public:
    explicit Scheduler(ServingConfig cfg);

    uint64_t add_request(std::vector<int> prompt, int max_new_tokens, uint64_t arrival_tick);

    bool idle() const { return waiting_.empty() && running_.empty(); }
    int waiting_size() const { return static_cast<int>(waiting_.size()); }
    int running_size() const { return static_cast<int>(running_.size()); }
    int preemptions() const { return preemptions_; }
    int live_kv_blocks() const;

    Sequence& seq(uint64_t id);
    const Sequence& seq(uint64_t id) const;

    Schedule schedule(BlockManager& bm);
    void finish(uint64_t id, BlockManager& bm);

    const ServingConfig& config() const { return cfg_; }

private:
    bool admit(uint64_t id, BlockManager& bm);
    bool ensure_decode_slot(Sequence& s, BlockManager& bm);
    void preempt_newest(BlockManager& bm);

    ServingConfig cfg_;
    uint64_t next_id_{1};
    std::deque<uint64_t> waiting_;
    std::vector<uint64_t> running_;
    std::unordered_map<uint64_t, Sequence> seqs_;
    int preemptions_{0};
};

} // namespace flowserve
