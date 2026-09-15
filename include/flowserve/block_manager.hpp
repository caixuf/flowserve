#pragma once

#include "flowserve/types.hpp"
#include <unordered_map>
#include <vector>

namespace flowserve {

class BlockManager {
public:
    BlockManager(int block_size, int num_blocks, bool prefix_cache);

    int block_size() const { return block_size_; }
    int num_blocks() const { return num_blocks_; }
    int free_blocks() const { return static_cast<int>(free_.size()); }
    int used_blocks() const { return num_blocks_ - free_blocks(); }
    int idle_cached_blocks() const;
    int live_blocks() const;
    int prefix_hits() const { return prefix_hits_; }
    int prefix_misses() const { return prefix_misses_; }

    bool can_allocate(int n_blocks) const { return n_blocks <= free_blocks(); }

    // Grow the sequence's block table to cover `token_count` tokens.
    // With prefix cache, full prompt blocks may be shared.
    bool allocate_to_cover(Sequence& seq, int token_count);

    // Reserve a contiguous-style footprint of max_model_len (naive KV).
    bool reserve_max_len(Sequence& seq, int max_model_len);

    void free_seq(Sequence& seq);

    double last_block_fragmentation() const;

private:
    int32_t alloc_fresh();
    void release_physical(int32_t pid);
    void take_idle(int32_t pid);

    int block_size_;
    int num_blocks_;
    bool prefix_cache_;
    std::vector<int> refcnt_;
    std::vector<uint64_t> block_hash_;
    std::vector<int32_t> free_;
    std::vector<int32_t> idle_cache_;
    std::unordered_map<uint64_t, int32_t> cache_;
    int prefix_hits_{0};
    int prefix_misses_{0};
};

} // namespace flowserve
