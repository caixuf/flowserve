#include "flowserve/block_manager.hpp"
#include <algorithm>

namespace flowserve {

static uint64_t mix64(uint64_t h, int tok) {
    h ^= static_cast<uint64_t>(static_cast<uint32_t>(tok)) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    return h;
}

static uint64_t hash_span(uint64_t parent, const int* t, int n) {
    uint64_t h = parent ^ 0xA5A5A5A5A5A5A5A5ULL;
    for (int i = 0; i < n; ++i) h = mix64(h, t[i]);
    return h == 0 ? 1 : h;
}

BlockManager::BlockManager(int block_size, int num_blocks, bool prefix_cache)
    : block_size_(block_size), num_blocks_(num_blocks), prefix_cache_(prefix_cache),
      refcnt_(static_cast<size_t>(num_blocks), 0),
      block_hash_(static_cast<size_t>(num_blocks), 0) {
    free_.reserve(static_cast<size_t>(num_blocks));
    for (int i = num_blocks - 1; i >= 0; --i) free_.push_back(i);
}

int32_t BlockManager::alloc_fresh() {
    if (!free_.empty()) {
        const int32_t pid = free_.back();
        free_.pop_back();
        refcnt_[static_cast<size_t>(pid)] = 1;
        block_hash_[static_cast<size_t>(pid)] = 0;
        return pid;
    }
    if (!idle_cache_.empty()) {
        const int32_t pid = idle_cache_.back();
        idle_cache_.pop_back();
        const uint64_t h = block_hash_[static_cast<size_t>(pid)];
        if (h != 0) {
            auto it = cache_.find(h);
            if (it != cache_.end() && it->second == pid) cache_.erase(it);
        }
        refcnt_[static_cast<size_t>(pid)] = 1;
        block_hash_[static_cast<size_t>(pid)] = 0;
        return pid;
    }
    return -1;
}

void BlockManager::take_idle(int32_t pid) {
    auto it = std::find(idle_cache_.begin(), idle_cache_.end(), pid);
    if (it != idle_cache_.end()) idle_cache_.erase(it);
}

void BlockManager::release_physical(int32_t pid) {
    if (pid < 0) return;
    auto& rc = refcnt_[static_cast<size_t>(pid)];
    if (rc <= 0) return;
    rc -= 1;
    if (rc > 0) return;
    const uint64_t h = block_hash_[static_cast<size_t>(pid)];
    if (h != 0) {
        idle_cache_.push_back(pid);
        return;
    }
    free_.push_back(pid);
}

bool BlockManager::allocate_to_cover(Sequence& seq, int token_count) {
    const int need = tokens_to_blocks(token_count, block_size_);
    while (static_cast<int>(seq.block_table.size()) < need) {
        const int logical = static_cast<int>(seq.block_table.size());
        const int lo = logical * block_size_;
        const int hi = std::min(lo + block_size_, token_count);
        const bool full = (hi - lo) == block_size_;
        uint64_t parent = 0;
        if (logical > 0) {
            const int32_t prev = seq.block_table[static_cast<size_t>(logical - 1)];
            parent = block_hash_[static_cast<size_t>(prev)];
        }
        const uint64_t h = (full && hi <= static_cast<int>(seq.token_ids.size()))
            ? hash_span(parent, seq.token_ids.data() + lo, hi - lo)
            : 0;

        if (prefix_cache_ && full && h != 0) {
            auto it = cache_.find(h);
            if (it != cache_.end()) {
                const int32_t pid = it->second;
                if (pid >= 0 && refcnt_[static_cast<size_t>(pid)] >= 0) {
                    take_idle(pid);
                    refcnt_[static_cast<size_t>(pid)] += 1;
                    seq.block_table.push_back(pid);
                    prefix_hits_ += 1;
                    continue;
                }
            }
        }

        const int32_t pid = alloc_fresh();
        if (pid < 0) return false;
        if (prefix_cache_ && full && h != 0) {
            block_hash_[static_cast<size_t>(pid)] = h;
            cache_[h] = pid;
        }
        seq.block_table.push_back(pid);
        if (prefix_cache_ && full) prefix_misses_ += 1;
    }
    return true;
}

bool BlockManager::reserve_max_len(Sequence& seq, int max_model_len) {
    const int need = tokens_to_blocks(max_model_len, block_size_);
    while (static_cast<int>(seq.block_table.size()) < need) {
        const int32_t pid = alloc_fresh();
        if (pid < 0) return false;
        seq.block_table.push_back(pid);
    }
    return true;
}

void BlockManager::free_seq(Sequence& seq) {
    for (int32_t pid : seq.block_table) release_physical(pid);
    seq.block_table.clear();
}

int BlockManager::idle_cached_blocks() const {
    return static_cast<int>(idle_cache_.size());
}

int BlockManager::live_blocks() const {
    return used_blocks() - idle_cached_blocks();
}

double BlockManager::last_block_fragmentation() const {
    return 0.0;
}

} // namespace flowserve
