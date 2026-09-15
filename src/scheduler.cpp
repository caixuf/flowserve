#include "flowserve/scheduler.hpp"
#include <algorithm>

namespace flowserve {

Scheduler::Scheduler(ServingConfig cfg) : cfg_(cfg) {}

uint64_t Scheduler::add_request(std::vector<int> prompt, int max_new_tokens, uint64_t arrival_tick) {
    Sequence s;
    s.id = next_id_++;
    s.token_ids = std::move(prompt);
    s.prompt_len = static_cast<int>(s.token_ids.size());
    s.max_new_tokens = max_new_tokens;
    s.arrival_tick = arrival_tick;
    const uint64_t id = s.id;
    seqs_.emplace(id, std::move(s));
    waiting_.push_back(id);
    return id;
}

Sequence& Scheduler::seq(uint64_t id) { return seqs_.at(id); }
const Sequence& Scheduler::seq(uint64_t id) const { return seqs_.at(id); }

bool Scheduler::admit(uint64_t id, BlockManager& bm) {
    Sequence& s = seqs_.at(id);
    if (!cfg_.paged_kv) {
        return bm.reserve_max_len(s, cfg_.max_model_len);
    }
    return bm.allocate_to_cover(s, s.prompt_len);
}

bool Scheduler::ensure_decode_slot(Sequence& s, BlockManager& bm) {
    const int next_tokens = static_cast<int>(s.token_ids.size()) + 1;
    if (next_tokens > cfg_.max_model_len) return false;
    if (!cfg_.paged_kv) return next_tokens <= cfg_.max_model_len;
    return bm.allocate_to_cover(s, next_tokens);
}

void Scheduler::preempt_newest(BlockManager& bm) {
    if (running_.empty()) return;
    const uint64_t id = running_.back();
    running_.pop_back();
    Sequence& s = seqs_.at(id);
    bm.free_seq(s);
    s.num_computed = 0;
    s.first_token_tick = 0;
    if (static_cast<int>(s.token_ids.size()) > s.prompt_len) {
        s.token_ids.resize(static_cast<size_t>(s.prompt_len));
    }
    waiting_.push_front(id);
    preemptions_ += 1;
}

void Scheduler::finish(uint64_t id, BlockManager& bm) {
    auto it = std::find(running_.begin(), running_.end(), id);
    if (it != running_.end()) running_.erase(it);
    Sequence& s = seqs_.at(id);
    s.finished = true;
    bm.free_seq(s);
}

int Scheduler::live_kv_blocks() const {
    int n = 0;
    for (uint64_t id : running_) {
        n += static_cast<int>(seqs_.at(id).block_table.size());
    }
    return n;
}

Schedule Scheduler::schedule(BlockManager& bm) {
    Schedule out;

    if (!cfg_.continuous_batching) {
        if (running_.empty()) {
            while (!waiting_.empty() && static_cast<int>(running_.size()) < cfg_.max_num_seqs) {
                const uint64_t id = waiting_.front();
                if (!admit(id, bm)) break;
                waiting_.pop_front();
                running_.push_back(id);
                out.prefill.push_back(id);
            }
        } else {
            for (uint64_t id : running_) {
                Sequence& s = seqs_.at(id);
                if (s.num_computed < s.prompt_len) {
                    out.prefill.push_back(id);
                    continue;
                }
                if (!ensure_decode_slot(s, bm)) {
                    preempt_newest(bm);
                    break;
                }
                out.decode.push_back(id);
            }
        }
        return out;
    }

    // Continuous batching: keep decoding live sequences, admit prefills into leftover budget.
    int batched_tokens = 0;
    std::vector<uint64_t> still_running;
    still_running.reserve(running_.size());
    for (uint64_t id : running_) {
        Sequence& s = seqs_.at(id);
        if (s.num_computed < s.prompt_len) {
            const int left = s.prompt_len - s.num_computed;
            if (batched_tokens + left > cfg_.max_num_batched_tokens) {
                still_running.push_back(id);
                continue;
            }
            out.prefill.push_back(id);
            batched_tokens += left;
            still_running.push_back(id);
            continue;
        }
        if (!ensure_decode_slot(s, bm)) {
            bm.free_seq(s);
            s.num_computed = 0;
            s.first_token_tick = 0;
            if (static_cast<int>(s.token_ids.size()) > s.prompt_len) {
                s.token_ids.resize(static_cast<size_t>(s.prompt_len));
            }
            waiting_.push_front(id);
            preemptions_ += 1;
            continue;
        }
        if (batched_tokens + 1 > cfg_.max_num_batched_tokens) {
            still_running.push_back(id);
            continue;
        }
        out.decode.push_back(id);
        batched_tokens += 1;
        still_running.push_back(id);
    }
    running_.swap(still_running);

    while (!waiting_.empty() && static_cast<int>(running_.size()) < cfg_.max_num_seqs) {
        const uint64_t id = waiting_.front();
        Sequence& s = seqs_.at(id);
        const int pt = s.prompt_len;
        if (batched_tokens + pt > cfg_.max_num_batched_tokens) break;
        if (!admit(id, bm)) {
            if (running_.empty()) break;
            preempt_newest(bm);
            if (!admit(id, bm)) break;
        }
        waiting_.pop_front();
        running_.push_back(id);
        out.prefill.push_back(id);
        batched_tokens += pt;
    }
    return out;
}

} // namespace flowserve
