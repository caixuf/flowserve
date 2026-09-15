#include "flowserve/engine.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace flowserve {

Engine::Engine(ServingConfig cfg) : cfg_(std::move(cfg)) {}

static int fake_next_token(const Sequence& s) {
    uint64_t h = s.id * 6364136223846793005ULL + static_cast<uint64_t>(s.token_ids.size());
    return static_cast<int>(h % 32000);
}

static double percentile(std::vector<double> xs, double p) {
    if (xs.empty()) return 0;
    std::sort(xs.begin(), xs.end());
    const double idx = p * static_cast<double>(xs.size() - 1);
    const size_t lo = static_cast<size_t>(idx);
    const size_t hi = std::min(lo + 1, xs.size() - 1);
    const double frac = idx - static_cast<double>(lo);
    return xs[lo] * (1.0 - frac) + xs[hi] * frac;
}

RunReport Engine::run(const std::vector<WorkloadItem>& items) {
    Scheduler sched(cfg_);
    BlockManager bm(cfg_.effective_block_size(), cfg_.num_gpu_blocks, cfg_.prefix_cache && cfg_.paged_kv);

    struct Pending {
        WorkloadItem item;
        bool admitted{false};
    };
    std::vector<Pending> pending;
    pending.reserve(items.size());
    for (const auto& it : items) pending.push_back(Pending{it, false});

    RunReport report;
    double now = 0;
    int steps = 0;
    const int kMaxSteps = 1'000'000;

    auto admit_ready = [&]() {
        for (auto& p : pending) {
            if (p.admitted) continue;
            if (static_cast<double>(p.item.arrival_tick) > now) continue;
            sched.add_request(p.item.prompt, p.item.max_new_tokens, p.item.arrival_tick);
            p.admitted = true;
        }
    };

    auto all_admitted = [&]() {
        for (const auto& p : pending) if (!p.admitted) return false;
        return true;
    };

    while (steps++ < kMaxSteps) {
        admit_ready();
        if (sched.idle() && all_admitted()) break;

        const Schedule sch = sched.schedule(bm);
        report.peak_used_blocks = std::max(report.peak_used_blocks, sched.live_kv_blocks());

        int prefill_tokens = 0;
        for (uint64_t id : sch.prefill) {
            const Sequence& s = sched.seq(id);
            prefill_tokens += (s.prompt_len - s.num_computed);
        }
        const int decode_n = static_cast<int>(sch.decode.size());
        double step_us = 0;
        if (prefill_tokens > 0) step_us += cfg_.prefill_us_per_token * prefill_tokens;
        if (decode_n > 0) {
            step_us += cfg_.decode_us_per_seq * decode_n;
            if (cfg_.enable_speculative) {
                step_us += cfg_.draft_us_per_token * cfg_.speculative_draft_tokens * decode_n;
            }
        }
        if (step_us <= 0) {
            // Nothing scheduled: jump to next arrival.
            uint64_t next_arr = std::numeric_limits<uint64_t>::max();
            for (const auto& p : pending) {
                if (!p.admitted) next_arr = std::min(next_arr, p.item.arrival_tick);
            }
            if (next_arr == std::numeric_limits<uint64_t>::max() || static_cast<double>(next_arr) <= now) {
                now += 1;
            } else {
                now = static_cast<double>(next_arr);
            }
            continue;
        }
        now += step_us;

        for (uint64_t id : sch.prefill) {
            Sequence& s = sched.seq(id);
            s.num_computed = s.prompt_len;
        }
        for (uint64_t id : sch.decode) {
            Sequence& s = sched.seq(id);
            int tokens_to_gen = 1;
            if (cfg_.enable_speculative) {
                const int k = cfg_.speculative_draft_tokens;
                const int accepted = std::clamp(static_cast<int>(std::round(k * cfg_.speculative_acceptance_rate)), 0, k);
                tokens_to_gen = 1 + accepted;
                report.speculative_drafted_tokens += k;
                report.speculative_accepted_tokens += accepted;
            }

            for (int t = 0; t < tokens_to_gen; ++t) {
                s.token_ids.push_back(fake_next_token(s));
                s.num_computed = static_cast<int>(s.token_ids.size());
                if (s.first_token_tick == 0) s.first_token_tick = static_cast<uint64_t>(now);
                const int out = s.num_computed - s.prompt_len;
                if (out >= s.max_new_tokens) {
                    break;
                }
            }

            const int out = s.num_computed - s.prompt_len;
            if (out >= s.max_new_tokens) {
                s.finish_tick = static_cast<uint64_t>(now);
                RequestMetric m;
                m.id = id;
                m.prompt_tokens = s.prompt_len;
                m.output_tokens = out;
                m.ttft_us = static_cast<double>(s.first_token_tick) - static_cast<double>(s.arrival_tick);
                if (out > 1) {
                    m.tpot_us = (static_cast<double>(s.finish_tick) - static_cast<double>(s.first_token_tick))
                                / static_cast<double>(out - 1);
                }
                report.per_request.push_back(m);
                sched.finish(id, bm);
            }
        }
    }

    report.simulated_us = now;
    report.prefix_block_hits = bm.prefix_hits();
    report.prefix_block_misses = bm.prefix_misses();
    report.preemptions = sched.preemptions();
    report.finished = static_cast<int>(report.per_request.size());

    std::vector<double> ttfts, tpots;
    ttfts.reserve(report.per_request.size());
    double sum_ttft = 0, sum_tpot = 0;
    int tpot_n = 0;
    for (const auto& m : report.per_request) {
        ttfts.push_back(m.ttft_us);
        sum_ttft += m.ttft_us;
        if (m.output_tokens > 1) {
            tpots.push_back(m.tpot_us);
            sum_tpot += m.tpot_us;
            tpot_n += 1;
        }
    }
    if (!report.per_request.empty()) {
        report.mean_ttft_us = sum_ttft / static_cast<double>(report.per_request.size());
        report.p95_ttft_us = percentile(std::move(ttfts), 0.95);
    }
    if (tpot_n > 0) report.mean_tpot_us = sum_tpot / static_cast<double>(tpot_n);
    return report;
}

} // namespace flowserve
