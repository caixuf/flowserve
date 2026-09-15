#pragma once

// ============================================================================
// flowserve - 现代 C++20 原生协程异步流式推理引擎 (Async Coroutine Engine)
// 
// 核心机制:
//   1. 异步请求提交: 客户端协程提交 prompt，立即获得 AsyncTokenStream;
//   2. 原生协程消费: 客户端使用 co_await 挂起并逐 token 唤醒流式推送 (SSE 友好);
//   3. 零阻塞轮询: 后台 Engine 事件循环每 step 生成 token 时通知对应流;
//   4. 恪守架构边界: 纯 C++20 标准协程，零外部污染，可直接对接 flowcoro 调度。
// ============================================================================

#include "flowserve/block_manager.hpp"
#include "flowserve/scheduler.hpp"
#include "flowserve/types.hpp"
#include <atomic>
#include <condition_variable>
#include <coroutine>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

namespace flowserve {

// 流式输出单个 Token 的事件包 (类似 OpenAI chunk)
struct StreamTokenChunk {
    int token_id{-1};
    std::string text_piece{};
    bool is_final{false};
    RequestMetric metric{};
};

// C++20 协程流式生成器 Promise
class AsyncTokenStream {
public:
    struct promise_type;
    using handle_type = std::coroutine_handle<promise_type>;

    struct promise_type {
        StreamTokenChunk current_chunk;
        std::exception_ptr exception{nullptr};

        AsyncTokenStream get_return_object() {
            return AsyncTokenStream{handle_type::from_promise(*this)};
        }
        std::suspend_always initial_suspend() noexcept { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }
        void return_void() noexcept {}
        void unhandled_exception() { exception = std::current_exception(); }

        std::suspend_always yield_value(StreamTokenChunk chunk) noexcept {
            current_chunk = std::move(chunk);
            return {};
        }
    };

    explicit AsyncTokenStream(handle_type h) : coro_(h) {}
    ~AsyncTokenStream() {
        if (coro_) coro_.destroy();
    }

    AsyncTokenStream(const AsyncTokenStream&) = delete;
    AsyncTokenStream& operator=(const AsyncTokenStream&) = delete;

    AsyncTokenStream(AsyncTokenStream&& other) noexcept : coro_(other.coro_) {
        other.coro_ = nullptr;
    }
    AsyncTokenStream& operator=(AsyncTokenStream&& other) noexcept {
        if (this != &other) {
            if (coro_) coro_.destroy();
            coro_ = other.coro_;
            other.coro_ = nullptr;
        }
        return *this;
    }

    bool next(StreamTokenChunk& chunk) {
        if (!coro_ || coro_.done()) return false;
        coro_.resume();
        if (coro_.done()) return false;
        chunk = coro_.promise().current_chunk;
        return true;
    }

private:
    handle_type coro_{nullptr};
};

// 异步推理会话队列与状态
struct AsyncRequestHandle {
    uint64_t seq_id{0};
    std::vector<int> prompt;
    int max_new_tokens{16};
    std::queue<StreamTokenChunk> ready_chunks;
    bool finished{false};
    std::mutex mtx;
    std::condition_variable cv;
};

class AsyncEngine {
public:
    explicit AsyncEngine(ServingConfig cfg)
        : cfg_(cfg),
          sched_(cfg),
          bm_(cfg.effective_block_size(), cfg.num_gpu_blocks, cfg.prefix_cache && cfg.paged_kv),
          running_(true) {
        worker_thread_ = std::thread([this]() { loop(); });
    }

    ~AsyncEngine() {
        stop();
    }

    std::shared_ptr<AsyncRequestHandle> submit(std::vector<int> prompt, int max_new_tokens) {
        auto handle = std::make_shared<AsyncRequestHandle>();
        handle->prompt = prompt;
        handle->max_new_tokens = max_new_tokens;

        {
            std::lock_guard<std::mutex> lock(queue_mtx_);
            pending_queue_.push_back(handle);
        }
        queue_cv_.notify_one();
        return handle;
    }

    // 将 Session Handle 转换为 C++20 协程流
    static AsyncTokenStream stream(std::shared_ptr<AsyncRequestHandle> handle) {
        while (true) {
            StreamTokenChunk chunk;
            {
                std::unique_lock<std::mutex> lock(handle->mtx);
                handle->cv.wait(lock, [&]() {
                    return !handle->ready_chunks.empty() || handle->finished;
                });

                if (!handle->ready_chunks.empty()) {
                    chunk = handle->ready_chunks.front();
                    handle->ready_chunks.pop();
                } else if (handle->finished) {
                    break;
                }
            }
            co_yield chunk;
            if (chunk.is_final) break;
        }
    }

    void stop() {
        if (running_.exchange(false)) {
            queue_cv_.notify_all();
            if (worker_thread_.joinable()) {
                worker_thread_.join();
            }
        }
    }

private:
    void loop() {
        while (running_) {
            // 1. 搬运待处理请求进 Scheduler
            std::vector<std::shared_ptr<AsyncRequestHandle>> to_admit;
            {
                std::unique_lock<std::mutex> lock(queue_mtx_);
                if (pending_queue_.empty() && sched_.idle()) {
                    queue_cv_.wait_for(lock, std::chrono::milliseconds(10), [&]() {
                        return !pending_queue_.empty() || !running_;
                    });
                }
                while (!pending_queue_.empty()) {
                    to_admit.push_back(pending_queue_.front());
                    pending_queue_.pop_front();
                }
            }

            if (!running_ && to_admit.empty() && sched_.idle()) {
                break;
            }

            for (auto& req : to_admit) {
                uint64_t id = sched_.add_request(req->prompt, req->max_new_tokens, 0);
                req->seq_id = id;
                active_sessions_[id] = req;
            }

            if (sched_.idle()) continue;

            // 2. 执行一步调度
            const Schedule sch = sched_.schedule(bm_);

            for (uint64_t id : sch.prefill) {
                Sequence& s = sched_.seq(id);
                s.num_computed = s.prompt_len;
            }

            for (uint64_t id : sch.decode) {
                Sequence& s = sched_.seq(id);
                // 模拟一个新 token
                int next_tok = static_cast<int>((s.id * 6364136223846793005ULL + s.token_ids.size()) % 32000);
                s.token_ids.push_back(next_tok);
                s.num_computed = static_cast<int>(s.token_ids.size());
                const int out = s.num_computed - s.prompt_len;
                const bool is_done = (out >= s.max_new_tokens);

                auto it = active_sessions_.find(id);
                if (it != active_sessions_.end()) {
                    auto& session = it->second;
                    StreamTokenChunk chunk;
                    chunk.token_id = next_tok;
                    chunk.text_piece = "tok_" + std::to_string(next_tok) + " ";
                    chunk.is_final = is_done;

                    {
                        std::lock_guard<std::mutex> lock(session->mtx);
                        session->ready_chunks.push(chunk);
                        if (is_done) session->finished = true;
                    }
                    session->cv.notify_one();

                    if (is_done) {
                        sched_.finish(id, bm_);
                        active_sessions_.erase(it);
                    }
                }
            }

            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
    }

    ServingConfig cfg_;
    Scheduler sched_;
    BlockManager bm_;
    std::atomic<bool> running_{false};
    std::thread worker_thread_;

    std::mutex queue_mtx_;
    std::condition_variable queue_cv_;
    std::deque<std::shared_ptr<AsyncRequestHandle>> pending_queue_;
    std::unordered_map<uint64_t, std::shared_ptr<AsyncRequestHandle>> active_sessions_;
};

} // namespace flowserve
