#include "flowserve/async_engine.hpp"
#include <cassert>
#include <iostream>
#include <vector>

using namespace flowserve;

int main() {
    ServingConfig cfg;
    cfg.block_size = 16;
    cfg.num_gpu_blocks = 64;
    cfg.max_num_seqs = 4;
    cfg.continuous_batching = true;
    cfg.paged_kv = true;

    AsyncEngine engine(cfg);

    // 提交两个并发异步请求
    auto handle1 = engine.submit({1, 2, 3}, 4);
    auto handle2 = engine.submit({4, 5, 6}, 3);

    auto stream1 = AsyncEngine::stream(handle1);
    auto stream2 = AsyncEngine::stream(handle2);

    std::vector<int> tokens1;
    StreamTokenChunk c1;
    while (stream1.next(c1)) {
        tokens1.push_back(c1.token_id);
    }
    assert(tokens1.size() == 4);

    std::vector<int> tokens2;
    StreamTokenChunk c2;
    while (stream2.next(c2)) {
        tokens2.push_back(c2.token_id);
    }
    assert(tokens2.size() == 3);

    engine.stop();
    std::cout << "[PASS] test_async_engine coroutine streaming passed! tokens1="
              << tokens1.size() << " tokens2=" << tokens2.size() << "\n";
    return 0;
}
