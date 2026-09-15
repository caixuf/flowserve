#include "flowserve/block_manager.hpp"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <iostream>
#include <vector>

using namespace flowserve;

int main() {
    BlockManager bm(16, 64, true);

    Sequence a;
    a.token_ids.assign(32, 7);
    assert(bm.allocate_to_cover(a, 32));
    assert(static_cast<int>(a.block_table.size()) == 2);
    const int after_a = bm.used_blocks();
    assert(after_a == 2);

    Sequence b;
    b.token_ids.assign(32, 7);
    assert(bm.allocate_to_cover(b, 32));
    assert(b.block_table == a.block_table);
    assert(bm.used_blocks() == after_a);
    assert(bm.prefix_hits() == 2);

    Sequence c;
    c.token_ids.assign(32, 9);
    assert(bm.allocate_to_cover(c, 32));
    assert(bm.used_blocks() == 4);
    assert(bm.prefix_misses() >= 2);

    bm.free_seq(a);
    bm.free_seq(b);
    assert(bm.used_blocks() == 4);  // 2 idle shared + 2 live unique
    bm.free_seq(c);
    assert(bm.used_blocks() == 4);  // hashed blocks stay until eviction
    assert(bm.free_blocks() == 60);

    std::cout << "PASS test_block_manager prefix_hits=" << bm.prefix_hits() << "\n";
    return 0;
}
