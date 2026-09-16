# 怎么学 FlowServe

目标：看懂 **KV 为什么按块分配、decode 中途为什么能插 prefill、前缀哈希为什么能省块**。  
不是：学会训大模型，也不是把网页对话当成自研模型能力。

网页默认挂的是开源 **Qwen2.5-0.5B-Instruct**（HuggingFace）。本仓 C++ 证的是调度；小模型 `tinymla_*.bin` 只负责 FLSV mmap 和 MLA 前向能跑通。

## 0. 先跑通，再读代码

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
ctest --test-dir build --output-on-failure
./build/bench_serving
./build/chat_cli --weights tinymla_chat.bin --prompt "User: 什么是 MLA？\nAssistant:"
```

`bench_serving` 会打出多条对照臂。先盯三列：**peak KV blocks、prefix hits、mean TTFT**。数字从哪来，下一节对着开关读。

## 1. 开关就是实验设计

读 `include/flowserve/types.hpp` 的 `ServingConfig`：

| 开关 | 关掉之后你该看到什么 |
|---|---|
| `paged_kv` | 一来就按 `max_model_len` 占满块 → peak 变大（本机 naive 约 48 vs paged 24） |
| `continuous_batching` | 整批 decode 完才接下一批 → 新请求 TTFT 变差 |
| `prefix_cache` | 相同前缀不共享 → `prefix hits` 掉到 0 |
| `enable_mla` | `effective_block_size()` 变大，同样 token 更少物理块 |
| `enable_speculative` | 调度按草稿多 token 前进，TTFT/TPOT 用**解析代价模型**变小，不是真 GPU 算子更快 |

`prefill_us_per_token` / `decode_us_per_seq` 是假 GPU 时钟。消融比的是调度几何，不是 kernel 吞吐。

## 2. BlockManager：物理块、引用计数、前缀

读 `include/flowserve/block_manager.hpp` + `src/block_manager.cpp`。

要抓住的不变量：

1. 序列不拥有「一条长 KV」，只拥有 **block table**（逻辑页 → 物理块 id）。
2. `allocate_to_cover(seq, token_count)` 只为**已经有的 token** 补块，不是按 `max_model_len` 一次订完。
3. `prefix_cache` 打开时：满块按内容哈希进 `cache_`；`refcnt` 共享；`refcnt==0` 进 idle，不是立刻丢。
4. `reserve_max_len` 是对照臂：模拟「一上来订满上下文」的 naive KV。

练习：改 `tests/test_block_manager.cpp` 里的 prompt 使两个序列前缀相同，断言 `prefix_hits` 增加。不要先改生产路径。

## 3. Scheduler：谁在跑、何时能插队

读 `include/flowserve/scheduler.hpp` + `src/scheduler.cpp`。

连续批的要点：decode 步与新 prefill **不是两个世界**。有空槽、有块，就可以在当前 batch 里插入。投机解码时 `ensure_decode_slot` 还要为草稿多占几步，这是调度约束，不是模型更聪明。

练习：跑 `test_scheduler` / `test_engine_ablation`，对照 `continuous_batching=false` 时新请求是否必须等整批结束。

## 4. Engine：一步时钟

读 `src/engine.cpp`。每一步：scheduler 组 batch → 按假代价模型推进时间 → 写回 token 与 KV 占用。  
真正的 TinyMLA 矩阵在 `include/flowserve/model.hpp`，走 `chat_cli` 时才会每步 `forward_step`。两条路径不要混：`bench_serving` 多数时候不跑完整 Transformer。

## 5. TinyMLA 与 FLSV

- 训练导出：`../flowtrain/train_tinymla.py` / `train_chatbot.py` → `tinymla_story.bin` / `tinymla_chat.bin`
- 加载：`include/flowserve/weight_loader.hpp`（magic `FLSV`）
- 结构：字节词表 259、2 层、hidden 64。会背训练集，OOD 会糊。这是格式闭环，不是对话能力。

## 6. Web 双后端

`tools/web_server.py`：

```bash
python3 tools/web_server.py --backend qwen      # 默认：Qwen Instruct
python3 tools/web_server.py --backend tinymla   # C++ chat_cli + FLSV
```

对话质量问题先看 `--backend`，再怪调度器。Qwen 侧说明见 `../qwen_chatbot/README.md`。

## 建议阅读顺序

`types.hpp` → `block_manager` → `scheduler` → `engine.cpp` → `bench_serving.cpp` → `model.hpp` / `weight_loader.hpp` → `web_server.py`
