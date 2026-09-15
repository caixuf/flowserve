# flowserve

Paged KV + continuous batching + prefix cache，挂在 [flowcoro](https://github.com/caixuf/flowcoro) 旁边的**独立推理调度仓**。

不改 flowcoro。假 token、解析 GPU 代价模型，不是生产 LLM server，也不是 vLLM 的完整复刻。

## 证什么

| 开关 | 对照 |
|---|---|
| `paged_kv` | 按实际 token 占 block，而不是一来就 `max_model_len` |
| `continuous_batching` | decode 中途插入新 prefill，而不是整批跑完再接下一批 |
| `prefix_cache` | 满 block 按内容哈希共享，`refcnt=0` 进 idle，缺块再淘汰 |

三个类：`BlockManager` / `Scheduler` / `Engine`。

## 构建

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/bench_serving
```

## 实测（本机，假 token）

`n=64 prompt=32 gen=16 max_seqs=8 gap=120us`，2026-09-15：

| 臂 | peak KV blocks | mean TTFT (us) | prefix hits |
|---|---:|---:|---:|
| full (paged+cont+prefix) | **24** | 16728 | **94** |
| no_prefix | 24 | 16728 | 0 |
| no_continuous_batch | 24 | 16895 | 94 |
| no_paged（reserve max_len） | **48** | 16728 | 0 |

Paged 相对 naive 预留，峰值块数减半。Prefix 命中可测。本负载下连续批处理的 TTFT 优势很小（到达间隔相对 decode 偏疏），不要写成已经打平 vLLM。

## 不是什么

- 不是真实模型 / CUDA kernel / HTTP 网关
- 不是 flowcoro 的一次版本升级
- IO 与协程接入是下一刀，本仓先把调度与 KV 锁死

## License

MIT
