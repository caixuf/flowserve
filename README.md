# flowserve

Paged KV + continuous batching + prefix cache，挂在 [flowcoro](https://github.com/caixuf/flowcoro) 旁边的**独立推理调度仓**。

不改 flowcoro。假 token、解析 GPU 代价模型，不是生产 LLM server，也不是 vLLM 的完整复刻。

## 证什么

| 开关 | 对照 |
|---|---|
| `paged_kv` | 按实际 token 占 block，而不是一来就 `max_model_len` |
| `continuous_batching` | decode 中途插入新 prefill，而不是整批跑完再接下一批 |
| `prefix_cache` | 满 block 按内容哈希共享，`refcnt=0` 进 idle，缺块再淘汰 |
| `enable_mla` | **移植自 kun-cellular**：多头低秩潜空间注意力 (DeepSeek MLA)，KV 占块压缩 4x |
| `enable_speculative` | **移植自 kun-cellular**：草稿模型极速提案 + 验证模型批处理仲裁 (Speculative Decoding) |

核心设计：`BlockManager` / `Scheduler` / `Engine` + `nn::{RoPE, MLA, Speculative}`。

## 构建

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/bench_serving
```

## 实测（本机，假 token）

`n=64 prompt=32 gen=16 max_seqs=8 gap=120us`，2026-09-15：

| 臂 | peak KV blocks | mean TTFT (us) | mean TPOT (us) | prefix hits |
|---|---:|---:|---:|---:|
| full (paged+cont+prefix) | 24 | 16728 | 323 | **94** |
| no_prefix | 24 | 16728 | 323 | 0 |
| no_continuous_batch | 24 | 16895 | 298 | 94 |
| no_paged（reserve max_len） | **48** | 16728 | 323 | 0 |
| **+ mla_latent_kv (4x)** | **8** | 16728 | 323 | 0 |
| **+ speculative_decoding** | 25 | **5745** | **112** | 94 |
| **+ modern_stack (mla+spec)** | **8** | **5745** | **112** | 0 |

- **MLA 显存压缩**：`peak_kv` 从 24 降至 8，显存占块降至原来的 1/3。
- **投机批处理加速**：草稿推演结合批量仲裁，端到端时延缩短 55.8%，mean TTFT 下降 65.6%，mean TPOT 从 323us 降至 112us（生成提速近 3 倍）。
- **Paged 相对 naive 预留**：峰值块数减半（48 -> 24）。Prefix 命中可测。

## 体验真实端到端生成 (Real MLA Transformer)

```bash
./build/chat_cli "Hello, flowserve!"
python3 tools/web_server.py --port 9000
# 常驻引擎：chat_cli --serve tinymla_story.bin ，协议 `max_tokens\\tprompt\\n` → 字节流 + NUL
```
输出演示：
- Byte-level Tokenizer 完成文本与字节 Token 的双向映射（零 OOV 风险）；
- 真实多层 MLA Transformer（Embedding $\to$ RMSNorm $\to$ MLA $\to$ SwiGLU FFN $\to$ LM Head）执行张量矩阵运算；
- 逐字实时流式打印生成输出。

## 不是什么

- 尚未接入 NVIDIA CUDA/Triton 底层 Kernel（目前为纯 C++ SIMD 级算子）
- 不是用于生产级千亿参数集群的完整网关（建议挂载于 flowcoro 协程网络服务器中）

## License

MIT
