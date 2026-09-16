# flowserve

独立 **LLM 推理调度**仓：Paged KV、连续批、前缀缓存、MLA 占块压缩、投机解码的调度几何。  
旁边是 [flowcoro](https://github.com/caixuf/flowcoro)（协程）和 [flowtrain](https://github.com/caixuf/flowtrain)（训练机制测试台），互不塞进对方仓库。

**学怎么工作：** [docs/LEARN.md](docs/LEARN.md)（开关 → BlockManager → Scheduler → 怎么读 bench）。

不是生产 LLM server，也不是 vLLM。`bench_serving` 用假 token + 解析 GPU 代价模型。网页默认对话走开源 Qwen，不是本仓 C++ 在跑 0.5B。

## 两套演示，别混

```text
浏览器 :9000
    └─ web_server.py --backend qwen     → Qwen2.5-0.5B-Instruct（Transformers / GPU）
    └─ web_server.py --backend tinymla  → chat_cli + tinymla_*.bin（C++ mmap + TinyMLA）
C++ 调度与消融
    └─ bench_serving / ctest            → BlockManager + Scheduler（假 token）
```

TinyMLA 约 10 万参数，会背 FLSV 里那点语料。Qwen 是阿里基座；小语料 LoRA 合并权重会把算术训坏，默认不要挂 `merged_model`。详见 [qwen_chatbot](https://github.com/caixuf/qwen_chatbot)（若未独立开仓，见同级目录 `../qwen_chatbot`）。

## 构建

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/bench_serving
./run_demo.sh          # 可选：菜单里选 Web / bench / TinyMLA
```

```bash
python3 tools/web_server.py --port 9000 --backend qwen
./build/chat_cli --weights tinymla_chat.bin --prompt "User: 什么是 MLA？\nAssistant:"
```

## 证什么

| 开关 | 你在比什么 |
|---|---|
| `paged_kv` | 按实际 token 占 block，而不是一来就 `max_model_len` |
| `continuous_batching` | decode 中途插入新 prefill |
| `prefix_cache` | 满 block 按内容哈希共享，`refcnt=0` 进 idle |
| `enable_mla` | 潜空间 KV，占块按 `effective_block_size()` 变少 |
| `enable_speculative` | 草稿多步槽位 + 验证；时延来自代价模型 |

源码入口：`include/flowserve/{types,block_manager,scheduler,engine}.hpp`。

## 本机消融（假 token）

`n=64 prompt=32 gen=16 max_seqs=8 gap=120us`：

| 臂 | peak KV blocks | mean TTFT (us) | mean TPOT (us) | prefix hits |
|---|---:|---:|---:|---:|
| full (paged+cont+prefix) | 24 | 16728 | 323 | **94** |
| no_prefix | 24 | 16728 | 323 | 0 |
| no_continuous_batch | 24 | 16895 | 298 | 94 |
| no_paged（reserve max_len） | **48** | 16728 | 323 | 0 |
| + mla_latent_kv (4x) | **8** | 16728 | 323 | 0 |
| + speculative_decoding | 25 | **5745** | **112** | 94 |
| + modern_stack (mla+spec) | **8** | **5745** | **112** | 0 |

Paged 相对 naive：48 → 24。MLA：24 → 8。投机数字是调度+代价模型，不是 CUDA kernel 加速。

## 不是什么

- TinyMLA 前向是 C++，未接 cuBLAS/FlashAttention；1B 级模型不能当本引擎的「已支持」。
- 不是千亿集群网关。
- 训练并行在 flowtrain；本仓不管 NCCL/ZeRO。

## License

MIT
