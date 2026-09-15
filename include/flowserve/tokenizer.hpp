#pragma once

// ============================================================================
// flowserve - 纯 C++ 字节级轻量分词器 (Byte-level Tokenizer)
// 
// 核心特性:
//   1. 零 OOV 风险: 基于 Byte-level 编码，100% 支持所有 UTF-8 字符（中文、英文、符号、emoji）；
//   2. 特殊 Token 支持: <bos>=0, <eos>=1, <pad>=2;
//   3. 基础词表: 256 个单字节 token (3~258)，预留可扩展高频子词 BPE 合并槽位;
//   4. 零第三方依赖: 纯 STL 标准库。
// ============================================================================

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace flowserve {

class Tokenizer {
public:
    static constexpr int BOS_TOKEN = 0;
    static constexpr int EOS_TOKEN = 1;
    static constexpr int PAD_TOKEN = 2;
    static constexpr int BYTE_OFFSET = 3;
    static constexpr int BASE_VOCAB_SIZE = 259; // 3 特殊标记 + 256 单字节

    Tokenizer() {
        init_vocab();
    }

    // 文本编码为 Token IDs
    std::vector<int> encode(const std::string& text, bool add_bos = true) const {
        std::vector<int> tokens;
        tokens.reserve(text.size() + (add_bos ? 1 : 0));
        if (add_bos) {
            tokens.push_back(BOS_TOKEN);
        }
        for (unsigned char c : text) {
            tokens.push_back(static_cast<int>(c) + BYTE_OFFSET);
        }
        return tokens;
    }

    // 单个 Token 解码为文本片段
    std::string decode_token(int token) const {
        if (token == BOS_TOKEN || token == PAD_TOKEN) {
            return "";
        }
        if (token == EOS_TOKEN) {
            return "<eos>";
        }
        if (token >= BYTE_OFFSET && token < BYTE_OFFSET + 256) {
            char c = static_cast<char>(token - BYTE_OFFSET);
            return std::string(1, c);
        }
        return "";
    }

    // 序列解码为完整文本
    std::string decode(const std::vector<int>& tokens) const {
        std::string result;
        result.reserve(tokens.size());
        for (int tok : tokens) {
            if (tok == BOS_TOKEN || tok == PAD_TOKEN || tok == EOS_TOKEN) continue;
            if (tok >= BYTE_OFFSET && tok < BYTE_OFFSET + 256) {
                result.push_back(static_cast<char>(tok - BYTE_OFFSET));
            }
        }
        return result;
    }

    size_t vocab_size() const { return BASE_VOCAB_SIZE; }

private:
    void init_vocab() {
        // 预留初始化
    }
};

} // namespace flowserve
