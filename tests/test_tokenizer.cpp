#include "flowserve/tokenizer.hpp"
#include <cassert>
#include <iostream>

using namespace flowserve;

int main() {
    Tokenizer tok;
    assert(tok.vocab_size() == 259);

    // 1. 英文测试
    std::string text_en = "Hello, flowserve!";
    auto tokens_en = tok.encode(text_en, false);
    assert(tokens_en.size() == text_en.size());
    std::string decoded_en = tok.decode(tokens_en);
    assert(decoded_en == text_en);

    // 2. 中文测试 (UTF-8)
    std::string text_zh = "你好，大模型框架！";
    auto tokens_zh = tok.encode(text_zh, true);
    assert(tokens_zh[0] == Tokenizer::BOS_TOKEN);
    std::string decoded_zh = tok.decode(tokens_zh);
    assert(decoded_zh == text_zh);

    // 3. 单字流式拼接测试
    std::string streamed;
    for (int t : tokens_zh) {
        streamed += tok.decode_token(t);
    }
    assert(streamed == text_zh);

    std::cout << "[PASS] test_tokenizer passed successfully.\n";
    return 0;
}
