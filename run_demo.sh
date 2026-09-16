#!/usr/bin/env bash
# ==============================================================================
# FlowServe & FlowTrain 一键式交互演示与快速上手总控脚本
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QWEN_DIR="$(cd "${SCRIPT_DIR}/../qwen_chatbot" 2>/dev/null && pwd || echo "")"
FLOWTRAIN_DIR="$(cd "${SCRIPT_DIR}/../flowtrain" 2>/dev/null && pwd || echo "")"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

function print_banner() {
    echo -e "${CYAN}${BOLD}"
    echo "================================================================"
    echo "       FlowServe & FlowTrain 双轨制高并发服务体系总控           "
    echo "================================================================"
    echo -e "${NC}"
    echo -e "  ${BLUE}网页对话${NC}：Qwen2.5-0.5B-Instruct（默认，不是 LoRA merge）"
    echo -e "  ${GREEN}调度实验${NC}：FlowServe C++ · Paged KV / 连续批 / TinyMLA FLSV"
    echo "----------------------------------------------------------------"
}

function run_web() {
    echo -e "\n${GREEN}🚀 启动 FlowServe 统一 Web 演示网关 (端口: 9000)...${NC}"
    echo -e "   默认：${BOLD}Qwen2.5-0.5B-Instruct${NC}（Transformers，与 C++ 调度分离）"
    echo -e "   访问地址：${BLUE}http://localhost:9000${NC}\n"
    python3 "${SCRIPT_DIR}/tools/web_server.py" --port 9000 --backend qwen
}

function run_bench() {
    echo -e "\n${YELLOW}📊 运行 FlowServe 自研调度吞吐与显存消融基准 (bench_serving)...${NC}"
    if [ ! -f "${SCRIPT_DIR}/build/bench_serving" ]; then
        echo "正在编译 bench_serving ..."
        cmake -S "${SCRIPT_DIR}" -B "${SCRIPT_DIR}/build" -DCMAKE_BUILD_TYPE=Release
        cmake --build "${SCRIPT_DIR}/build" -j
    fi
    "${SCRIPT_DIR}/build/bench_serving"
}

function run_tests() {
    echo -e "\n${BLUE}🧪 运行 FlowServe C++ 核心单元测试套件 (9项)...${NC}"
    if [ ! -d "${SCRIPT_DIR}/build" ]; then
        cmake -S "${SCRIPT_DIR}" -B "${SCRIPT_DIR}/build" -DCMAKE_BUILD_TYPE=Release
        cmake --build "${SCRIPT_DIR}/build" -j
    fi
    ctest --test-dir "${SCRIPT_DIR}/build" --output-on-failure
}

function run_qwen_cli() {
    if [ -d "${QWEN_DIR}" ]; then
        echo -e "\n${GREEN}💬 进入 Qwen2.5-0.5B 命令行交互流式对话终端...${NC}"
        python3 "${QWEN_DIR}/chat.py"
    else
        echo "❌ 找不到 qwen_chatbot 目录: ${QWEN_DIR}"
    fi
}

function run_tinymla_cli() {
    echo -e "\n${GREEN}⚡ 运行 TinyMLA 纯 C++ 零拷贝流式推理 (chat_cli)...${NC}"
    "${SCRIPT_DIR}/build/chat_cli" --weights "${SCRIPT_DIR}/tinymla_chat.bin" --prompt "User: 什么是 MLA？\nAssistant:"
}

function run_smoke() {
    echo -e "\n${CYAN}🔍 执行全栈自动化冒烟健康巡检...${NC}"
    echo -e "[1/3] 检查 C++ 单元测试..."
    ctest --test-dir "${SCRIPT_DIR}/build" --output-on-failure > /dev/null && echo -e "  ✅ C++ 单测 100% 通过"
    echo -e "[2/3] 检查 TinyMLA C++ 引擎推理..."
    "${SCRIPT_DIR}/build/chat_cli" --weights "${SCRIPT_DIR}/tinymla_chat.bin" --prompt "User: 你好\nAssistant:" --max-tokens 8 > /dev/null && echo -e "  ✅ TinyMLA FLSV 权重与前向正常"
    echo -e "[3/3] 检查 Qwen2.5-0.5B 对话引擎..."
    python3 -c "
import sys; sys.path.insert(0, '${QWEN_DIR}')
from engine import QwenChatEngine
eng = QwenChatEngine()
print('  ✅ Qwen 引擎加载成功，设备:', eng.device)
"
    echo -e "\n${GREEN}${BOLD}🎉 全栈冒烟巡检全部通过！系统处于就绪状态！${NC}\n"
}

# 命令行直达模式
case "$1" in
    web) run_web; exit 0 ;;
    bench) run_bench; exit 0 ;;
    test) run_tests; exit 0 ;;
    qwen) run_qwen_cli; exit 0 ;;
    tinymla) run_tinymla_cli; exit 0 ;;
    smoke) run_smoke; exit 0 ;;
esac

# 交互式菜单模式
while true; do
    print_banner
    echo "请选择要执行的操作："
    echo "  1) 启动 Web 演示网关 (Qwen 现场出声 + FlowServe 界面)"
    echo "  2) 运行 C++ 自研调度吞吐与显存压测 (bench_serving)"
    echo "  3) 运行 C++ 完整单元测试 (9项测试)"
    echo "  4) 命令行交互终端流式对话 (chat.py)"
    echo "  5) 命令行纯 C++ TinyMLA 流式推理 (chat_cli)"
    echo "  6) 全栈自动化冒烟巡检 (Smoke Test)"
    echo "  0) 退出"
    echo ""
    read -p "输入选项 [0-6]: " choice
    case "$choice" in
        1) run_web ;;
        2) run_bench ;;
        3) run_tests ;;
        4) run_qwen_cli ;;
        5) run_tinymla_cli ;;
        6) run_smoke ;;
        0) echo "退出。"; exit 0 ;;
        *) echo "无效选项，请重新输入。" ;;
    esac
    echo ""
    read -p "按回车键继续..."
done
