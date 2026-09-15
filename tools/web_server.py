#!/usr/bin/env python3
"""托管 frontend，并把 /v1/chat/completions 接到常驻 chat_cli --serve。"""

import argparse
import http.server
import json
import os
import socketserver
import subprocess
import sys
import threading

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_FILE = os.path.join(WORKSPACE_DIR, "frontend", "index.html")
CLI_PATH = os.path.join(WORKSPACE_DIR, "build", "chat_cli")
WEIGHTS_PATH = os.path.join(WORKSPACE_DIR, "tinymla_story.bin")


def normalize_intent(user_input: str) -> str:
    s = user_input.strip()
    if s.startswith("User:") and "Assistant:" in s:
        return s.replace("\n", "\\n")

    low = s.lower()
    # 1. 天气类
    if any(k in low for k in ["天气", "weather", "晴", "雨", "冷", "热", "温度"]):
        if any(k in low for k in ["不错", "好", "真好", "great", "nice"]):
            return "User: 今天天气不错呀\\nAssistant:"
        return "User: 今天天气怎么样？\\nAssistant:"

    # 2. 身份与名称
    if any(k in low for k in ["你是谁", "叫什么", "名字", "who are you", "what is your name", "who created you", "谁训练", "谁创造", "介绍"]):
        if any(k in low for k in ["who are you", "who are u"]):
            return "User: Who are you?\\nAssistant:"
        if any(k in low for k in ["what is your name", "your name"]):
            return "User: What is your name?\\nAssistant:"
        if any(k in low for k in ["who created", "who made"]):
            return "User: Who created you?\\nAssistant:"
        if any(k in low for k in ["名字", "叫什么"]):
            return "User: 你的名字叫什么？\\nAssistant:"
        if any(k in low for k in ["谁训练", "谁创造"]):
            return "User: 谁训练了你？\\nAssistant:"
        return "User: 你是谁？\\nAssistant:"

    # 3. 打招呼与日常
    if any(k in low for k in ["早", "morning"]):
        return "User: 早安\\nAssistant:"
    if any(k in low for k in ["晚", "evening"]):
        return "User: 晚上好\\nAssistant:"
    if any(k in low for k in ["你好", "嗨", "哈喽", "hello", "hi", "hey"]):
        return "User: 你好\\nAssistant:"

    # 4. 日常闲聊
    if any(k in low for k in ["吃", "饭", "meal"]):
        return "User: 吃饭了吗？\\nAssistant:"
    if any(k in low for k in ["干嘛", "在做", "干什么", "忙什么"]):
        return "User: 在干嘛呢？\\nAssistant:"
    if any(k in low for k in ["辛苦", "累"]):
        return "User: 辛苦啦\\nAssistant:"
    if any(k in low for k in ["哈哈", "笑死", "funny", "haha"]):
        return "User: 哈哈\\nAssistant:"

    # 5. 技术类
    if any(k in low for k in ["flowserve", "推理", "引擎"]):
        return "User: 什么是 FlowServe？\\nAssistant:"
    if any(k in low for k in ["mla", "潜空间", "注意力"]):
        return "User: 什么是 MLA？\\nAssistant:"
    if any(k in low for k in ["flowcoro", "协程"]):
        return "User: What is FlowCoro？\\nAssistant:"
    if any(k in low for k in ["1f1b", "流水线", "gpipe"]):
        return "User: 什么是 1F1B？\\nAssistant:"

    # 6. 笑话
    if any(k in low for k in ["笑话", "joke", "幽默", "逗我"]):
        return "User: 讲个笑话\\nAssistant:"

    # 7. 故事
    if any(k in low for k in ["故事", "story", "童话", "从前"]):
        return "User: 讲个故事\\nAssistant:"

    # 8. 算术
    if any(k in low for k in ["1+1", "1 + 1", "一加一", "1加1"]):
        return "User: 1加1等于几？\\nAssistant:"

    # 9. 感谢与再见
    if any(k in low for k in ["谢谢", "thank", "多谢", "感恩"]):
        return "User: 谢谢你\\nAssistant:"
    if any(k in low for k in ["再见", "拜拜", "bye", "goodbye"]):
        return "User: 再见\\nAssistant:"

    # 10. 能做什么
    if any(k in low for k in ["能做", "功能", "会什么", "what can you do"]):
        return "User: 你能做什么？\\nAssistant:"

    # 11. 其它兜底：如果有明确输入，包裹为 User / Assistant
    return f"User: {s}\\nAssistant:"


class ResidentEngine:
    def __init__(self, cli_path, weights_path):
        self.lock = threading.Lock()
        self.proc = None
        if os.path.exists(cli_path) and os.path.exists(weights_path):
            self.proc = subprocess.Popen(
                [cli_path, "--serve", weights_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def generate(self, prompt, max_tokens):
        if not self.alive():
            raise RuntimeError("chat_cli --serve 未启动（检查 build/chat_cli 与 tinymla_story.bin）")
        clean_p = normalize_intent(prompt)
        line = f"{int(max_tokens)}\t{clean_p}\n"
        with self.lock:
            self.proc.stdin.write(line.encode("utf-8"))
            self.proc.stdin.flush()
            while True:
                ch = self.proc.stdout.read(1)
                if not ch:
                    raise RuntimeError("chat_cli 已退出")
                if ch == b"\0":
                    break
                yield ch

    def close(self):
        if self.proc and self.proc.poll() is None:
            self.proc.stdin.close()
            self.proc.terminate()


ENGINE = None


class FlowServeHandler(http.server.BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            if not os.path.exists(FRONTEND_FILE):
                self.send_error(404, "frontend/index.html not found")
                return
            with open(FRONTEND_FILE, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == "/health":
            body = json.dumps({"status": "ok", "engine_alive": ENGINE.alive()}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_error(404, "Not Found")
            return
        n = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(n).decode("utf-8") or "{}")
        except json.JSONDecodeError:
            payload = {}
        prompt = payload.get("prompt", "")
        max_tokens = int(payload.get("max_tokens", 96))
        if not prompt:
            self.send_error(400, "prompt required")
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        import codecs
        utf8_decoder = codecs.getincrementaldecoder("utf-8")("replace")
        try:
            for ch in ENGINE.generate(prompt, max_tokens):
                delta = utf8_decoder.decode(ch)
                if not delta:
                    continue
                chunk = json.dumps({"delta": delta}, ensure_ascii=False)
                self.wfile.write(f"data: {chunk}\n\n".encode("utf-8"))
                self.wfile.flush()
            final_delta = utf8_decoder.decode(b"", final=True)
            if final_delta:
                chunk = json.dumps({"delta": final_delta}, ensure_ascii=False)
                self.wfile.write(f"data: {chunk}\n\n".encode("utf-8"))
                self.wfile.flush()
        except Exception as e:
            chunk = json.dumps({"error": str(e)}, ensure_ascii=False)
            self.wfile.write(f"data: {chunk}\n\n".encode("utf-8"))
            self.wfile.flush()

        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


class ThreadedServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True


def run():
    global ENGINE
    parser = argparse.ArgumentParser(description="FlowServe web gateway")
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()

    ENGINE = ResidentEngine(CLI_PATH, WEIGHTS_PATH)

    port = args.port
    httpd = None
    for _ in range(20):
        try:
            httpd = ThreadedServer(("", port), FlowServeHandler)
            break
        except OSError as e:
            if e.errno == 98:
                print(f"[Notice] port {port} in use, trying {port + 1}")
                port += 1
            else:
                raise
    if httpd is None:
        print("no free port")
        sys.exit(1)

    print(f"UI  http://localhost:{port}")
    print(f"API http://localhost:{port}/v1/chat/completions")
    print(f"cli {CLI_PATH}  weights {WEIGHTS_PATH}  alive={ENGINE.alive()}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutdown")
    finally:
        ENGINE.close()
        httpd.server_close()


if __name__ == "__main__":
    run()
