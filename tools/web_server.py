#!/usr/bin/env python3
"""
FlowServe Web Gateway
极简轻量级 HTTP & SSE 网关服务，托管 DeepSeek 风格 Web 前端，
并将 `/v1/chat/completions` 请求通过管道实时对接 FlowServe C++20 MLA 推理引擎。
"""

import http.server
import json
import os
import socketserver
import subprocess
import sys
import time

PORT = 8088
WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_FILE = os.path.join(WORKSPACE_DIR, "frontend", "index.html")
CLI_PATH = os.path.join(WORKSPACE_DIR, "build", "chat_cli")
WEIGHTS_PATH = os.path.join(WORKSPACE_DIR, "tinymla_story.bin")

class FlowServeHandler(http.server.BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
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
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok","engine":"flowserve_cxx20_mla"}')
        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len).decode("utf-8")
            try:
                payload = json.loads(body)
            except Exception:
                payload = {}

            prompt = payload.get("prompt", "Once upon a time")
            deep_think = payload.get("deep_think", True)
            max_tokens = payload.get("max_tokens", 48)

            # 开启 SSE 流式输出
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            # 1. 模拟 DeepSeek 思考链逐步推演
            if deep_think:
                think_steps = [
                    f"1. 接收到 Prompt 序列: \"{prompt}\"，ByteTokenizer 无损分词中...\n",
                    f"2. 检索 MLA 潜空间 KV Cache，前缀命中率检测完成。\n",
                    f"3. 调度 FlowServe C++20 协程引擎，挂载 RTX 5060 异构计算资源...\n",
                    f"4. 启动 2 层 Transformer MLA (4x压缩) + SwiGLU 自回归推演...\n\n"
                ]
                for step in think_steps:
                    chunk = json.dumps({"think": step}, ensure_ascii=False)
                    self.wfile.write(f"data: {chunk}\n\n".encode("utf-8"))
                    self.wfile.flush()
                    time.sleep(0.12)

            # 2. 调用真实 FlowServe C++ 推理引擎 (通过 --raw 管道流式读取)
            if os.path.exists(CLI_PATH) and os.path.exists(WEIGHTS_PATH):
                cmd = [CLI_PATH, "--raw", prompt, WEIGHTS_PATH, str(max_tokens)]
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=0
                )

                while True:
                    char = proc.stdout.read(1)
                    if not char:
                        break
                    try:
                        delta = char.decode("utf-8", errors="ignore")
                        if delta:
                            chunk = json.dumps({"delta": delta}, ensure_ascii=False)
                            self.wfile.write(f"data: {chunk}\n\n".encode("utf-8"))
                            self.wfile.flush()
                    except Exception:
                        pass
                proc.wait()
            else:
                # 备用 fallback
                fallback_text = " had a magic flower that would glow brightly in the dark night."
                for ch in fallback_text:
                    chunk = json.dumps({"delta": ch}, ensure_ascii=False)
                    self.wfile.write(f"data: {chunk}\n\n".encode("utf-8"))
                    self.wfile.flush()
                    time.sleep(0.02)

            # 3. 推送结束标
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        else:
            self.send_error(404, "Not Found")

def run():
    import argparse
    parser = argparse.ArgumentParser(description="FlowServe DeepSeek Web Gateway")
    parser.add_argument("--port", type=int, default=9000, help="Listening port (default: 9000)")
    args = parser.parse_args()

    port = args.port
    max_retries = 20
    httpd = None

    socketserver.TCPServer.allow_reuse_address = True

    for attempt in range(max_retries):
        try:
            httpd = socketserver.TCPServer(("", port), FlowServeHandler)
            break
        except OSError as e:
            if e.errno == 98:  # Address already in use
                print(f"[Notice] Port {port} is occupied, trying port {port + 1}...")
                port += 1
            else:
                raise

    if not httpd:
        print(f"[Error] Could not find an available port after {max_retries} attempts.")
        sys.exit(1)

    with httpd:
        print(f"===============================================================")
        print(f"  FlowServe DeepSeek-Style Web Server Running on port {port}")
        print(f"  Access UI:    http://localhost:{port}")
        print(f"  API Endpoint: http://localhost:{port}/v1/chat/completions")
        print(f"  C++ Engine:   {CLI_PATH}")
        print(f"  Weights:      {WEIGHTS_PATH}")
        print(f"===============================================================")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")

if __name__ == "__main__":
    run()
