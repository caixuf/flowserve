#!/usr/bin/env python3
"""托管 frontend，默认接到 Qwen2.5-0.5B；可用 --backend tinymla 回退 chat_cli。"""

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
QWEN_DIR = os.path.join(os.path.dirname(WORKSPACE_DIR), "qwen_chatbot")
_CHAT_BIN = os.path.join(WORKSPACE_DIR, "tinymla_chat.bin")
_STORY_BIN = os.path.join(WORKSPACE_DIR, "tinymla_story.bin")
WEIGHTS_PATH = _CHAT_BIN if os.path.exists(_CHAT_BIN) else _STORY_BIN

if QWEN_DIR not in sys.path:
    sys.path.insert(0, QWEN_DIR)


class TinyMlaEngine:
    def __init__(self, cli_path, weights_path):
        self.kind = "tinymla"
        self.model_path = weights_path
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
            raise RuntimeError("chat_cli --serve 未启动")
        clean = prompt.replace("\n", " ")
        line = f"{int(max_tokens)}\tUser: {clean}\\nAssistant:\n"
        with self.lock:
            self.proc.stdin.write(line.encode("utf-8"))
            self.proc.stdin.flush()
            while True:
                ch = self.proc.stdout.read(1)
                if not ch:
                    raise RuntimeError("chat_cli 已退出")
                if ch == b"\0":
                    break
                yield ch.decode("utf-8", errors="ignore")

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
            body = json.dumps({
                "status": "ok",
                "engine": getattr(ENGINE, "kind", "unknown"),
                "model_path": getattr(ENGINE, "model_path", ""),
                "engine_alive": ENGINE.alive(),
            }).encode()
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
        max_tokens = int(payload.get("max_tokens", 256))
        if not prompt:
            self.send_error(400, "prompt required")
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        try:
            for delta in ENGINE.generate(prompt, max_tokens):
                if not delta:
                    continue
                chunk = json.dumps({"delta": delta}, ensure_ascii=False)
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
    parser.add_argument("--backend", choices=("qwen", "tinymla"), default="qwen")
    parser.add_argument("--weights", default=WEIGHTS_PATH, help="tinymla FLSV 或覆盖 Qwen 目录")
    args = parser.parse_args()

    if args.backend == "qwen":
        from engine import QwenChatEngine, default_model_dir
        model_path = args.weights if os.path.isdir(args.weights) else default_model_dir()
        print(f"Loading Qwen2.5-0.5B from {model_path} ...")
        ENGINE = QwenChatEngine(model_path)
        ENGINE.kind = "qwen2.5-0.5b"
    else:
        ENGINE = TinyMlaEngine(CLI_PATH, args.weights)

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
    print(f"backend {getattr(ENGINE, 'kind', '?')}  model {getattr(ENGINE, 'model_path', '')}  alive={ENGINE.alive()}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutdown")
    finally:
        if hasattr(ENGINE, "close"):
            ENGINE.close()
        httpd.server_close()


if __name__ == "__main__":
    run()
