"""本机演员模型服务（2026-11-26，接游戏第一步：Godot 在 Mac 上用 HTTPRequest 调这里）。

只听 127.0.0.1（不对外）。一次只处理一个请求（MLX 一张卡一张卡地算；游戏同一时间也只有一段对话在等）。

  .venv/bin/python scripts/serve.py [模型目录] [--port 8765]

接口（JSON，UTF-8）：
  GET  /health                 → {"ok": true, "model": "..."}
  POST /warm  {"card": {...}}  → {"seconds": 0.98}            NPC 进入视野时调：预读档案前缀（【世界】+【说话人】）
  POST /say   {"card": {...}, "seed": 0}
                               → {"lines": [{"name","emotion","action","text"}] | null,
                                  "reject": "", "fixes": [...], "tries": 1, "seconds": 0.8, "first_token": 0.35}
     lines 为 null = 退回，游戏用手写台词。card 格式同 docs/戏卡格式.md（代码里的 dict，见 exam/考卷.jsonl）。
  出错 → HTTP 400 / 500 + {"error": "..."}

  --watch-pid PID：那个进程（游戏）没了就自己退出。游戏开着时自动拉起服务用（Godot 那边的客户端），
  这样从编辑器点停止（会直接杀掉游戏、来不及收尾）也不会留下一个占 2 GB 的服务。
"""

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from actor import Actor  # noqa: E402

DEFAULT_MODEL = HERE.parent / "models" / "q35-2b-v3-dpo2-s1-mlx4"   # 当前最好


def check_card(card):
    if not isinstance(card, dict):
        raise ValueError("card 要是一个对象")
    for k in ("world", "speaker", "listener", "scene", "goal", "rules"):
        if k not in card:
            raise ValueError(f"card 缺 {k}")
    r = card["rules"]
    for k in ("use", "lines", "max_chars", "emotions", "speakers"):
        if k not in r:
            raise ValueError(f"card.rules 缺 {k}")


def make_handler(actor, model_name, log):
    class H(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):      # 默认日志太吵，自己记
            pass

        def do_GET(self):
            if self.path == "/health":
                return self._send(200, {"ok": True, "model": model_name})
            self._send(404, {"error": "没有这个接口"})

        def do_POST(self):
            try:
                n = int(self.headers.get("Content-Length") or 0)
                req = json.loads(self.rfile.read(n).decode("utf-8") or "{}")
                card = req.get("card")
                check_card(card)
            except Exception as e:
                return self._send(400, {"error": f"请求不对：{e}"})
            try:
                if self.path == "/warm":
                    s = actor.warm(card)
                    log(f"warm {card['speaker'].get('name')} {s:.2f}s")
                    return self._send(200, {"seconds": round(s, 3)})
                if self.path == "/say":
                    res = actor.say(card, seed=int(req.get("seed", time.time_ns() % 100000)))
                    log(f"say {card['speaker'].get('name')} {res.seconds:.2f}s 试{res.tries}"
                        + (f" 退回:{res.reject}" if not res.lines else "") + (f" 修:{'；'.join(res.fixes)}" if res.fixes else ""))
                    return self._send(200, {"lines": res.lines, "reject": res.reject, "fixes": res.fixes, "tries": res.tries,
                                            "seconds": round(res.seconds, 3), "first_token": round(res.first_token, 3)})
                self._send(404, {"error": "没有这个接口"})
            except Exception as e:
                log(f"出错 {e!r}")
                self._send(500, {"error": repr(e)})
    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", nargs="?", default=str(DEFAULT_MODEL))
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--watch-pid", type=int, default=0, help="这个进程没了就退出")
    a = ap.parse_args()

    def log(msg):
        print(time.strftime("%H:%M:%S"), msg, flush=True)
    t = time.perf_counter()
    actor = Actor(a.model)
    log(f"模型加载 {time.perf_counter() - t:.1f}s：{a.model}")
    srv = HTTPServer(("127.0.0.1", a.port), make_handler(actor, Path(a.model).name, log))
    log(f"在 http://127.0.0.1:{a.port} 等游戏调用（Ctrl+C 退出）")
    if a.watch_pid:
        def watch():
            while True:
                time.sleep(2)
                try:
                    os.kill(a.watch_pid, 0)
                except ProcessLookupError:
                    log(f"游戏进程 {a.watch_pid} 没了，退出")
                    os._exit(0)
                except PermissionError:
                    pass
        threading.Thread(target=watch, daemon=True).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
