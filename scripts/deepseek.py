"""最小的 DeepSeek API 客户端（不用装任何库）。key 从 .env 的 DEEPSEEK_API_KEY 读。"""

import json
import os
import time
import urllib.request
from pathlib import Path

API = "https://api.deepseek.com/chat/completions"
ROOT = Path(__file__).resolve().parent.parent


def _key():
    if os.environ.get("DEEPSEEK_API_KEY"):
        return os.environ["DEEPSEEK_API_KEY"]
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if line.startswith("DEEPSEEK_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("没找到 DEEPSEEK_API_KEY（.env）")


def chat(system, user, model="deepseek-v4-pro", json_mode=False, retries=3, timeout=600):
    """返回 (回复文字, usage)。"""
    body = {"model": model, "messages": [{"role": "system", "content": system},
                                         {"role": "user", "content": user}]}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    req = urllib.request.Request(API, data=json.dumps(body).encode("utf-8"), method="POST",
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {_key()}"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"], data.get("usage", {})
        except Exception as e:  # 网络抖动 / 限流：等一下再试
            if attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))


def chat_json(system, user, **kw):
    text, usage = chat(system, user, json_mode=True, **kw)
    return json.loads(text), usage
