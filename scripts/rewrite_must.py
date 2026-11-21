"""小说卡只重写「要办成」（2026-11-21）：改成原话**实际办到的那一步**，写法照考卷（一两件具体的事）。

为什么：训练样本体检（judge_train.py）发现小说原话完全办到「要办成」只有 26%（DeepSeek 样本 60–68%）——
11-20 规定小说卡「要办成」只写大方向，而原话常常只是长场景里的一步；带小说训练的版本考卷上「没办成」多 9–11 个百分点。
其余字段一概不动。机器查：答案里任意 4 字出现在新「要办成」里 → 带提示重写，最多 3 次，还不行就保留旧的并标记。

用法：.venv/bin/python scripts/rewrite_must.py [--workers 8]   输出 data/novel-must/samples.json（中断重跑会接着做）
"""
import argparse
import copy
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from deepseek import chat_json  # noqa: E402
from judge_train import NOVEL, kept, lines_of  # noqa: E402
from render_card import render_card  # noqa: E402

MODEL = "deepseek-flash"
SYS = """你在给「演员训练数据」修一张戏卡。卡上【目的】的「要办成」写得太虚，和下面这段【原话】实际做到的事对不上。请只重写「要办成」。

要求：
1. 「要办成」= 这段原话在这一场里**实际办到的事**（结果），写一两件，具体到能核对「办到没有」。像这样：
   「回绝拜师，让他死了这条心回家去」「把怪话当胡话岔开，把客人拉回买牲口的生意上」「不跟他吵翻，但要让他把错字改了」「请对方先带自己去见她父亲」。
2. 原话只做了一步（比如只是先稳住对方、先约见、先问一句），就只写这一步，**不要写成整件大事的最终目标**。
3. 写结果、不写说法：不引用原话，不写他用什么词、讲什么理由。原话里任意连着的 4 个字都不能出现在「要办成」里。
4. 不能把【心里想的】或「别说漏」里要藏的东西写进「要办成」。
5. 用现代白话，一句话，30 字以内为宜。

只输出 JSON：{"must": "新的要办成"}"""


def leak4(must, output):
    for o in output:
        t = re.sub(r"[，。！？、；：…—「」『』\s]", "", o["text"])
        for k in range(0, max(0, len(t) - 3)):
            if t[k:k + 4] in must:
                return t[k:k + 4]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    out_dir = ROOT / "data/novel-must"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "samples.json"
    done = {r["meta"]["from"]: r for r in json.loads(out.read_text(encoding="utf-8"))} if out.exists() else {}
    todo, seen = [], set()
    for d in NOVEL:
        for r in kept(d):
            k = json.dumps(r["output"], ensure_ascii=False)
            if k not in seen:
                seen.add(k)
                if f"{d}:{r['id']}" not in done:
                    todo.append((d, r))
    print(f"要重写 {len(todo)} 条（已做 {len(done)}），模型 {MODEL}", flush=True)
    lock, usage = threading.Lock(), {"in": 0, "out": 0, "retry": 0, "fail": 0}

    def one(item):
        d, r = item
        body = "\n".join(lines_of(r))
        user = f"{render_card(r['card'])}\n\n【原话】\n{body}"
        hint, new = "", None
        for _ in range(3):
            data, u = chat_json(SYS, user + hint, model=MODEL, thinking=True)
            with lock:
                usage["in"] += u.get("prompt_tokens", 0); usage["out"] += u.get("completion_tokens", 0)
            m = (data.get("must") or "").strip()
            bad = leak4(m, r["output"]) if m else "（空）"
            if not bad:
                new = m
                break
            hint = f"\n\n（上一版「要办成」用了原话里的「{bad}」，违反第 3 条，请换说法重写。）"
            with lock:
                usage["retry"] += 1
        s = copy.deepcopy(r)
        s["meta"] = dict(r.get("meta", {}), **{"from": f"{d}:{r['id']}", "must_old": r["card"]["goal"].get("must", ""),
                                               "must_ok": new is not None})
        if new:
            s["card"]["goal"]["must"] = new
        with lock:
            if new is None:
                usage["fail"] += 1
            done[s["meta"]["from"]] = s
            if len(done) % 20 == 0:
                out.write_text(json.dumps(list(done.values()), ensure_ascii=False, indent=1), encoding="utf-8")

    with ThreadPoolExecutor(a.workers) as ex:
        for f in [ex.submit(one, t) for t in todo]:
            try:
                f.result()
            except Exception as e:
                print("失败：", e, flush=True)
    out.write_text(json.dumps(list(done.values()), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"完成，共 {len(done)} 条；重写重试 {usage['retry']} 次，保留旧的 {usage['fail']} 条；token 入 {usage['in']} 出 {usage['out']}")


if __name__ == "__main__":
    main()
