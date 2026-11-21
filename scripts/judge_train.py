"""训练样本体检（2026-11-21）：用和考卷盲评完全一样的评审说明（judge_exam.SYS_JUDGE_T，v1）评训练样本本身的「原话 / 标准答案」，
看小说样本是不是常常没把卡上「要办成」办到（假设：模型从小说学到「要办成只是个大概」→ 考卷上没办成变多）。

对照：DeepSeek 写的样本（batch-1/2 旧数据、batch-3）各随机抽一些，用同一说明评。
用法：.venv/bin/python scripts/judge_train.py 输出.jsonl [--n-ds 90] [--workers 8]
统计：同一脚本加 --stats 只读输出文件打表。
"""
import argparse
import json
import random
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from deepseek import chat_json  # noqa: E402
from judge_exam import HARD, MODEL, SYS_JUDGE_T  # noqa: E402
from render_card import render_card  # noqa: E402
from to_mlx import TRAINABLE  # noqa: E402

NOVEL = ["novel-1", "novel-2", "novel-3", "novel-3b", "novel-4"]
DS = {"旧数据": ["batch-1", "batch-2"], "batch-3": ["batch-3"]}


def kept(d):
    p = ROOT / "data" / d / "samples.json"
    return [r for r in json.loads(p.read_text(encoding="utf-8")) if r.get("meta", {}).get("keep") is True and r["source"] in TRAINABLE]


def lines_of(r):
    out = []
    for x in r["output"]:
        emo = f"[{x['emotion']}]" if x.get("emotion") else ""
        out.append(f"{x['name']}{emo}: {x['text']}")
    return out


def group_of(d, r):
    if d in NOVEL:
        return "小说:" + ((r.get("meta", {}).get("ref") or {}).get("book") or d)
    return next(g for g, ds in DS.items() if d in ds)


def stats(path):
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    by = defaultdict(list)
    for r in rows:
        g = r["group"]
        by["小说（全部）" if g.startswith("小说:") else g].append(r)
        if g.startswith("小说:"):
            by[g].append(r)
    print(f"{'组':14s} {'条数':>4s} {'办成2':>6s} {'办成1':>6s} {'办成0':>6s} {'心里话说出口':>8s} {'全过':>6s} {'像不像':>6s}")
    for g in sorted(by, key=lambda k: (k.startswith("小说:"), k)):
        v = by[g]; n = len(v); b = Counter(x["办成"] for x in v)
        pct = lambda k: f"{100 * k / n:.0f}%"
        print(f"{g:14s} {n:4d} {pct(b[2]):>6s} {pct(b[1]):>6s} {pct(b[0]):>6s} {pct(sum('心里话说出口' in x['problems'] for x in v)):>8s} "
              f"{pct(sum(x['pass'] for x in v)):>6s} {sum(x['像不像'] or 0 for x in v) / n:6.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n-ds", type=int, default=90, help="每组 DeepSeek 样本抽几条")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--stats", action="store_true")
    a = ap.parse_args()
    if a.stats:
        return stats(a.out)
    sys_judge = SYS_JUDGE_T.format(inner_extra="")
    rnd = random.Random(20261121)
    todo, seen = [], set()
    for d in NOVEL:
        for r in kept(d):
            k = json.dumps(r["output"], ensure_ascii=False)
            if k not in seen:
                seen.add(k)
                todo.append((d, r))
    for g, ds in DS.items():
        pool = [(d, r) for d in ds for r in kept(d)]
        rnd.shuffle(pool)
        todo += pool[:a.n_ds]
    out = Path(a.out)
    done = set()
    if out.exists():
        done = {json.loads(l)["key"] for l in out.read_text(encoding="utf-8").splitlines() if l.strip()}
    todo = [(d, r) for d, r in todo if f"{d}:{r['id']}" not in done]
    rnd.shuffle(todo)
    print(f"要评 {len(todo)} 条（已评 {len(done)}），模型 {MODEL}", flush=True)
    lock, usage, n = threading.Lock(), {"in": 0, "out": 0}, [0]

    def one(item):
        d, r = item
        body = "\n".join(f"{k}. {s}" for k, s in enumerate(lines_of(r), 1))
        data, u = chat_json(sys_judge, f"{render_card(r['card'])}\n\n【台词】\n{body}", model=MODEL, thinking=True)
        probs = [k for k in HARD if data.get(k) is True]
        dv = data.get("办成", 0) if isinstance(data.get("办成"), int) else 0
        if dv < 1:
            probs.append("没办成")
        rec = {"key": f"{d}:{r['id']}", "group": group_of(d, r), "lines": "\n".join(lines_of(r)), "pass": not probs,
               "problems": probs, "办成": dv, "像不像": data.get("像不像"), "理由": data.get("理由", ""), "model": MODEL}
        with lock:
            with out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            usage["in"] += u.get("prompt_tokens", 0); usage["out"] += u.get("completion_tokens", 0); n[0] += 1
            if n[0] % 50 == 0:
                print(f"  {n[0]}/{len(todo)}  token 入 {usage['in']} 出 {usage['out']}", flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        for f in [ex.submit(one, t) for t in todo]:
            try:
                f.result()
            except Exception as e:
                print("失败：", e, flush=True)
    print(f"完成 {n[0]} 条；token 入 {usage['in']} 出 {usage['out']}")


if __name__ == "__main__":
    main()
