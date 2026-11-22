"""DPO 数据准备（2026-11-21）。目标：同一张卡，「藏住心思且办成」的台词当好例子（chosen），「心里话说出口 / 没办成」的当坏例子（rejected）。

卡：只用 v3 训练时**没见过**的训练卡（batch-1/2/3 挑稿留下的，带「心里想的」）；**不用考卷卡**（会污染考试）。
步骤：
  1. cards：.venv/bin/python scripts/dpo_prep.py cards [--n 300]  → data/dpo/cards.jsonl（考卷格式：id / card / ref）
  2. 本机生成：.venv/bin/python scripts/run_exam.py models/v3 data/dpo/gen --exam data/dpo/cards.jsonl --seeds 0,1,2,3 --temp 0.9
     截断：.venv/bin/python scripts/cut_exam.py data/dpo/gen data/dpo/gencut --exam data/dpo/cards.jsonl
  3. 盲评：.venv/bin/python scripts/judge_exam.py data/dpo/judge.jsonl data/dpo/gencut --exam data/dpo/cards.jsonl
  3b. 原答案也评一遍（只评要拿来顶的那些）：写成 data/dpo/ref-s0.txt 再 judge_exam → data/dpo/judge-ref.jsonl
  4. pairs：.venv/bin/python scripts/dpo_prep.py pairs → data/dpo/pairs.jsonl（{"prompt": [system, user], "chosen": 台词, "rejected": 台词, ...}）

2026-11-22 第二轮迭代 DPO：--dir data/dpo2 换一个目录；cards --all = 池里所有 v3 没见过的卡（含第一轮那 300 张，重新采样是 Iterative RPO 的做法），由 dpoa-s1 生成。
"""
import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from judge_train import kept  # noqa: E402
from render_card import render_card  # noqa: E402
from to_mlx import SYS_LINES, lines_text  # noqa: E402

D = ROOT / "data/dpo"


def cards(n, take_all=False):
    v3 = [json.loads(l) for l in (ROOT / "data/v3/mlx-v3/train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    seen = {c["messages"][1]["content"] for c in v3}
    pools = {}
    for d in ("batch-1", "batch-2", "batch-3"):
        pools[d] = [r for r in kept(d) if render_card(r["card"]) not in seen and (r["card"].get("goal") or {}).get("inner")]
    rnd = random.Random(20261121)
    want = {"batch-3": n // 2, "batch-1": n // 4, "batch-2": n - n // 2 - n // 4}
    if take_all:
        want = {d: len(v) for d, v in pools.items()}
    out = []
    for d, k in want.items():
        rows = pools[d][:]
        rnd.shuffle(rows)
        for r in rows[:k]:
            out.append({"id": f"dpo-{d.replace('batch-', 'b')}-{r['id']}", "card": r["card"], "ref": lines_text(r["output"])})
    D.mkdir(parents=True, exist_ok=True)
    (D / "cards.jsonl").write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in out) + "\n", encoding="utf-8")
    print(f"cards.jsonl：{len(out)} 张（{dict(Counter(c['id'].split('-')[1] for c in out))}），可用池 {({k: len(v) for k, v in pools.items()})}")


def good(j):   # 藏住且办成：没有任何硬伤，办成 = 2
    return j["pass"] and j["办成"] == 2


BAD_MODE = "target"


def bad(j):
    """target（定向版）= 心里话说出口 / 没办成；any（加宽版，11-21 加）= 任何硬伤都算（含语句不通或不合人设，练习卡上占 55%）。"""
    if BAD_MODE == "any":
        return not j["pass"]
    return "心里话说出口" in j["problems"] or "没办成" in j["problems"]


def pairs(use_ref, ref_judge=None, out_name="pairs.jsonl"):
    ok_ref = None
    if ref_judge and Path(ref_judge).exists():   # 原答案也用同一评审评过：只有全过的原答案才能当好例子（11-21：113 条里 11 条自己就说漏心里话）
        ok_ref = {json.loads(l)["id"] for l in Path(ref_judge).read_text(encoding="utf-8").splitlines() if l.strip() and json.loads(l)["pass"]}
    cards = {json.loads(l)["id"]: json.loads(l) for l in (D / "cards.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    by = defaultdict(list)
    for l in (D / "judge.jsonl").read_text(encoding="utf-8").splitlines():
        if l.strip():
            j = json.loads(l)
            by[j["id"]].append(j)
    rnd = random.Random(0)
    out, stat = [], Counter()
    for cid, js in by.items():
        g = [j for j in js if good(j)]
        b = [j for j in js if bad(j)]
        stat["卡"] += 1
        stat["有好"] += bool(g)
        stat["有坏"] += bool(b)
        if not b:
            continue
        ref_ok = use_ref and (ok_ref is None or cid in ok_ref)
        chosen = [(j["lines"], "model") for j in g] or ([(cards[cid]["ref"], "ref")] if ref_ok else [])
        if not chosen:
            stat["有坏没好（跳过）"] += 1
            continue
        for k, rj in enumerate(b):                     # 每个坏例子配一个好例子（轮流用）
            ch, src = chosen[k % len(chosen)]
            user = render_card(cards[cid]["card"])
            out.append({"id": cid, "prompt": [{"role": "system", "content": SYS_LINES}, {"role": "user", "content": user}],
                        "chosen": ch, "rejected": rj["lines"], "chosen_src": src,
                        "why": rj["problems"] if BAD_MODE == "any" else [p for p in rj["problems"] if p in ("心里话说出口", "没办成")]})
            stat["对（" + src + "）"] += 1
            for p in out[-1]["why"]:
                stat["坏：" + p] += 1
    rnd.shuffle(out)
    (D / out_name).write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in out) + "\n", encoding="utf-8")
    print(f"{out_name}：{len(out)} 对；{dict(stat)}")


def need_ref(ref_judge):
    cards = {json.loads(l)["id"]: json.loads(l) for l in (D / "cards.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    done = {json.loads(l)["id"] for l in Path(ref_judge).read_text(encoding="utf-8").splitlines() if l.strip()} if Path(ref_judge).exists() else set()
    by = defaultdict(list)
    for l in (D / "judge.jsonl").read_text(encoding="utf-8").splitlines():
        if l.strip():
            j = json.loads(l); by[j["id"]].append(j)
    todo = [cid for cid, js in by.items() if any(bad(j) for j in js) and not any(good(j) for j in js) and cid not in done]
    blk = []
    for cid in sorted(todo):
        blk.append(f"=== {cid}（{cards[cid]['card']['speaker']['name']}）"); blk += cards[cid]["ref"].split("\n")
    (D / "ref-todo-s0.txt").write_text("\n".join(blk) + "\n", encoding="utf-8")
    print(f"还要评的原答案 {len(todo)} 条 → ref-todo-s0.txt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cards", "pairs"])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--no-ref", action="store_true", help="没有好的生成时不拿原答案顶")
    ap.add_argument("--bad", choices=["target", "any"], default="target", help="target = 定向版；any = 加宽版")
    ap.add_argument("--out", default="pairs.jsonl")
    ap.add_argument("--need-ref", action="store_true", help="只列出这一版要拿原答案顶、但原答案还没评过的卡，写成 ref-todo.txt（judge_exam 能直接评）")
    ap.add_argument("--ref-judge", default=None, help="原答案的评审结果（默认 <dir>/judge-ref.jsonl）；有就只用全过的原答案")
    ap.add_argument("--dir", default=str(D), help="工作目录（默认 data/dpo；第二轮用 data/dpo2）")
    ap.add_argument("--all", action="store_true", help="cards：取池里所有 v3 没见过的卡，忽略 --n")
    a = ap.parse_args()
    BAD_MODE = a.bad
    D = Path(a.dir)
    a.ref_judge = a.ref_judge or str(D / "judge-ref.jsonl")
    if a.cmd == "cards":
        cards(a.n, a.all)
    elif a.need_ref:
        need_ref(a.ref_judge)
    else:
        pairs(not a.no_ref, a.ref_judge, a.out)
