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
2026-11-23 第三轮（data/dpo3，dpo2-s1 生成，评审标准 v4）：--bad v4 = 坏例子认「该藏的说出口 / 语句不通 / 不合人设 / 没办成」（主攻语句不通）；
  --max-per-card 1 = 每张卡最多 1 对（11-23 Ken 定，197 对，规模同第二轮）；--min-subtext 3 = 好例子还要潜台词分项 ≥ 3（实测练习卡只有 1.5% 够格，没用）（读 judge 记录的「分项」）。judge 记录没有 rule 字段的按 v1 处理（说出口叫「心里话说出口」）。
"""
import argparse
import json
import random
import re
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


MAX_PER_CARD = 0   # --max-per-card：每张卡最多几对（0 = 不限）；限了就在这张卡的坏例子、好例子里各随机挑（种子固定）
MIN_SUBTEXT = 0   # --min-subtext：好例子要求的潜台词分项下限（0 = 不看）


def good(j):   # 藏住且办成：没有任何硬伤，办成 = 2；可再要求潜台词 ≥ MIN_SUBTEXT
    if not (j["pass"] and j["办成"] == 2):
        return False
    if MIN_SUBTEXT:
        v = (j.get("分项") or {}).get("潜台词")
        return isinstance(v, int) and v >= MIN_SUBTEXT
    return True


BAD_MODE = "target"

SPK = re.compile(r"^([^:：\[\]（(]+)(\[[^\]]*\])?[:：]")


def speaks_for_others(card, lines):
    """台词里有别人开口的行（模型替对方说话，11-23 第三轮发现 2472 条里 148 条，评审漏判 3 条）。这种不能当好例子。"""
    me = card["speaker"]["name"]
    return any((m := SPK.match(ln)) and m.group(1).strip() != me for ln in lines.split("\n"))


def say_name(j):   # 各版评审标准里「说出口」硬伤的名字：v4 = 该藏的说出口（老定义另存在「心里话说出口_旧」，不进 problems）；没写 rule = v1
    return "该藏的说出口" if j.get("rule", "v1") == "v4" else "心里话说出口"


def why(j):
    """这条坏例子是因为哪些硬伤坏的（空 = 不是坏例子）。
    target（定向版）= 说出口 / 没办成；any（加宽版，11-21 加）= 任何硬伤；v4（第三轮，11-23 加）= 说出口 / 语句不通 / 不合人设 / 没办成。"""
    if BAD_MODE == "any":
        return list(j["problems"])
    want = {say_name(j), "没办成"} | ({"语句不通", "不合人设"} if BAD_MODE == "v4" else set())
    return [p for p in j["problems"] if p in want]


def bad(j):
    return bool(why(j))


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
        g2 = [j for j in g if not speaks_for_others(cards[cid]["card"], j["lines"])]
        stat["好例子替别人说话（去掉）"] += len(g) - len(g2)
        chosen = [(j["lines"], "model") for j in g2] or ([(cards[cid]["ref"], "ref")] if ref_ok else [])
        if not chosen:
            stat["有坏没好（跳过）"] += 1
            continue
        if MAX_PER_CARD:
            b = rnd.sample(b, min(MAX_PER_CARD, len(b)))
            chosen = rnd.sample(chosen, len(chosen))
        for k, rj in enumerate(b):                     # 每个坏例子配一个好例子（轮流用）
            ch, src = chosen[k % len(chosen)]
            user = render_card(cards[cid]["card"])
            out.append({"id": cid, "prompt": [{"role": "system", "content": SYS_LINES}, {"role": "user", "content": user}],
                        "chosen": ch, "rejected": rj["lines"], "chosen_src": src,
                        "why": why(rj)})
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
    ap.add_argument("--bad", choices=["target", "any", "v4"], default="target", help="target = 定向版；any = 加宽版；v4 = 第三轮（说出口 / 语句不通 / 不合人设 / 没办成）")
    ap.add_argument("--max-per-card", type=int, default=0, help="每张卡最多几对（0 = 不限；11-23 第三轮用 1）")
    ap.add_argument("--min-subtext", type=int, default=0, help="好例子还要潜台词分项 ≥ 这个数（v3 / v4 评审才有分项；0 = 不看）")
    ap.add_argument("--out", default="pairs.jsonl")
    ap.add_argument("--need-ref", action="store_true", help="只列出这一版要拿原答案顶、但原答案还没评过的卡，写成 ref-todo.txt（judge_exam 能直接评）")
    ap.add_argument("--ref-judge", default=None, help="原答案的评审结果（默认 <dir>/judge-ref.jsonl）；有就只用全过的原答案")
    ap.add_argument("--dir", default=str(D), help="工作目录（默认 data/dpo；第二轮用 data/dpo2）")
    ap.add_argument("--all", action="store_true", help="cards：取池里所有 v3 没见过的卡，忽略 --n")
    a = ap.parse_args()
    BAD_MODE = a.bad
    MIN_SUBTEXT = a.min_subtext
    MAX_PER_CARD = a.max_per_card
    D = Path(a.dir)
    a.ref_judge = a.ref_judge or str(D / "judge-ref.jsonl")
    if a.cmd == "cards":
        cards(a.n, a.all)
    elif a.need_ref:
        need_ref(a.ref_judge)
    else:
        pairs(not a.no_ref, a.ref_judge, a.out)
