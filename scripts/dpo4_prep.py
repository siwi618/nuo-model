"""第四轮迭代 DPO 偏好对（2026-11-24，Ken 定：a 长度配平 + b 好例子混 DeepSeek 原答案）。

不重新生成：用第三轮 dpo2-s1 对 618 张练习卡各写的 4 条（data/dpo3/gencut-s*.txt，v4 已评）+ 每张卡的 DeepSeek 原答案。
挑对的信号（Ken 11-24 同意）：
  - 长度门槛（代码）：一对里两条字数差 / 较长那条 < --gate（默认 0.2；字数去说话人、去表情括号，同 card_len.clen）。
  - 硬条件（好例子必须没有）：该藏的说出口 / 说漏秘密 / 没办成（办成 0）/ 称呼或人物错 / 出戏（评审 v5）；
    替别人说话（代码）；卡外编造且不是卡缺（invent_check，Ken 定的 ①②③）。
  - 同卡比 v5「每句有毛病率」（毛病句数 / 句数，不分轻重）：坏例子要更高；或者坏例子自己有硬伤（说出口 / 说漏 / 没办成）。
  - 像不像、办成：坏例子没有硬伤时，好例子都不能更低（办成 1 的原答案很多，试跑 10 张卡 4 张是 1，所以不硬要 2）。
分步花钱（Ken 选「分步评」）：
  1. cands（免费）：用 v4 结果配候选。好例子候选 = v4 全过且办成 2 的模型台词（最多 2 条）+ 原答案；
     每个好例子候选配字数够近、v4 没全过的模型台词当坏例子候选（最多 2 条，v4 硬伤的排前面）。→ data/dpo4/cands.jsonl
  2. invent：所有好例子候选查卡外编造。→ data/dpo4/invent.jsonl
  3. judge：每张卡按顺序取第一个没编造的好例子候选，v5 评；硬条件过了再评它的坏例子候选，不过就换下一个。→ data/dpo4/judge5.jsonl
  4. pairs：每卡最多 1 对。→ data/dpo4/pairs.jsonl（格式同第三轮）
用法（仓库根目录）：.venv/bin/python scripts/dpo4_prep.py cands|invent|judge|pairs [--limit N]
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
from card_len import clen  # noqa: E402
from dpo_prep import speaks_for_others  # noqa: E402
from render_card import render_card  # noqa: E402
from to_mlx import SYS_LINES  # noqa: E402

SRC = ROOT / "data/dpo3"
D = ROOT / "data/dpo4"
HARD_V4 = {"该藏的说出口", "说漏秘密", "没办成"}
HARD_GOOD = {"该藏的说出口", "说漏秘密", "没办成", "称呼或人物错", "出戏"}   # 好例子不能有
HARD_BAD = {"该藏的说出口", "说漏秘密", "没办成"}                         # 坏例子有了就直接算坏
MAX_MODEL_GOOD = 2
MAX_BAD = 2


def jl(p):
    return [json.loads(t) for t in Path(p).read_text(encoding="utf-8").splitlines() if t.strip()]


def near(a, b, gate):
    la, lb = clen(a), clen(b)
    return max(la, lb) > 0 and abs(la - lb) / max(la, lb) < gate


def load_cards():
    return {c["id"]: c for c in jl(SRC / "cards.jsonl")}


def cands(gate):
    cards = load_cards()
    by = defaultdict(list)
    for j in jl(SRC / "judge.jsonl"):
        by[j["id"]].append(j)
    rnd = random.Random(20261124)
    out, st = [], Counter()
    for cid in sorted(by):
        js, card = by[cid], cards[cid]["card"]
        bads = [j for j in js if not j["pass"]]
        rnd.shuffle(bads)
        bads.sort(key=lambda j: not (set(j["problems"]) & HARD_V4))   # v4 硬伤的排前面（稳定排序，其余随机）
        goods = [(j["lines"], "model") for j in js if j["pass"] and j["办成"] == 2 and not speaks_for_others(card, j["lines"])]
        rnd.shuffle(goods)
        goods = goods[:MAX_MODEL_GOOD]
        if not speaks_for_others(card, cards[cid]["ref"]):
            goods.append((cards[cid]["ref"], "ref"))
        cl = []
        for g, src in goods:
            rej = list(dict.fromkeys(b["lines"] for b in bads if b["lines"] != g and near(g, b["lines"], gate)))[:MAX_BAD]
            if rej:
                cl.append({"src": src, "lines": g, "rej": rej})
        st["卡"] += 1
        if cl:
            st["有候选的卡"] += 1
            st["第一候选=" + cl[0]["src"]] += 1
            out.append({"id": cid, "cands": cl})
    D.mkdir(parents=True, exist_ok=True)
    (D / "cands.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    ng = sum(len(r["cands"]) for r in out)
    print(f"cands.jsonl：{dict(st)}；好例子候选 {ng} 条（要查编造），每卡第一候选 + 坏例子约 {sum(1 + len(r['cands'][0]['rej']) for r in out)} 条（v5 至少要评）")


class Cache:
    """按（卡, 台词）存结果的 jsonl，断了重跑会跳过已做的。"""

    def __init__(self, path):
        self.path, self.lock, self.d = path, threading.Lock(), {}
        if path.exists():
            for r in jl(path):
                self.d[(r["id"], r["lines"])] = r
        self.use = {"in": 0, "out": 0}

    def get(self, cid, lines, fn):
        if (cid, lines) in self.d:
            return self.d[(cid, lines)]
        r, u = fn()
        r = {"id": cid, **r, "lines": lines}
        with self.lock:
            self.d[(cid, lines)] = r
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            self.use["in"] += u.get("prompt_tokens", 0)
            self.use["out"] += u.get("completion_tokens", 0)
        return r


def invented(r):   # 卡外编造（不算卡缺）
    return any(not e.get("卡缺") for e in r.get("编造", []))


def run_invent(limit, workers):
    from invent_check import check_one
    cards = load_cards()
    C = Cache(D / "invent.jsonl")
    todo = [(r["id"], c["lines"]) for r in jl(D / "cands.jsonl")[:limit] for c in r["cands"]]

    def one(t):
        cid, lines = t
        return C.get(cid, lines, lambda: (lambda e, u: ({"编造": e}, u))(*check_one(render_card(cards[cid]["card"]), lines.split("\n"))))
    _pool(one, todo, workers, "查编造")
    print(f"token 入 {C.use['in']} 出 {C.use['out']}")
    R = [C.d[t] for t in todo if t in C.d]
    print(f"好例子候选 {len(R)} 条：有编造（不算卡缺）{sum(invented(r) for r in R)} 条；按来源 "
          f"{dict(Counter((r['lines'] == cards[r['id']]['ref'] and 'ref' or 'model', invented(r)) for r in R))}")


def hard_ok(j):   # 办成 1 也行（原答案常被判「推进了、对方还没应」= 1）；成对时再要求不比坏例子低
    return j["办成"] >= 1 and not (set(j["problems"]) & HARD_GOOD)


def rate(j):
    return j["毛病句数"] / j["句数"] if j["句数"] else 1.0


def verdict(g, b):
    """好例子 g、坏例子 b（v5 记录）能不能成对；能就返回理由，不能返回 None。"""
    hb = sorted(set(b["problems"]) & HARD_BAD)
    if hb:
        return hb
    if g["像不像"] is None or b["像不像"] is None or g["像不像"] < b["像不像"] or g["办成"] < b["办成"]:
        return None
    if rate(b) > rate(g):
        return [f"每句毛病 {rate(b):.2f}>{rate(g):.2f}"]
    return None


def run_judge(limit, workers):
    from judge_exam import judge_one
    cards = load_cards()
    I = Cache(D / "invent.jsonl")
    J = Cache(D / "judge5.jsonl")

    def j5(cid, lines):
        return J.get(cid, lines, lambda: judge_one(cards[cid]["card"], lines.split("\n"), "v5"))

    def one(r):
        cid = r["id"]
        for c in r["cands"]:
            inv = I.d.get((cid, c["lines"]))
            if inv is None or invented(inv):
                continue
            if not hard_ok(j5(cid, c["lines"])):
                continue
            for b in c["rej"]:
                j5(cid, b)
            return
    _pool(one, jl(D / "cands.jsonl")[:limit], workers, "v5 评卡")
    print(f"本次 token 入 {J.use['in']} 出 {J.use['out']}；judge5.jsonl 共 {len(J.d)} 条")


def _pool(fn, items, workers, what):
    n, lock = [0], threading.Lock()

    def wrap(x):
        try:
            fn(x)
        except Exception as e:  # 单条失败不拖垮整批，重跑会补
            print("失败：", e, flush=True)
        with lock:
            n[0] += 1
            if n[0] % 50 == 0:
                print(f"  {what} {n[0]}/{len(items)}", flush=True)
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(wrap, items))


def pairs():
    cards = load_cards()
    I = Cache(D / "invent.jsonl").d
    J = Cache(D / "judge5.jsonl").d
    out, st = [], Counter()
    for r in jl(D / "cands.jsonl"):
        cid = r["id"]
        st["卡"] += 1
        got = None
        for c in r["cands"]:
            inv = I.get((cid, c["lines"]))
            if inv is None:
                st["没查编造"] += 1
                continue
            if invented(inv):
                st[f"好例子候选有编造（{c['src']}）"] += 1
                continue
            g = J.get((cid, c["lines"]))
            if g is None:
                st["没评 v5"] += 1
                continue
            if not hard_ok(g):
                st[f"好例子候选 v5 硬伤（{c['src']}）"] += 1
                continue
            vs = [(verdict(g, J[(cid, b)]), J[(cid, b)]) for b in c["rej"] if (cid, b) in J]
            vs = [(w, b) for w, b in vs if w]
            if not vs:
                st[f"坏例子不比它差（{c['src']}）"] += 1
                continue
            vs.sort(key=lambda wb: (not (set(wb[1]["problems"]) & HARD_BAD), -(rate(wb[1]) - rate(g))))
            got = (c["src"], g, vs[0])
            break
        if not got:
            continue
        src, g, (why, b) = got
        out.append({"id": cid, "prompt": [{"role": "system", "content": SYS_LINES}, {"role": "user", "content": render_card(cards[cid]["card"])}],
                    "chosen": g["lines"], "rejected": b["lines"], "chosen_src": src, "why": why,
                    "len": [clen(g["lines"]), clen(b["lines"])], "rate": [round(rate(g), 3), round(rate(b), 3)],
                    "like": [g["像不像"], b["像不像"]]})
    random.Random(0).shuffle(out)
    (D / "pairs.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in out), encoding="utf-8")
    n = len(out) or 1
    mean = lambda xs: sum(xs) / max(1, len(xs))  # noqa: E731
    print(f"pairs.jsonl：{len(out)} 对；{dict(st)}")
    print(f"  好例子来源：{dict(Counter(p['chosen_src'] for p in out))}")
    print(f"  字数：好 {mean([p['len'][0] for p in out]):.1f} / 坏 {mean([p['len'][1] for p in out]):.1f}；"
          f"好例子更短 {sum(p['len'][0] < p['len'][1] for p in out) / n:.0%}、一样长 {sum(p['len'][0] == p['len'][1] for p in out) / n:.0%}")
    print(f"  每句毛病率：好 {mean([p['rate'][0] for p in out]):.2f} / 坏 {mean([p['rate'][1] for p in out]):.2f}；"
          f"像不像：好 {mean([p['like'][0] for p in out]):.1f} / 坏 {mean([p['like'][1] for p in out]):.1f}")
    print(f"  坏在哪：{dict(Counter(w if not w.startswith('每句') else '只是每句毛病更多' for p in out for w in p['why']))}")
    for src in ("model", "ref"):
        ps = [p for p in out if p["chosen_src"] == src]
        if ps:
            print(f"  {src}：{len(ps)} 对，好 {mean([p['len'][0] for p in ps]):.1f} 字 / 坏 {mean([p['len'][1] for p in ps]):.1f} 字，好例子更短 {sum(p['len'][0] < p['len'][1] for p in ps) / len(ps):.0%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cands", "invent", "judge", "pairs"])
    ap.add_argument("--gate", type=float, default=0.2, help="字数差 / 较长那条的上限（Ken 11-24 定 0.2）")
    ap.add_argument("--limit", type=int, default=None, help="只做前 N 张卡（试跑量 token 用）")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.cmd == "cands":
        cands(a.gate)
    elif a.cmd == "invent":
        run_invent(a.limit, a.workers)
    elif a.cmd == "judge":
        run_judge(a.limit, a.workers)
    else:
        pairs()
