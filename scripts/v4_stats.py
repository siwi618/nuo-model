"""评审标准 v4 的统计（2026-11-23）。exam_stats / sp_stats 只认 v1 的硬伤名，v4 先用这个。

每个版本报：说出口（新：该藏的说出口；旧：心里话说出口_旧）、全过（新 / 旧）、没办成、语句不通、不合人设、像不像（0–40）、潜台词（0–4）；
再按「同卡同种子」配对比两个版本（按卡整组重抽样 2000 次给 95% 区间）。
文件名里 cut-pen 前面的是版本标签；带 r 的是新采样种子（如 c3r = v3 的新种子），合并到同一个版本里。

用法（仓库根目录）：
  .venv/bin/python scripts/v4_stats.py data/cloud-v3/judge-r4.jsonl [--v3 data/cloud-v3/judge-r3.jsonl] [--base c3 --vs cdpo2s1,cdpoas1]
"""
import argparse
import json
import random
import re
from collections import defaultdict


def version_of(fname):
    lab = re.sub(r"cut-pen-s\d+\.txt$", "", fname).replace("sp9-", "")
    return lab[:-1] if lab.endswith("r") else lab


def seed_of(fname):
    return int(re.search(r"-s(\d+)\.txt$", fname).group(1))


METRICS = [
    ("说出口·新", lambda r: "该藏的说出口" in r["problems"]),
    ("说出口·旧", lambda r: r.get("心里话说出口_旧", "心里话说出口" in r["problems"])),
    ("全过·新", lambda r: r["pass"]),
    ("全过·旧", lambda r: r.get("pass_旧", r["pass"])),
    ("没办成", lambda r: "没办成" in r["problems"]),
    ("语句不通", lambda r: "语句不通" in r["problems"]),
    ("不合人设", lambda r: "不合人设" in r["problems"]),
]


def num(r, k):
    if k == "像不像":
        return r.get("像不像") if isinstance(r.get("像不像"), (int, float)) else None
    v = (r.get("分项") or {}).get(k)
    return v if isinstance(v, int) else None


def table(recs, title):
    by = defaultdict(list)
    for r in recs:
        by[version_of(r["file"])].append(r)
    print(f"\n{title}")
    head = ["版本", "条数"] + [m for m, _ in METRICS] + ["像不像/40", "潜台词/4"]
    print("  " + " | ".join(head))
    for v in sorted(by):
        rs = by[v]
        cells = [v, str(len(rs))] + [f"{100 * sum(f(r) for r in rs) / len(rs):.1f}%" for _, f in METRICS]
        for k in ("像不像", "潜台词"):
            xs = [num(r, k) for r in rs if num(r, k) is not None]
            cells.append(f"{sum(xs) / len(xs):.2f}" if xs else "—")
        print("  " + " | ".join(cells))


def paired(recs, base, other, title, n_boot=2000):
    idx = defaultdict(dict)   # (seed, card) -> {version: rec}
    for r in recs:
        idx[(seed_of(r["file"]), r["id"])][version_of(r["file"])] = r
    pairs = [(d[base], d[other], k[1]) for k, d in idx.items() if base in d and other in d]
    if not pairs:
        return
    cards = sorted({c for _, _, c in pairs})
    bycard = defaultdict(list)
    for a, b, c in pairs:
        bycard[c].append((a, b))
    rnd = random.Random(0)
    print(f"\n{title}：{other} 对 {base}（{len(pairs)} 对，{len(cards)} 张卡；差 = {other} − {base}，[95% 区间]）")
    tests = [(m, (lambda f: lambda r: 100.0 * f(r))(f)) for m, f in METRICS] + \
            [("像不像/40", lambda r: num(r, "像不像")), ("潜台词/4", lambda r: num(r, "潜台词"))]
    for m, g in tests:
        def diff(sample):
            xs = [g(b) - g(a) for a, b in sample if g(a) is not None and g(b) is not None]
            return sum(xs) / len(xs) if xs else 0.0
        d0 = diff([(x, y) for x, y, _ in pairs])
        boots = []
        for _ in range(n_boot):
            s = []
            for c in rnd.choices(cards, k=len(cards)):
                s += bycard[c]
            boots.append(diff(s))
        boots.sort()
        lo, hi = boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]
        mark = "  *" if lo > 0 or hi < 0 else ""
        unit = "" if "/" in m else " 个百分点"
        print(f"  {m}：{d0:+.2f}{unit} [{lo:+.2f} ~ {hi:+.2f}]{mark}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("judge", nargs="+")
    ap.add_argument("--v3", nargs="*", default=[], help="同样条目的 v3 评审结果，用来对照语句不通 / 说出口 / 像不像")
    ap.add_argument("--base", default="c3")
    ap.add_argument("--vs", default="cdpo2s1,cdpoas1")
    a = ap.parse_args()
    for path in a.judge:
        recs = [json.loads(t) for t in open(path, encoding="utf-8") if t.strip()]
        recs = [r for r in recs if r.get("rule") == "v4"]
        table(recs, f"== {path}（v4，{len(recs)} 条）")
        for o in a.vs.split(","):
            paired(recs, a.base, o, "配对")
    for path in a.v3:
        recs = [json.loads(t) for t in open(path, encoding="utf-8") if t.strip()]
        recs = [r for r in recs if r.get("rule") == "v3"]
        for r in recs:   # v3 没有新定义：两栏都用老的
            r["心里话说出口_旧"] = "心里话说出口" in r["problems"]
            r["problems"] = r["problems"] + (["该藏的说出口"] if r["心里话说出口_旧"] else [])
        table(recs, f"== 对照 {path}（v3，{len(recs)} 条；说出口·新 / 全过·新两栏用的也是老定义）")


if __name__ == "__main__":
    main()
