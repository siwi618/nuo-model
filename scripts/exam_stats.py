"""考卷 v2 打分：多种子、按难度分档、加权总分、逐卡配对比较（只查格式，口气 / 目的 / 守秘要人看）。

用法（在仓库根目录跑）：
  .venv/bin/python scripts/exam_stats.py v1=data/x/v1-s0.txt,data/x/v1-s1.txt,data/x/v1-s2.txt v2a=...
  也可以写前缀：v1=data/x/v1   （自动找 data/x/v1-s*.txt）
  [--exam exam/考卷.jsonl] [--weights 简单=0.4,中等=0.4,困难=0.2]

输出：
1. 每个版本：各档合规率（张数 × 种子数里有几次全合规）、加权总分、95% 区间（按卡重抽样）、问题计数。
2. 第一个版本当基准，其他版本逐卡配对：同一张卡、同一组种子的合规率差 → 赢 / 输 / 平几张、
   加权差及 95% 区间。区间跨过 0 = 这次比不出差别。
种子要对齐（同名 -sN 对同名 -sN），版本之间才算配对。
"""

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from score_exam import check, load_rules, parse_output  # noqa: E402

TIERS = ("简单", "中等", "困难")


def expand(spec):
    paths = [p for p in spec.split(",") if p]
    if len(paths) == 1 and not Path(paths[0]).exists():
        p = Path(paths[0])
        paths = sorted(str(x) for x in p.parent.glob(p.name + "-s*.txt"))
    if not paths:
        sys.exit(f"找不到输出文件：{spec}")
    return paths


def seed_of(path):
    m = re.search(r"-s(\d+)\.txt$", path)
    return int(m.group(1)) if m else path


def weighted(card_rate, tier_of, weights, cards):
    tot = 0.0
    for t, w in weights.items():
        cs = [c for c in cards if tier_of[c] == t]
        if cs:
            tot += w * sum(card_rate[c] for c in cs) / len(cs)
    return tot / sum(w for t, w in weights.items() if any(tier_of[c] == t for c in cards))


def boot(fn, cards, tier_of, n=2000, seed=0):
    """按档分层、按卡重抽样，给 95% 区间。"""
    rng = random.Random(seed)
    by = {t: [c for c in cards if tier_of[c] == t] for t in TIERS}
    vals = []
    for _ in range(n):
        sample = [rng.choice(by[t]) for t in TIERS for _ in by[t]]
        vals.append(fn(sample))
    vals.sort()
    return vals[int(0.025 * n)], vals[int(0.975 * n) - 1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="标签=文件1,文件2,... 或 标签=输出前缀")
    ap.add_argument("--exam", default=str(HERE.parent / "exam" / "考卷.jsonl"))
    ap.add_argument("--weights", default="简单=0.4,中等=0.4,困难=0.2")
    a = ap.parse_args()

    weights = {k: float(v) for k, v in (x.split("=") for x in a.weights.split(","))}
    rules = load_rules(a.exam)
    tier_of = {}
    for t in Path(a.exam).read_text(encoding="utf-8").splitlines():
        if t.strip():
            d = json.loads(t)
            tier_of[d["id"]] = (d.get("meta") or {}).get("tier") or "中等"

    versions = {}
    for item in a.runs:
        label, _, spec = item.partition("=")
        per_seed = {}
        for p in expand(spec):
            res = parse_output(p)
            per_seed[seed_of(p)] = {i: (not check(res[i], rules[i]), check(res[i], rules[i]))
                                    for i in rules if i in res}
        versions[label] = per_seed

    cards = sorted(set.intersection(*[set(s) for v in versions.values() for s in v.values()]))
    missing = len(rules) - len(cards)
    print(f"考卷 {len(rules)} 张，共同考过 {len(cards)} 张" + (f"（{missing} 张缺，不算）" if missing else "")
          + "　档：" + "　".join(f"{t} {sum(tier_of[c] == t for c in cards)}" for t in TIERS)
          + f"　权重 {a.weights}\n")

    rates = {}
    for label, per_seed in versions.items():
        rate = {c: sum(s[c][0] for s in per_seed.values()) / len(per_seed) for c in cards}
        rates[label] = rate
        tier_txt = []
        for t in TIERS:
            cs = [c for c in cards if tier_of[c] == t]
            if cs:
                ok = sum(rate[c] for c in cs) * len(per_seed)
                tier_txt.append(f"{t} {ok:.0f}/{len(cs) * len(per_seed)}（{100 * ok / len(cs) / len(per_seed):.0f}%）")
        w = weighted(rate, tier_of, weights, cards)
        lo, hi = boot(lambda cs: weighted(rate, tier_of, weights, cs), cards, tier_of)
        probs = Counter(p.split(":")[0] for s in per_seed.values() for c in cards for p in s[c][1])
        print(f"{label}（{len(per_seed)} 个种子）: 加权 {100 * w:.1f}%  [95% {100 * lo:.0f}–{100 * hi:.0f}]  "
              + "  ".join(tier_txt))
        print(f"    问题 {dict(probs.most_common())}")

    labels = list(versions)
    if len(labels) > 1:
        base = labels[0]
        print(f"\n逐卡配对（和 {base} 比；同卡同种子）：")
        for label in labels[1:]:
            if set(versions[label]) != set(versions[base]):
                print(f"  {label}: 种子和 {base} 不一致，不能配对，跳过")
                continue
            d = {c: rates[label][c] - rates[base][c] for c in cards}
            win, lose = sum(v > 0 for v in d.values()), sum(v < 0 for v in d.values())
            diff = weighted(rates[label], tier_of, weights, cards) - weighted(rates[base], tier_of, weights, cards)
            lo, hi = boot(lambda cs: weighted(rates[label], tier_of, weights, cs)
                          - weighted(rates[base], tier_of, weights, cs), cards, tier_of)
            verdict = "比不出差别" if lo <= 0 <= hi else ("更好" if diff > 0 else "更差")
            per_tier = "  ".join(
                f"{t} {100 * sum(d[c] for c in cards if tier_of[c] == t) / max(1, sum(tier_of[c] == t for c in cards)):+.0f}"
                for t in TIERS if any(tier_of[c] == t for c in cards))
            print(f"  {label}: 加权 {100 * diff:+.1f} 个百分点 [95% {100 * lo:+.0f} ~ {100 * hi:+.0f}] → {verdict}；"
                  f"赢 {win} 张 / 输 {lose} 张 / 平 {len(cards) - win - lose} 张；各档 {per_tier}")


if __name__ == "__main__":
    main()
