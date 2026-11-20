"""考卷 v2 打分：多种子、按难度分档、加权总分、逐卡配对比较（只查格式，口气 / 目的 / 守秘要人看）。

用法（在仓库根目录跑）：
  .venv/bin/python scripts/exam_stats.py v1=data/x/v1-s0.txt,data/x/v1-s1.txt,data/x/v1-s2.txt v2a=...
  也可以写前缀：v1=data/x/v1   （自动找 data/x/v1-s*.txt）
  [--exam exam/考卷.jsonl] [--weights 简单=0.4,中等=0.4,困难=0.2]

输出：
1. 每个版本：各档合规率（张数 × 种子数里有几次全合规）、加权总分、95% 区间（按卡重抽样）、问题计数。
2. 第一个版本当基准，其他版本逐卡配对：同一张卡、同一组种子的合规率差 → 赢 / 输 / 平几张、
   加权差及 95% 区间。区间跨过 0 = 这次比不出差别。
3. 内容模式（content / all）另报「按考点」：每项问题只拿考这一项的卡当分母（exam/考点.json，
   2026-11-20 加，防「隐形缺陷」：弱版本某项问题少，可能只是没机会犯）；加 --mask 再报遮挡表。
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
from score_exam import check, fix_lines, load_rules, parse_output  # noqa: E402

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
    ap.add_argument("--mode", default="raw,fixed", help="raw = 原样查格式；fixed = 先按游戏代码把关修一遍再查（删括号动作、清单外表情）；"
                    "content = 只看内容盲评（要 --judge）；all = 修后格式 + 内容都过。逗号分隔可报多种")
    ap.add_argument("--judge", help="judge_exam.py 的结果文件（jsonl）")
    ap.add_argument("--probes", default=str(HERE.parent / "exam" / "考点.json"), help="考点标注；给空字符串就不报")
    ap.add_argument("--mask", action="store_true", help="内容模式下加报遮挡表")
    a = ap.parse_args()
    for mode in a.mode.split(","):
        report(a, mode.strip())


MODE_NAME = {"raw": "格式（原样）", "fixed": "格式（代码把关后）", "content": "内容（DeepSeek 盲评）", "all": "综合（把关后格式 + 内容）"}


def load_judge(path):
    out = {}
    if path:
        for t in Path(path).read_text(encoding="utf-8").splitlines():
            if t.strip():
                d = json.loads(t)
                out[(d["file"], d["id"])] = d
    return out


# 每项问题看哪些考点的卡（左边考点的卡当分母；多个就分开报）
PROBE_OF = [
    ("心里话说出口", ["心里话·要藏", "心里话·可说"]),
    ("说漏秘密", ["说漏秘密", "秘密·被追问"]),
    ("出戏", ["出戏"]),
    ("称呼或人物错", ["有旁人"]),
    ("口头禅滥用", ["口头禅"]),
    ("没办成", ["多件事"]),
]
MASKS = ["语句不通或不合人设", "没办成", "称呼或人物错"]
TARGETS = ["心里话说出口", "说漏秘密", "出戏", "口头禅滥用"]


def load_probes(path):
    if not path or not Path(path).exists():
        return {}
    return {i: set(v["考"]) for i, v in json.loads(Path(path).read_text(encoding="utf-8"))["卡"].items()}


def pct(n, d):
    return f"{n}/{d}（{100 * n / d:.0f}%）" if d else "0/0"


def probe_lines(sc, probes):
    out = []
    for prob, tags in PROBE_OF:
        parts = []
        for t in tags:
            js = [j for j in sc if t in probes.get(j["id"], ())]
            parts.append(f"{t} {pct(sum(prob in j['problems'] for j in js), len(js))}")
        out.append(f"{prob}：" + "，".join(parts))
    return out


def mask_lines(sc):
    """遮挡表：某项大毛病出现 / 没出现时，别的问题各占多少。两边差得多 = 可能被遮住或被带出来。"""
    out = []
    for m in MASKS:
        yes = [j for j in sc if m in j["problems"]]
        no = [j for j in sc if m not in j["problems"]]
        cells = [f"{t} 有 {pct(sum(t in j['problems'] for j in yes), len(yes))} / 没有 {pct(sum(t in j['problems'] for j in no), len(no))}"
                 for t in TARGETS]
        out.append(f"「{m}」：" + "；".join(cells))
    return out


def report(a, mode):

    weights = {k: float(v) for k, v in (x.split("=") for x in a.weights.split(","))}
    rules = load_rules(a.exam)
    tier_of = {}
    for t in Path(a.exam).read_text(encoding="utf-8").splitlines():
        if t.strip():
            d = json.loads(t)
            tier_of[d["id"]] = (d.get("meta") or {}).get("tier") or "中等"

    judge = load_judge(a.judge)
    probes = load_probes(a.probes)
    if mode in ("content", "all") and not judge:
        sys.exit("--mode content / all 要给 --judge")

    def verdict(path, i, lines):
        fmt = check(fix_lines(lines, rules[i]) if mode != "raw" else lines, rules[i])
        if mode in ("raw", "fixed"):
            return not fmt, fmt
        j = judge.get((Path(path).name, i))
        if j is None:
            return None
        con = j["problems"]
        if mode == "content":
            return j["pass"], con
        return (not fmt) and j["pass"], fmt + con

    versions = {}
    for item in a.runs:
        label, _, spec = item.partition("=")
        per_seed = {}
        for p in expand(spec):
            res = parse_output(p)
            got = {i: verdict(p, i, res[i]) for i in rules if i in res}
            per_seed[seed_of(p)] = {i: v for i, v in got.items() if v is not None}
        versions[label] = per_seed

    cards = sorted(set.intersection(*[set(s) for v in versions.values() for s in v.values()]))
    missing = len(rules) - len(cards)
    print(f"######## {MODE_NAME.get(mode, mode)}")
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
        if mode in ("content", "all"):
            paths = expand(dict(x.partition("=")[::2] for x in a.runs)[label])
            sc = [judge[(Path(p).name, c)] for p in paths for c in cards if (Path(p).name, c) in judge]
            like = [j["像不像"] for j in sc if isinstance(j.get("像不像"), (int, float))]
            got = Counter(j["办成"] for j in sc)
            if like:
                print(f"    像不像 平均 {sum(like) / len(like):.2f}（1–5）；办成 2 / 1 / 0：{got[2]} / {got[1]} / {got[0]}")
            if probes:
                print("    按考点：" + "；".join(probe_lines(sc, probes)))
            if a.mask:
                for t in mask_lines(sc):
                    print("    遮挡 " + t)

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
    print()


if __name__ == "__main__":
    main()
