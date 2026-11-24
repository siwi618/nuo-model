"""按卡看台词长度（2026-11-24，第三轮 DPO 台词变短后查「该说长的卡」有没有受伤）。

每张卡：基准版本的平均字数（去说话人、去表情括号），各版本平均字数、办成=2 比例、没办成、语句不通、
评审理由里提到「简略 / 太短 / 没展开 / 敷衍 …」的条数。
「长卡」= 基准版本在这张卡上平均 ≥ --long 字（默认 50）。长卡 / 其余卡分开汇总，再列缩得最多的卡。

用法（仓库根目录）：
  python3 scripts/card_len.py data/cloud-v3/judge-r4.jsonl [data/cloud-v3/judge-sp9-r4.jsonl] --base cdpo2s1 --vs cdpo3s1,cdpo3s2
"""
import argparse
import json
import re
import statistics as st
from collections import defaultdict

SHORT_WORDS = re.compile(r"简略|太短|过短|没展开|未展开|敷衍|单薄|一句带过|草草|不够具体|太简单|过于简单")


def version_of(fname):
    lab = re.sub(r"cut-pen-s\d+\.txt$", "", fname).replace("sp9-", "")
    return lab[:-1] if lab.endswith("r") else lab


def clen(t):
    t = re.sub(r"^[^:：\n]*[:：]\s*", "", t, flags=re.M)
    t = re.sub(r"[\[（(【][^\]）)】]*[\]）)】]", "", t)
    return len(re.sub(r"\s", "", t))


def summ(rs):
    n = len(rs)
    return {"n": n, "len": st.mean(clen(r["lines"]) for r in rs),
            "ok2": sum(r.get("办成") == 2 for r in rs) / n,
            "fail": sum("没办成" in r["problems"] for r in rs) / n,
            "budong": sum("语句不通" in r["problems"] for r in rs) / n,
            "short": sum(bool(SHORT_WORDS.search(r.get("理由", ""))) for r in rs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--base", default="cdpo2s1")
    ap.add_argument("--vs", default="cdpo3s1,cdpo3s2")
    ap.add_argument("--long", type=float, default=50)
    ap.add_argument("--top", type=int, default=8)
    a = ap.parse_args()
    vers = [a.base] + a.vs.split(",")
    for path in a.files:
        cell = defaultdict(list)                      # (版本, 卡) -> 记录
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            v = version_of(r["file"])
            if v in vers:
                cell[(v, r["id"])].append(r)
        cards = sorted({c for v, c in cell if v == a.base and all((w, c) in cell for w in vers)})
        base_len = {c: summ(cell[(a.base, c)])["len"] for c in cards}
        groups = {"长卡": [c for c in cards if base_len[c] >= a.long], "其余卡": [c for c in cards if base_len[c] < a.long]}
        print(f"\n== {path}（{len(cards)} 张卡；长卡 = {a.base} 平均 ≥ {a.long:g} 字）")
        print("  分组 | 版本 | 卡数 | 条数 | 平均字数 | 办成=2 | 没办成 | 语句不通 | 理由说太简略")
        for g, cs in groups.items():
            if not cs:
                continue
            for v in vers:
                s = summ([r for c in cs for r in cell[(v, c)]])
                print(f"  {g} | {v} | {len(cs)} | {s['n']} | {s['len']:.1f} | {s['ok2']:.0%} | {s['fail']:.1%} | {s['budong']:.0%} | {s['short']}")
        # 缩得最多的卡：各 vs 版本平均
        rows = []
        for c in cards:
            b = summ(cell[(a.base, c)])
            vs = [summ(cell[(w, c)]) for w in vers[1:]]
            rows.append((st.mean(x["len"] for x in vs) - b["len"], c, b, vs))
        rows.sort()
        print(f"  缩得最多的 {a.top} 张卡（字数 / 办成=2 / 没办成条数，{a.base} → {' , '.join(vers[1:])}）")
        for d, c, b, vs in rows[:a.top]:
            vtxt = " , ".join(f"{x['len']:.0f}字 {x['ok2']:.0%} {round(x['fail'] * x['n'])}" for x in vs)
            print(f"    {c}  {b['len']:.0f}字 {b['ok2']:.0%} {round(b['fail'] * b['n'])} → {vtxt}  （差 {d:+.0f}）")


if __name__ == "__main__":
    main()
