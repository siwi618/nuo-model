"""查评审（v4）是不是嫌长（2026-11-24，第三轮 DPO 盲测后：Ken 觉得 s2 并不比 dpo2-s1 顺，评审却说语句不通少了 15 个百分点）。

控制实验，都用公开卷、v4 已评过的、至少 3 句的台词：
- U 组（评审判了语句不通，80 条）：理由里「」引用的、在台词里原样出现的片段 = 毛病。
  · full：原台词一字不改再判一次（量评审自己的随机波动）；
  · keep：只留含引文的句子（毛病还在、台词变短；删掉的句子都不含引文）。含引文的句子 = 全部句子的跳过。
  删中间句子可能让台词接不上，只会让 keep 更容易判不通 → 如果 keep 反而更常被判通，结论只会更保守。
- T 组（评审判通的，40 条）：full 再判一次；half 只留前一半句子（向上取整）。
生成 run_exam 格式的文件（同一文件里一张卡只能出现一次，所以分成多组文件），评审结果存 data/lenbias/judge.jsonl。

用法（仓库根目录）：
  python3 scripts/lenbias_build.py data/lenbias
  .venv/bin/python -u scripts/judge_exam.py data/lenbias/judge.jsonl data/lenbias/*.txt --rule v4
  python3 scripts/lenbias_build.py data/lenbias --score
"""
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "cloud-v3" / "judge-r4.jsonl"
QUOTE = re.compile(r"「([^」]{4,})」")


def speaker(r):
    m = re.match(r"([^:：\[]+)", r["lines"].strip())
    return m.group(1).strip() if m else "?"


def build(out):
    rnd = random.Random(20261124)
    recs = []
    for t in SRC.read_text(encoding="utf-8").splitlines():
        r = json.loads(t)
        ls = [x for x in r["lines"].strip().splitlines() if x.strip()]
        if r.get("rule") == "v4" and len(ls) >= 3:
            r["ls"] = ls
            recs.append(r)
    rnd.shuffle(recs)
    samples = []
    for r in recs:
        if sum(s["grp"] == "U" for s in samples) >= 80:
            break
        if "语句不通" not in r["problems"]:
            continue
        quotes = [q for q in QUOTE.findall(r.get("理由", "")) if q in r["lines"]]
        keep = [x for x in r["ls"] if any(q in x for q in quotes)]
        if not keep or len(keep) == len(r["ls"]):
            continue
        samples.append({"grp": "U", "src": r, "full": r["ls"], "short": keep, "quotes": quotes})
    for r in recs:
        if sum(s["grp"] == "T" for s in samples) >= 40:
            break
        if r["problems"]:
            continue
        samples.append({"grp": "T", "src": r, "full": r["ls"], "short": r["ls"][:(len(r["ls"]) + 1) // 2], "quotes": []})
    # 分组写文件：同一文件里卡不重复
    groups = []
    for s in samples:
        for g in groups:
            if s["src"]["id"] not in g:
                g[s["src"]["id"]] = s
                break
        else:
            groups.append({s["src"]["id"]: s})
    out.mkdir(parents=True, exist_ok=True)
    meta = []
    for k, g in enumerate(groups):
        for var in ("full", "short"):
            fn = out / f"{var}-g{k:02d}.txt"
            fn.write_text("".join(f"=== {cid}（{speaker(s['src'])}）\n" + "\n".join(s[var]) + "\n" for cid, s in g.items()), encoding="utf-8")
        for cid, s in g.items():
            meta.append({"grp": s["grp"], "card": cid, "g": k, "src_file": s["src"]["file"], "quotes": s["quotes"],
                         "len_full": len("".join(s["full"])), "len_short": len("".join(s["short"])),
                         "n_full": len(s["full"]), "n_short": len(s["short"])})
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"U {sum(m['grp'] == 'U' for m in meta)} 条、T {sum(m['grp'] == 'T' for m in meta)} 条，{len(groups)} 组文件 → {out}")


def score(out):
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    j = {}
    for t in (out / "judge.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(t)
        j[(r["file"], r["id"])] = r
    res = defaultdict(list)
    for m in meta:
        f = j.get((f"full-g{m['g']:02d}.txt", m["card"]))
        s = j.get((f"short-g{m['g']:02d}.txt", m["card"]))
        if f and s:
            res[m["grp"]].append((m, "语句不通" in f["problems"], "语句不通" in s["problems"]))
    for g, rows in res.items():
        n = len(rows)
        lf = sum(m["len_full"] for m, _, _ in rows) / n
        ls = sum(m["len_short"] for m, _, _ in rows) / n
        bf = sum(a for _, a, _ in rows)
        bs = sum(b for _, _, b in rows)
        both = sum(a and b for _, a, b in rows)
        only_f = sum(a and not b for _, a, b in rows)
        only_s = sum(b and not a for _, a, b in rows)
        lab = "U（原判不通；short = 只留含引文的句子）" if g == "U" else "T（原判通；short = 前一半句子）"
        print(f"\n{lab}：{n} 条，平均字数 full {lf:.0f} → short {ls:.0f}")
        print(f"  再判语句不通：full {bf}/{n}（{bf / n:.0%}），short {bs}/{n}（{bs / n:.0%}）")
        print(f"  配对：两版都不通 {both}，只有 full 不通 {only_f}，只有 short 不通 {only_s}")
        if only_f + only_s:
            from math import comb
            k, N = max(only_f, only_s), only_f + only_s
            p = min(1, 2 * sum(comb(N, i) for i in range(k, N + 1)) / 2 ** N)
            print(f"  符号检验（双侧）p ≈ {p:.3f}")


if __name__ == "__main__":
    o = Path(sys.argv[1])
    score(o) if "--score" in sys.argv else build(o)
