"""评审 v5.1 人工核对打分（2026-11-24）：Ken 的逐句标注 vs 评审 v5.1（同一批 40 条），再和 v4 的整条判断比。

用法：.venv/bin/python scripts/v5_check_score.py data/human-check/v5 Ken.txt（结果码那一行存成的文件，放在目录里）
句子标签：0 = 没问题，1 = 轻，2 = 重（评审一句有几处毛病时取最重的）。
"""
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TYPES = ["语句不通", "用词难懂", "逻辑不对", "不合人设", "其他"]
SENT_TYPES = {"语句不通", "用词难懂", "逻辑不对", "不合人设"}   # Ken 页面上能选的；评审其余类型（说漏 / 说出口等）单列


def kappa(pairs):
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n / n
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main():
    d = Path(sys.argv[1])
    code = (d / sys.argv[2]).read_text(encoding="utf-8").strip()
    _, n_seen, marks, notes = code.split("|", 3)
    marks, notes = json.loads(marks), json.loads(notes)
    key = {k["id"]: k for k in json.loads((d / "key.json").read_text(encoding="utf-8"))}
    J = {r["id"]: r for r in map(json.loads, (d / "judge-v51.jsonl").read_text(encoding="utf-8").splitlines())}
    v4 = {}
    for fn in ["judge-r4.jsonl", "judge-sp9-r4.jsonl"]:
        for t in (ROOT / "data" / "cloud-v3" / fn).read_text(encoding="utf-8").splitlines():
            r = json.loads(t)
            v4[(r["id"], r["file"], r["lines"].strip())] = r
    print(f"Ken 看过 {n_seen} / {len(key)} 条")

    rows = []      # (qid, ver, 句号, ken, judge, ken_types, judge_types)
    other = []
    for q, k in key.items():
        jr = J[q]
        js = {}
        for e in jr["毛病"]:
            no = e.get("句")
            if not isinstance(no, int):
                continue
            if e["类型"] not in SENT_TYPES:
                other.append((q, e["类型"], e.get("原话")))
            sev = 2 if e.get("轻重") == "重" else 1
            cur = js.get(no, (0, set()))
            js[no] = (max(cur[0], sev), cur[1] | {e["类型"]})
        for no in range(1, k["n"] + 1):
            m = marks.get(f"{q}:{no}")
            ks = int(m[0]) if m else 0
            kt = {TYPES[int(c)] for c in m[1:]} if m else set()
            jsv, jt = js.get(no, (0, set()))
            rows.append((q, k["ver"], no, ks, jsv, kt, jt))

    n = len(rows)
    print(f"\n== 按句（{n} 句）")
    any_pairs = [(r[3] > 0, r[4] > 0) for r in rows]
    agree = sum(a == b for a, b in any_pairs)
    print(f"有没有毛病：一致 {agree}/{n} = {agree / n:.0%}，κ {kappa(any_pairs):.2f}")
    both = sum(a and b for a, b in any_pairs)
    kn, jn = sum(a for a, _ in any_pairs), sum(b for _, b in any_pairs)
    print(f"  Ken 标了 {kn} 句，评审标了 {jn} 句，两边都标 {both} 句 → 评审抓到 Ken 的 {both / max(kn, 1):.0%}；评审标的里 Ken 也认 {both / max(jn, 1):.0%}")
    sev_pairs = [(r[3], r[4]) for r in rows]
    print(f"三档（没 / 轻 / 重）：一致 {sum(a == b for a, b in sev_pairs) / n:.0%}，κ {kappa(sev_pairs):.2f}")
    heavy = [(a == 2, b == 2) for a, b in sev_pairs]
    hb = sum(a and b for a, b in heavy)
    print(f"只看「重」：Ken {sum(a for a, _ in heavy)} 句、评审 {sum(b for _, b in heavy)} 句、都标 {hb}；一致 {sum(a == b for a, b in heavy) / n:.0%}，κ {kappa(heavy):.2f}")
    tab = Counter(sev_pairs)
    print("  交叉表（行 Ken，列评审）：     没   轻   重")
    for a, lab in enumerate(["没", "轻", "重"]):
        print(f"                       {lab}  " + "  ".join(f"{tab[(a, b)]:3d}" for b in range(3)))
    print("\n类型（两边都标了的句子里，类型有交集的比例）：", end=" ")
    tb = [r for r in rows if r[3] and r[4] and r[5]]
    print(f"{sum(bool(r[5] & r[6]) for r in tb)}/{len(tb)}" if tb else "Ken 没选类型")
    print("Ken 选的类型：", dict(Counter(t for r in rows for t in r[5])), "；评审：", dict(Counter(t for r in rows for t in r[6])))

    print("\n== 版本（每句出错率）")
    for v in ("cdpo2s1", "cdpo3s2"):
        rs = [r for r in rows if r[1] == v]
        print(f"{v}: {len(rs)} 句；Ken 有毛病 {sum(r[3] > 0 for r in rs) / len(rs):.0%}（重 {sum(r[3] == 2 for r in rs) / len(rs):.0%}）；"
              f"评审 {sum(r[4] > 0 for r in rs) / len(rs):.0%}（重 {sum(r[4] == 2 for r in rs) / len(rs):.0%}）")

    print("\n== 整条（和 v4 比：Ken 任何一句有毛病 = 整条有毛病）")
    it = []
    for q, k in key.items():
        kq = any(r[3] for r in rows if r[0] == q)
        jq = any(r[4] for r in rows if r[0] == q)
        r4 = next((r for (cid, f, ln), r in v4.items() if cid == k["card"] and ln == "\n".join(k["lines"]).strip()), None)
        p4 = None if r4 is None else bool({"语句不通", "不合人设"} & set(r4["problems"]))
        it.append((kq, jq, p4))
    print(f"v5.1 和 Ken 一致 {sum(a == b for a, b, _ in it)}/{len(it)}")
    ok4 = [(a, c) for a, _, c in it if c is not None]
    print(f"v4（语句不通 / 不合人设）和 Ken 一致 {sum(a == c for a, c in ok4)}/{len(ok4)}")

    print("\n== 分歧明细")
    for r in rows:
        if r[3] != r[4]:
            k = key[r[0]]
            sent = [e for e in J[r[0]]["毛病"] if e.get("句") == r[2]]
            print(f"{r[0]}〔{r[2]}〕Ken {r[3]} {sorted(r[5])} / 评审 {r[4]} {[(e['类型'], e.get('原话')) for e in sent]}")
    if other:
        print("\n评审另报（页面上没有的类型）：", other)
    if notes:
        print("\nKen 的旁注：", json.dumps(notes, ensure_ascii=False))


if __name__ == "__main__":
    main()
