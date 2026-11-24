"""「哪条更顺」重测打分（2026-11-24 对尺子）：Ken 这次 vs 上次，同一对选的是不是同一条。

用法（仓库根目录）：.venv/bin/python scripts/smooth_retest_score.py data/human-check/smooth-retest Ken.txt
（Ken.txt = 结果码那一行存成的文件，放在目录里）
"""
import json
import sys
from collections import Counter
from pathlib import Path


def main():
    d = Path(sys.argv[1])
    _, prefs, notes = (d / sys.argv[2]).read_text(encoding="utf-8").strip().split("|", 2)
    key = json.loads((d / "key.json").read_text(encoding="utf-8"))
    tab = Counter()
    rows = []
    for k, p in zip(key, prefs):
        now = {"A": k["A"], "B": k["B"], "=": "="}.get(p, "?")
        old = k["orig_pref"]
        kind = "一致" if now == old else ("反了" if "=" not in (now, old) else "一边差不多")
        tab[kind] += 1
        rows.append((k["id"], k["test"], k["orig_q"], old, now, kind))
    n = len(rows)
    print(f"{n} 对：一致 {tab['一致']}（{tab['一致'] / n:.0%}），一次选边一次差不多 {tab['一边差不多']}，选边反了 {tab['反了']}")
    sided = [r for r in rows if "=" not in (r[3], r[4])]
    if sided:
        print(f"两次都选了一边的 {len(sided)} 对：同一条 {sum(r[3] == r[4] for r in sided)}")
    old_eq = [r for r in rows if r[3] == "="]
    print(f"上次差不多的 {len(old_eq)} 对，这次仍差不多 {sum(r[4] == '=' for r in old_eq)}")
    for r in rows:
        if r[5] != "一致":
            print("  ", *r)
    if json.loads(notes):
        print("旁注：", notes)


if __name__ == "__main__":
    main()
