"""把考卷输出截到每张卡【要求】的最多句数（模拟游戏代码的把关：多出来的句子不要）。

每块 `=== 卡号（名字）` 后面只留前 N 个非空行，N = 考卷里这张卡 card.rules.lines[1]；空行去掉。
2026-11-20 起用（原来是内联代码，做法一样）。

用法（在仓库根目录跑）：
  .venv/bin/python scripts/cut_exam.py 输出前缀 截断后前缀 [--exam exam/考卷.jsonl]
  例：.venv/bin/python scripts/cut_exam.py data/x/c2bh-pen data/x/c2bhcut-pen   # 处理 -s0/-s1/-s2
"""

import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent


def cut_text(text, max_lines):
    out, cur, n = [], None, 0
    for line in text.splitlines():
        m = re.match(r"=== (\S+?)（", line)
        if m:
            cur, n = m.group(1), 0
            out.append(line)
        elif cur and line.strip():
            n += 1
            if n <= max_lines.get(cur, 10**9):
                out.append(line.strip())
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--exam", default=str(HERE.parent / "exam" / "考卷.jsonl"))
    a = ap.parse_args()
    max_lines = {}
    for t in Path(a.exam).read_text(encoding="utf-8").splitlines():
        if t.strip():
            d = json.loads(t)
            max_lines[d["id"]] = d["card"]["rules"]["lines"][1]
    src = Path(a.src)
    files = sorted(src.parent.glob(src.name + "-s*.txt"))
    for p in files:
        q = Path(a.dst + p.name[len(src.name):])
        q.write_text(cut_text(p.read_text(encoding="utf-8"), max_lines), encoding="utf-8")
        print(p, "→", q)


if __name__ == "__main__":
    main()
