"""v3n2（2026-11-21）：和 v3n 完全一样的 700 条、一样的顺序，只把其中的小说样本换成 rewrite_must.py 重写过「要办成」的版本。
对比：v3n2 vs v3（两边各 2 个种子，v3 / v3 s1 已训过）、vs v3n（看重写的作用）。
用法：.venv/bin/python scripts/build_v3n2.py   输出 data/v3/mlx-v3n2/
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from to_mlx import chat  # noqa: E402
from judge_train import NOVEL, kept  # noqa: E402


def jl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def key(c):
    return c["messages"][2]["content"]


new = {}
for r in json.loads((ROOT / "data/novel-must/samples.json").read_text(encoding="utf-8")):
    d, i = r["meta"]["from"].split(":", 1)
    old = next(x for x in kept(d) if x["id"] == i)
    new[key(chat(old))] = chat(r)
src = jl(ROOT / "data/v3/mlx-v3n/train.jsonl")
rows, n = [], 0
for c in src:
    if key(c) in new:
        rows.append(new[key(c)]); n += 1
    else:
        rows.append(c)
man = json.loads((ROOT / "data/v3/manifest.json").read_text(encoding="utf-8"))["mlx-v3n"]
n_nov = sum(1 for s in man if s.split(":")[0] in ("v2b2", "novel-3", "novel-3b", "novel-4"))
d = ROOT / "data/v3/mlx-v3n2"
d.mkdir(parents=True, exist_ok=True)
(d / "train.jsonl").write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in rows) + "\n", encoding="utf-8")
(d / "valid.jsonl").write_text((ROOT / "data/v3/mlx-v3n/valid.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
print(f"mlx-v3n2: {len(rows)} 条，换掉小说 {n} / {n_nov} 条")
assert n == n_nov, "有小说样本没对上"
