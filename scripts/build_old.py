"""实验 B（2026-11-21）：只用旧数据（v2a2）的 700 条，和 v3 同条数、同验证集、同参数，只差「batch-3 257 条 → 换成 257 条旧数据」。

做法：v3 里的 443 条 v2a2 原样保留，再从 v2a2 剩下的 329 条里随机补 257 条；顺序打乱（种子同 v3）。
用法：.venv/bin/python scripts/build_old.py   输出 data/v3/mlx-old/（train.jsonl / valid.jsonl）
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def jl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def key(c):
    return c["messages"][2]["content"]


v2a2 = jl(ROOT / "data/v2/mlx-a2/train.jsonl")
v3 = jl(ROOT / "data/v3/mlx-v3/train.jsonl")
old_keys = {key(c) for c in v2a2}
keep = [c for c in v3 if key(c) in old_keys]
used = {key(c) for c in keep}
rest = [c for c in v2a2 if key(c) not in used]
n_add = len(v3) - len(keep)
random.Random(20261121).shuffle(rest)
rows = keep + rest[:n_add]
random.Random(20261121).shuffle(rows)
assert len(rows) == len(v3) and len({key(c) for c in rows}) == len(rows)
d = ROOT / "data/v3/mlx-old"
d.mkdir(parents=True, exist_ok=True)
(d / "train.jsonl").write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in rows) + "\n", encoding="utf-8")
(d / "valid.jsonl").write_text((ROOT / "data/v3/mlx-v3/valid.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
print(f"mlx-old: 保留 v3 里的旧数据 {len(keep)} + 补 {n_add} = {len(rows)} 条（v2a2 共 {len(v2a2)}），valid 同 v3")
