"""组装第 4 批训练数据（2026-11-26，设计见 docs/batch-4-设计.md）：在 dpo2-s1 上接着训一段新 LoRA。

- 新数据：batch-4 挑稿留下的（meta.keep 为真）。留出 HOLD 条做验证（按说话人分，不和训练重名），其余进训练。
- 防忘：从 v3 训练集（data/v3/mlx-v3/train.jsonl，700 条）随机抽「新数据条数 × OLD_RATIO」条。
- 验证集 = batch-4 留出的 HOLD 条 + v3 验证集前 VAL_OLD 条（train_lora 只用前 45 条，所以新的放前面）。
  只用 v3 的验证集挑检查点，会偏向「一点没学」的早期检查点。
- 新样本转换前过 exam_guard（公开卷、隐藏卷、专项卷）。

用法：.venv/bin/python scripts/build_b4.py
输出：data/b4/mlx-b4/（train.jsonl / valid.jsonl）、data/b4/manifest.json
"""

import json
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from build_v3 import guard, jl, kept  # noqa: E402
from to_mlx import chat  # noqa: E402

OLD_RATIO = 2
HOLD = 12
VAL_OLD = 33
SEED = 20261126


def main():
    rnd = random.Random(SEED)
    b4 = kept("data/batch-4/samples.json")
    guard(b4, "batch-4")

    # 留出：整个说话人一起留出，直到够 HOLD 条
    by = {}
    for r in b4:
        by.setdefault(r["card"]["speaker"]["name"], []).append(r)
    names = sorted(by)
    rnd.shuffle(names)
    hold, train_new = [], []
    for n in names:
        (hold if len(hold) + len(by[n]) <= HOLD and len(hold) < HOLD else train_new).extend(by[n])

    old = jl(ROOT / "data/v3/mlx-v3/train.jsonl")
    rnd.shuffle(old)
    n_old = len(train_new) * OLD_RATIO
    old = old[:n_old]

    rows = [(f"batch-4:{r['id']}", chat(r)) for r in train_new] + [("v3", c) for c in old]
    random.Random(SEED + 1).shuffle(rows)
    valid = [chat(r) for r in hold] + jl(ROOT / "data/v3/mlx-v3/valid.jsonl")[:VAL_OLD]

    out = ROOT / "data/b4/mlx-b4"
    out.mkdir(parents=True, exist_ok=True)
    (out / "train.jsonl").write_text("\n".join(json.dumps(c, ensure_ascii=False) for _, c in rows) + "\n", encoding="utf-8")
    (out / "valid.jsonl").write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in valid) + "\n", encoding="utf-8")
    types = Counter(r["meta"]["plan"]["b4_type"] for r in train_new)
    manifest = {"train": [s for s, _ in rows], "valid_b4": [r["id"] for r in hold], "valid_v3": VAL_OLD,
                "train_new": len(train_new), "train_old": len(old), "types": dict(types)}
    (ROOT / "data/b4/manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"训练 {len(rows)} 条 = batch-4 {len(train_new)}（{dict(sorted(types.items()))}）+ v3 旧 {len(old)}")
    print(f"验证 {len(valid)} 条 = batch-4 留出 {len(hold)}（{len({r['card']['speaker']['name'] for r in hold})} 个说话人）+ v3 {VAL_OLD}")


if __name__ == "__main__":
    main()
