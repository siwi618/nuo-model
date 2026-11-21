"""组装 v3 / v3n 训练数据（2026-11-20 Ken 定：同条数对比，总约 700 条，v3n 里小说占 25%）。

- DeepSeek 样本池 = v2a2（data/v2/mlx-a2/train.jsonl，772 条，已是 chat 格式）+ batch-3 挑稿留下的（data/batch-3）。
- 小说池 = v2b2 里的旧小说（mlx-b2 比 mlx-a2 多出来的那些）+ novel-3 / novel-3b / novel-4 挑稿留下的；按答案去重。
- v3  = DeepSeek 池里按来源分层抽 TOTAL 条；
  v3n = 和 v3 共用其中 TOTAL − N_NOVEL 条（同样分层）+ 小说 N_NOVEL 条（每本书最多占小说的 25%）。
  两版条数一样、验证集一样（v2a2 的 valid，87 条），只差「小说 vs 同样多的 DeepSeek 样本」。
- 新样本（batch-3、新小说）转换前过 exam_guard：公开卷、隐藏卷、专项卷三份都查。

用法：.venv/bin/python scripts/build_v3.py [--total 700] [--novel-share 0.25]
输出：data/v3/mlx-v3/、data/v3/mlx-v3n/（train.jsonl / valid.jsonl）、data/v3/manifest.json（每条来源，可复查）。
"""

import argparse
import json
import random
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from to_mlx import TRAINABLE, chat  # noqa: E402

EXAMS = ["exam/考卷.jsonl", "exam/隐藏卷.jsonl", "exam/专项-心里话.jsonl"]


def jl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def kept(path):
    rows = json.loads((ROOT / path).read_text(encoding="utf-8"))
    return [r for r in rows if r.get("meta", {}).get("keep") is True and r["source"] in TRAINABLE]


def guard(rows, name):
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        f.write("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    for ex in EXAMS:
        g = subprocess.run([sys.executable, str(HERE / "exam_guard.py"), f.name, "--exam", str(ROOT / ex)],
                           capture_output=True, text=True)
        if g.returncode != 0:
            sys.exit(f"exam_guard 不通过（{name} × {ex}）：\n{g.stdout[-800:]}")
    print(f"  守卫通过：{name} {len(rows)} 条 × {len(EXAMS)} 份考卷")


def key(c):
    return c["messages"][2]["content"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=700)
    ap.add_argument("--novel-share", type=float, default=0.25)
    ap.add_argument("--full", action="store_true", help="另出正式版候选：DeepSeek 池全部 + 小说池全部 → data/v3/mlx-full")
    a = ap.parse_args()
    n_nov = round(a.total * a.novel_share)
    rnd = random.Random(20261120)

    # DeepSeek 池
    v2a2 = jl(ROOT / "data/v2/mlx-a2/train.jsonl")
    b3 = kept("data/batch-3/samples.json")
    guard(b3, "batch-3")
    ds = {"v2a2": [("v2a2", c) for c in v2a2], "batch-3": [(f"batch-3:{r['id']}", chat(r)) for r in b3]}

    # 小说池（旧小说 = mlx-b2 比 mlx-a2 多出来的）
    a_keys = {key(c) for c in v2a2}
    old_nov = [c for c in jl(ROOT / "data/v2/mlx-b2/train.jsonl") if key(c) not in a_keys]
    nov = [("旧小说", "v2b2", c) for c in old_nov]
    for d in ("novel-3", "novel-3b", "novel-4"):
        rows = kept(f"data/{d}/samples.json")
        guard(rows, d)
        nov += [(r["meta"]["ref"]["book"], f"{d}:{r['id']}", chat(r)) for r in rows]
    seen, uniq = set(), []
    for b, src, c in nov:
        if key(c) not in seen:
            seen.add(key(c))
            uniq.append((b, src, c))
    print(f"DeepSeek 池：v2a2 {len(ds['v2a2'])} + batch-3 {len(ds['batch-3'])}；小说池 {len(uniq)} 条（旧 {len(old_nov)}）")
    if len(uniq) < n_nov:
        sys.exit(f"小说只有 {len(uniq)} 条，不够 {n_nov} 条")

    # 小说抽 n_nov 条：新小说按书限 25%（旧小说当一组，原本就混了 6 种书）
    cap = int(n_nov * 0.25)
    rnd.shuffle(uniq)
    by, pick = Counter(), []
    for b, src, c in uniq:
        if len(pick) < n_nov and (b == "旧小说" or by[b] < cap):
            pick.append((b, src, c))
            by[b] += 1
    if len(pick) < n_nov:
        sys.exit(f"按每本书 ≤ {cap} 条只抽到 {len(pick)} 条")

    # DeepSeek 分层抽：共用部分 n_common、v3 另加 n_nov 条，各来源按池子大小比例
    n_common = a.total - n_nov
    tot = sum(len(v) for v in ds.values())
    common, extra = [], []
    for name, rows in ds.items():
        rows = rows[:]
        rnd.shuffle(rows)
        k_all = round(a.total * len(rows) / tot)
        k_common = round(n_common * len(rows) / tot)
        common += rows[:k_common]
        extra += rows[k_common:k_all]
    common, extra = common[:n_common], extra[:n_nov]
    valid = jl(ROOT / "data/v2/mlx-a2/valid.jsonl")

    out = ROOT / "data/v3"
    manifest = {}
    for name, rows in (("mlx-v3", common + extra), ("mlx-v3n", common + [(s, c) for _, s, c in pick])):
        rows = rows[:]
        random.Random(20261121).shuffle(rows)
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "train.jsonl").write_text("\n".join(json.dumps(c, ensure_ascii=False) for _, c in rows) + "\n", encoding="utf-8")
        (d / "valid.jsonl").write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in valid) + "\n", encoding="utf-8")
        manifest[name] = [s for s, _ in rows]
        print(f"{name}: train {len(rows)}（{dict(Counter(s.split(':')[0] for s, _ in rows))}），valid {len(valid)}")
    if a.full:                       # 正式版候选（11-20 晚 Ken 定先训着）：全部 DeepSeek + 全部小说，验证集同一份
        rows = [x for v in ds.values() for x in v] + [(src, c) for _, src, c in uniq]
        random.Random(20261122).shuffle(rows)
        d = out / "mlx-full"
        d.mkdir(parents=True, exist_ok=True)
        (d / "train.jsonl").write_text("\n".join(json.dumps(c, ensure_ascii=False) for _, c in rows) + "\n", encoding="utf-8")
        (d / "valid.jsonl").write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in valid) + "\n", encoding="utf-8")
        manifest["mlx-full"] = [s for s, _ in rows]
        print(f"mlx-full: train {len(rows)}，valid {len(valid)}")
    manifest["小说按书"] = dict(by)
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print("小说按书：", dict(by.most_common()))


if __name__ == "__main__":
    main()
