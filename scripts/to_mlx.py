"""把挑过的样本转成 mlx-lm 微调用的格式（chat：system / user / assistant）。

- system = 写台词那一步给 DeepSeek 的同一份说明（pilot.SYS_LINES），以后真用时也照这个给；
- user = 渲染好的戏卡（render_card），和生成时 DeepSeek 看到的一样；
- assistant = 台词，一句一行「名字[表情]: 话」，表情是「平常」时不写方括号（只统一格式，不改字）。
训练时用 --mask-prompt，只对 assistant 部分算损失。

只收 meta.keep 明确为 true 的样本（没挑过的不收），source 必须是 deepseek / human / human_edited。
每份来源先过 exam_guard，不过不许转。

用法：
  python3 scripts/to_mlx.py 输出目录 --src a.jsonl b.json ... [--train-only c.json ...] [--ratio 0.1]
  --src 的样本按比例抽验证集；--train-only 的全部进训练集（比如小说样本：这样加不加它们，验证集都是同一份，
  验证损失才能直接比）。.jsonl 一行一条；.json 是样本数组（samples.json）。
旧用法 `to_mlx.py 输入.jsonl 输出目录 [比例]` 仍然能用。
"""

import argparse
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pilot import SYS_LINES  # noqa: E402
from render_card import render_card  # noqa: E402

TRAINABLE = ("deepseek", "human", "human_edited")


def lines_text(output):
    out = []
    for o in output:
        emo = o.get("emotion", "")
        tag = f"[{emo}]" if emo and emo != "平常" else ""
        out.append(f"{o['name']}{tag}: {o['text']}")
    return "\n".join(out)


def read(path):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    rows = json.loads(text) if p.suffix == ".json" else [json.loads(l) for l in text.splitlines() if l.strip()]
    return [r for r in rows if r.get("meta", {}).get("keep") is True and r["source"] in TRAINABLE]


def guard(rows, name):
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        f.write("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    g = subprocess.run([sys.executable, str(HERE / "exam_guard.py"), f.name])
    if g.returncode != 0:
        raise SystemExit(f"exam_guard 不通过（{name}），不转。")


def chat(r):
    return {"messages": [{"role": "system", "content": SYS_LINES},
                         {"role": "user", "content": render_card(r["card"])},
                         {"role": "assistant", "content": lines_text(r["output"])}]}


def main():
    if len(sys.argv) >= 3 and not sys.argv[1].startswith("-") and Path(sys.argv[1]).suffix in (".jsonl", ".json") \
            and not any(a.startswith("--") for a in sys.argv[2:]):
        sys.argv = [sys.argv[0], sys.argv[2], "--src", sys.argv[1]] + (["--ratio", sys.argv[3]] if len(sys.argv) > 3 else [])
    ap = argparse.ArgumentParser()
    ap.add_argument("dst")
    ap.add_argument("--src", nargs="+", default=[])
    ap.add_argument("--train-only", nargs="*", default=[])
    ap.add_argument("--ratio", type=float, default=0.1)
    a = ap.parse_args()

    pool, extra = [], []
    for path in a.src:
        rows = read(path)
        guard(rows, path)
        pool += rows
        print(f"{path}：{len(rows)} 条")
    for path in a.train_only:
        rows = read(path)
        guard(rows, path)
        extra += rows
        print(f"{path}：{len(rows)} 条（全进训练集）")
    data = [chat(r) for r in pool]
    random.Random(20261118).shuffle(data)
    n_valid = max(1, round(len(data) * a.ratio))
    valid, train = data[:n_valid], data[n_valid:] + [chat(r) for r in extra]
    random.Random(20261119).shuffle(train)
    dst = Path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)
    for name, part in (("valid", valid), ("train", train)):
        (dst / f"{name}.jsonl").write_text("\n".join(json.dumps(d, ensure_ascii=False) for d in part) + "\n", encoding="utf-8")
        print(f"{name}: {len(part)} 条")


if __name__ == "__main__":
    main()
