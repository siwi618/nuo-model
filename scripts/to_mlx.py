"""把挑过的样本转成 mlx-lm 微调用的格式（chat：system / user / assistant）。

- system = 写台词那一步给 DeepSeek 的同一份说明（pilot.SYS_LINES），以后真用时也照这个给；
- user = 渲染好的戏卡（render_card），和生成时 DeepSeek 看到的一样；
- assistant = DeepSeek 写的台词，一句一行「名字[表情]: 话」，表情是「平常」时不写方括号（只统一格式，不改字）。
训练时用 --mask-prompt，只对 assistant 部分算损失。

用法：python3 scripts/to_mlx.py data/batch-1/train-kept.jsonl data/batch-1/mlx [验证集比例，默认 0.1]
会先跑 exam_guard，不过不许转。
"""

import json
import random
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pilot import SYS_LINES  # noqa: E402
from render_card import render_card  # noqa: E402


def lines_text(output):
    out = []
    for o in output:
        emo = o.get("emotion", "")
        tag = f"[{emo}]" if emo and emo != "平常" else ""
        out.append(f"{o['name']}{tag}: {o['text']}")
    return "\n".join(out)


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    ratio = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
    guard = subprocess.run([sys.executable, str(HERE / "exam_guard.py"), str(src)])
    if guard.returncode != 0:
        raise SystemExit("exam_guard 不通过，不转。")
    rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r.get("meta", {}).get("keep", True) and r["source"] in ("deepseek", "human", "human_edited")]
    data = [{"messages": [{"role": "system", "content": SYS_LINES},
                          {"role": "user", "content": render_card(r["card"])},
                          {"role": "assistant", "content": lines_text(r["output"])}]} for r in rows]
    random.Random(20261118).shuffle(data)
    n_valid = max(1, round(len(data) * ratio))
    dst.mkdir(parents=True, exist_ok=True)
    for name, part in (("valid", data[:n_valid]), ("train", data[n_valid:])):
        (dst / f"{name}.jsonl").write_text("\n".join(json.dumps(d, ensure_ascii=False) for d in part) + "\n", encoding="utf-8")
        print(f"{name}: {len(part)} 条")


if __name__ == "__main__":
    main()
