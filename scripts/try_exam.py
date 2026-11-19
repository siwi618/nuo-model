"""拿几张考卷戏卡让本地模型写台词，只用来看流程通不通、训练前后有没有变化（不算正式考试）。

用法：.venv/bin/python scripts/try_exam.py 模型 [--adapter 适配器目录] [--ids exam-S01,exam-N05]
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pilot import SYS_LINES  # noqa: E402
from render_card import render_card  # noqa: E402

from mlx_lm import generate, load  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--adapter")
    ap.add_argument("--ids", default="exam-S01,exam-N05")
    ap.add_argument("--max-tokens", type=int, default=300)
    a = ap.parse_args()
    exam = {json.loads(l)["id"]: json.loads(l) for l in (HERE.parent / "exam" / "考卷.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    model, tok = load(a.model, adapter_path=a.adapter)
    for i in a.ids.split(","):
        msgs = [{"role": "system", "content": SYS_LINES}, {"role": "user", "content": render_card(exam[i]["card"])}]
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        print(f"=== {i}（{exam[i]['card']['speaker']['name']}）")
        print(generate(model, tok, prompt, max_tokens=a.max_tokens).strip())


if __name__ == "__main__":
    main()
