"""拿几张考卷戏卡让本地模型写台词，只用来看流程通不通、训练前后有没有变化（不算正式考试）。

用法：.venv/bin/python scripts/try_exam.py 模型 [--adapter 适配器目录] [--ids exam-S01,exam-N05]
     [--temp 0.7 --top-p 0.9 --rep-penalty 1.15 --seed 0]
不给 --temp 就是贪心解码（每步挑最可能的字，结果固定）。2026-11-19：贪心解码下考卷约 1/4 有整段重复
（训练数据里只有 4%），所以加了采样和重复惩罚的开关，接进游戏时也该用这些。
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pilot import SYS_LINES  # noqa: E402
from render_card import render_card  # noqa: E402

import mlx.core as mx  # noqa: E402
from mlx_lm import generate, load  # noqa: E402
from mlx_lm.sample_utils import make_logits_processors, make_sampler  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--adapter")
    ap.add_argument("--ids", default="exam-S01,exam-N05")
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=0.0)
    ap.add_argument("--rep-penalty", type=float, default=None, help="重复惩罚，1.0 = 不罚，常用 1.1–1.2")
    ap.add_argument("--rep-context", type=int, default=64, help="重复惩罚往回看多少个 token")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    exam = {json.loads(l)["id"]: json.loads(l) for l in (HERE.parent / "exam" / "考卷.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    model, tok = load(a.model, adapter_path=a.adapter)
    sampler = make_sampler(temp=a.temp, top_p=a.top_p)
    procs = make_logits_processors(repetition_penalty=a.rep_penalty, repetition_context_size=a.rep_context) if a.rep_penalty else None
    for i in a.ids.split(","):
        mx.random.seed(a.seed)          # 每张卡同一个种子：结果能复现，换卡顺序也不影响
        msgs = [{"role": "system", "content": SYS_LINES}, {"role": "user", "content": render_card(exam[i]["card"])}]
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        print(f"=== {i}（{exam[i]['card']['speaker']['name']}）")
        print(generate(model, tok, prompt, max_tokens=a.max_tokens, sampler=sampler, logits_processors=procs).strip())


if __name__ == "__main__":
    main()
