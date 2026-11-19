"""考卷 v2 跑法：整份考卷（或指定几张）× 多个随机种子，台词按 score_exam 的格式存，顺带记速度和内存。

用法（在仓库根目录跑）：
  .venv/bin/python scripts/run_exam.py 模型 输出前缀 [--adapter 适配器目录] [--exam exam/考卷.jsonl]
      [--seeds 0,1,2] [--temp 0.7 --top-p 0.9 --rep-penalty 1.15] [--ids exam-S01,...]
输出：输出前缀-s0.txt、-s1.txt……（每个种子一份），输出前缀.timing（速度 / 内存汇总）。

默认用「带一点随机 + 重复惩罚」（接进游戏也打算这么用），每张卡、每个种子单独设种子：
同一种子在不同模型版本之间可以逐卡配对比较（exam_stats.py）。--temp 0 = 贪心解码（几个种子结果一样，只跑一次）。
2026-11-19：贪心解码下「理论上等价」的两个版本 24 张里 15 张台词不同、全合规差 4 张 → 改成多种子 + 扩卷。
"""

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pilot import SYS_LINES  # noqa: E402
from render_card import render_card  # noqa: E402

import mlx.core as mx  # noqa: E402
from mlx_lm import load, stream_generate  # noqa: E402
from mlx_lm.sample_utils import make_logits_processors, make_sampler  # noqa: E402


def format_ids(tok, card):
    """名字、表情、格式符号的 token：每行都要重复写，不能罚。"""
    r = card["rules"]
    words = list(r.get("speakers", [])) + list(r.get("emotions", [])) + ["[", "]", ":", "：", "\n", "（", "）"]
    out = set()
    for w in words:
        for pre in ("", "\n"):
            out.update(tok.encode(pre + w, add_special_tokens=False))
    return mx.array(sorted(out))


def gen_only_penalty(prompt_len, exempt, penalty, context):
    """2026-11-19 发现：mlx-lm 原版重复惩罚把戏卡末尾也算进去，而卡末尾正是【要求】里的「表情：…」「谁能开口：…」，
    结果模型一开口就被罚去换名字、换表情（错人、表情外暴增）。这里只罚自己写出来的，且不罚名字 / 表情 / 格式符号。"""
    def proc(tokens, logits):
        gen = tokens[prompt_len:][-context:]
        if gen.size == 0:
            return logits
        hit = mx.zeros((logits.shape[-1],), dtype=mx.bool_)
        hit[gen] = True
        hit[exempt] = False
        pen = mx.where(logits < 0, logits * penalty, logits / penalty)
        return mx.where(hit[None, :], pen, logits)
    return proc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("out", help="输出前缀，如 data/x/v2a")
    ap.add_argument("--adapter")
    ap.add_argument("--exam", default=str(HERE.parent / "exam" / "考卷.jsonl"))
    ap.add_argument("--ids", help="只考这几张，逗号分隔；不给 = 整份")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--rep-penalty", type=float, default=1.15)
    ap.add_argument("--rep-context", type=int, default=64)
    ap.add_argument("--rep-mode", choices=["gen", "all"], default="gen",
                    help="gen（默认）= 只罚模型自己写出来的字，且不罚名字 / 表情 / 格式符号；all = mlx-lm 原版（连戏卡末尾一起罚）")
    ap.add_argument("--max-tokens", type=int, default=300)
    a = ap.parse_args()

    exam = [json.loads(t) for t in Path(a.exam).read_text(encoding="utf-8").splitlines() if t.strip()]
    if a.ids:
        want = set(a.ids.split(","))
        exam = [e for e in exam if e["id"] in want]
    seeds = [int(s) for s in a.seeds.split(",")] if a.temp > 0 else [0]
    sampler = make_sampler(temp=a.temp, top_p=a.top_p)
    use_pen = bool(a.rep_penalty) and a.rep_penalty != 1.0

    t0 = time.time()
    model, tok = load(a.model, adapter_path=a.adapter)
    load_s = time.time() - t0
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    firsts, totals, tps, peak = [], [], [], 0.0
    for seed in seeds:
        blocks = []
        for e in exam:
            msgs = [{"role": "system", "content": SYS_LINES}, {"role": "user", "content": render_card(e["card"])}]
            prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
            ids = tok.encode(prompt, add_special_tokens=False)
            procs = None
            if use_pen and a.rep_mode == "all":
                procs = make_logits_processors(repetition_penalty=a.rep_penalty, repetition_context_size=a.rep_context)
            elif use_pen:
                procs = [gen_only_penalty(len(ids), format_ids(tok, e["card"]), a.rep_penalty, a.rep_context)]
            mx.random.seed(seed)
            t, first, text, r = time.time(), None, "", None
            for r in stream_generate(model, tok, ids, max_tokens=a.max_tokens, sampler=sampler, logits_processors=procs):
                if first is None:
                    first = time.time() - t
                text += r.text
            totals.append(time.time() - t)
            firsts.append(first)
            tps.append(r.generation_tps)
            peak = max(peak, r.peak_memory)
            blocks.append(f"=== {e['id']}（{e['card']['speaker']['name']}）\n{text.strip()}")
        Path(f"{out}-s{seed}.txt").write_text("\n".join(blocks) + "\n", encoding="utf-8")

    def mid(x):
        return sorted(x)[len(x) // 2]
    summary = (f"{len(exam)} 张 × {len(seeds)} 个种子 | 加载 {load_s:.1f}s | 首字 中位 {mid(firsts):.2f}s 最长 {max(firsts):.2f}s"
               f" | 整句 中位 {mid(totals):.2f}s 最长 {max(totals):.2f}s | 生成 {sum(tps) / len(tps):.0f} tok/s | 峰值 {peak:.2f} GB")
    Path(f"{out}.timing").write_text(summary + "\n", encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
