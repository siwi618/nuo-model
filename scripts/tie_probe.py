"""低温下为什么还有多样性（2026-11-25）：逐步记录模型对下一个 token 的概率，看「第一名和第二名挨得很近」的步骤有多少、出在哪类字上。

做法：和 run_exam 完全一样的提示和重复惩罚，按指定温度采样（top_p 0.9）。每一步用模型给的对数概率（已过重复惩罚、温度 1）
算出「按温度 T 缩放后第一名的概率」p1(T) = 1 / Σ exp((l_i − l_1) / T)。
  - 接近平局：p1(T) < 0.9（每 10 次至少 1 次会选别的）
  - 整句一字不差的期望 = 各步 p1(T) 连乘（近似，忽略 top_p 截断）
  - 实际分岔：这一步采到的不是第一名

用法（仓库根目录，插电）：.venv/bin/python scripts/tie_probe.py models/q35-2b-v3-dpo2-s1-mlx4 --temp 0.15 --n 20
"""
import argparse
import math
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mlx.core as mx  # noqa: E402
from mlx_lm import load, stream_generate  # noqa: E402
from mlx_lm.sample_utils import make_sampler  # noqa: E402
import json  # noqa: E402
from run_exam import SYS_LINES, format_ids, gen_only_penalty  # noqa: E402
from render_card import render_card  # noqa: E402

FUNC = set("的了着过吧呢啊呀嘛吗哦哟么就都也还又再才把被给让在是有这那个们")
PUNCT = set("，。！？、；：…—“”‘’「」（）()[]【】,.!?:;\n ")


def kind(s):
    s = s.strip() or s
    if all(c in PUNCT for c in s):
        return "标点 / 换行"
    if len(s) <= 2 and all(c in FUNC for c in s):
        return "虚词（的了吧呢……）"
    if "[" in s or "]" in s or "【" in s:
        return "表情标签"
    return "实词 / 别的"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--temp", type=float, default=0.15)
    ap.add_argument("--n", type=int, default=20, help="抽几张公开卷的卡")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rep-penalty", type=float, default=1.15)
    a = ap.parse_args()
    exam = [json.loads(t) for t in (HERE.parent / "exam" / "考卷.jsonl").read_text(encoding="utf-8").splitlines() if t.strip()]
    exam = random.Random(1).sample(exam, a.n)
    model, tok = load(a.model)
    sampler = make_sampler(temp=a.temp, top_p=0.9)
    T = a.temp
    steps, ties, forks = 0, 0, 0
    tie_kind, fork_kind, first_fork = Counter(), Counter(), []
    seq_same, examples = [], []
    for e in exam:
        msgs = [{"role": "system", "content": SYS_LINES}, {"role": "user", "content": render_card(e["card"])}]
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        ids = tok.encode(prompt, add_special_tokens=False)
        procs = [gen_only_penalty(len(ids), format_ids(tok, e["card"]), a.rep_penalty, 64)]
        mx.random.seed(a.seed)
        logp_same, pos, got_fork, text = 0.0, 0, False, ""
        for r in stream_generate(model, tok, ids, max_tokens=300, sampler=sampler, logits_processors=procs):
            lp = r.logprobs
            top = mx.argsort(-lp)[:5].tolist()
            l = [lp[i].item() for i in top]
            p1 = 1 / sum(math.exp((x - l[0]) / T) for x in l)       # 只算前 5 名，后面的可忽略
            steps += 1
            pos += 1
            logp_same += math.log(p1)
            t1, t2 = tok.decode([top[0]]), tok.decode([top[1]])
            if p1 < 0.9:
                ties += 1
                tie_kind[kind(t1)] += 1
                if len(examples) < 400:
                    examples.append((e["id"], text[-12:], t1, t2, p1))
            if r.token != top[0]:
                forks += 1
                fork_kind[kind(t1)] += 1
                if not got_fork:
                    first_fork.append(pos)
                    got_fork = True
            text += r.text
        seq_same.append(math.exp(logp_same))
    print(f"温度 {T}，{a.n} 张卡，共 {steps} 步（token）")
    print(f"接近平局（p1(T) < 0.9）：{ties} 步 = {ties / steps:.1%}，平均每条 {ties / a.n:.1f} 处")
    print(f"实际采到非第一名：{forks} 步 = {forks / steps:.1%}；有分岔的台词 {len(first_fork)}/{a.n}，第一次分岔在第 {sorted(first_fork)[len(first_fork) // 2] if first_fork else '-'} 步（中位）")
    ss = sorted(seq_same)
    print(f"整句一字不差（同卡再采一次）的期望概率：中位 {ss[len(ss) // 2]:.0%}，平均 {sum(ss) / len(ss):.0%}")
    print("接近平局出在哪类字：", dict(tie_kind.most_common()))
    print("实际分岔出在哪类字：", dict(fork_kind.most_common()))
    print("\n例子（前文末尾 | 第一名 vs 第二名 | p1(T)）：")
    rnd = random.Random(0)
    for cid, ctx, t1, t2, p in rnd.sample(examples, min(25, len(examples))):
        print(f"  {cid} …{ctx!r} | {t1!r} vs {t2!r} | {p:.2f}")


if __name__ == "__main__":
    main()
