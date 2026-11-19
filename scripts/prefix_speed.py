"""测「档案前缀缓存」能把第一个字提前多少。

固定前缀 = 系统提示 + 【世界】 + 【说话人】（同一个 NPC 每次都一样，游戏里走近时就能先算好）；
其余（【对方】【场景】【前文】【目的】【要求】）每次现算。对每张卡比两种：
  不缓存：整张卡从头读，量第一个字的时间；
  缓存：  先把前缀读进 KV 缓存（记「预读」时间，游戏里不算在等待里），复制一份缓存，只喂剩下的部分，量第一个字。

用法（仓库根目录，电脑空闲时跑；几个模型紧挨着测）：
  .venv/bin/python scripts/prefix_speed.py 模型目录 [--n 8] [--max-tokens 120] [--check]
--check：第一张卡用贪心解码对比两种方式的输出是否一致（验证缓存没接错）。
"""

import argparse
import copy
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
from mlx_lm.models.cache import make_prompt_cache  # noqa: E402
from mlx_lm.sample_utils import make_sampler  # noqa: E402

CUT = "\n\n【对方】"


def split_ids(tok, card):
    msgs = [{"role": "system", "content": SYS_LINES}, {"role": "user", "content": render_card(card)}]
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    ids = tok.encode(prompt, add_special_tokens=False)
    pre = tok.encode(prompt[: prompt.index(CUT)], add_special_tokens=False)
    n = 0
    while n < len(pre) and pre[n] == ids[n]:
        n += 1
    return ids, n


def run(model, tok, ids, sampler, max_tokens, cache=None):
    t, first, text, r = time.perf_counter(), None, "", None
    for r in stream_generate(model, tok, ids, max_tokens=max_tokens, sampler=sampler, prompt_cache=cache):
        if first is None:
            first = time.perf_counter() - t
        text += r.text
    return first, time.perf_counter() - t, text, r


def prefill(model, ids):
    cache = make_prompt_cache(model)
    t = time.perf_counter()
    model(mx.array(ids)[None], cache=cache)
    mx.eval([c.state for c in cache])
    return cache, time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--exam", default=str(HERE.parent / "exam" / "考卷.jsonl"))
    ap.add_argument("--n", type=int, default=8, help="每隔 60/n 张取一张")
    ap.add_argument("--max-tokens", type=int, default=120)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    exam = [json.loads(t) for t in Path(a.exam).read_text(encoding="utf-8").splitlines() if t.strip()]
    exam = exam[:: max(1, len(exam) // a.n)][: a.n]
    model, tok = load(a.model)
    greedy = make_sampler(temp=0.0)

    # 热身（编译内核），不计时
    ids, n = split_ids(tok, exam[0]["card"])
    run(model, tok, ids, greedy, 8)
    c, _ = prefill(model, ids[:n])
    run(model, tok, ids[n:], greedy, 8, cache=c)

    if a.check:
        _, _, t1, _ = run(model, tok, ids, greedy, a.max_tokens)
        c, _ = prefill(model, ids[:n])
        _, _, t2, _ = run(model, tok, ids[n:], greedy, a.max_tokens, cache=c)
        print("一致" if t1 == t2 else f"不一致！\n--- 不缓存\n{t1}\n--- 缓存\n{t2}")

    rows = []
    for e in exam:
        ids, n = split_ids(tok, e["card"])
        f0, tot0, _, r0 = run(model, tok, ids, greedy, a.max_tokens)
        base, pre_s = prefill(model, ids[:n])
        f1, tot1, _, r1 = run(model, tok, ids[n:], greedy, a.max_tokens, cache=copy.deepcopy(base))
        rows.append((e["id"], len(ids), n, f0, f1, pre_s, r0.generation_tps, max(r0.peak_memory, r1.peak_memory)))
        print(f"{e['id']}  卡 {len(ids)} tok，前缀 {n}（{n / len(ids):.0%}）| 首字 不缓存 {f0:.2f}s → 缓存 {f1:.2f}s"
              f" | 预读 {pre_s:.2f}s | {r0.generation_tps:.0f} tok/s | 峰值 {rows[-1][-1]:.2f} GB", flush=True)
        del base

    def mid(x):
        x = sorted(x)
        return x[len(x) // 2]
    print(f"== {Path(a.model).name}：{len(rows)} 张 | 前缀占 {mid([r[2] / r[1] for r in rows]):.0%}"
          f" | 首字中位 不缓存 {mid([r[3] for r in rows]):.2f}s → 缓存 {mid([r[4] for r in rows]):.2f}s"
          f"（最长 {max(r[4] for r in rows):.2f}s）| 预读中位 {mid([r[5] for r in rows]):.2f}s"
          f" | 生成 {mid([r[6] for r in rows]):.0f} tok/s | 峰值 {max(r[7] for r in rows):.2f} GB")


if __name__ == "__main__":
    main()
