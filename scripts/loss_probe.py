"""查 DPO 后「v3 验证集普通损失」为什么涨（2026-11-24，第三轮 1.841 → 1.96）。

本机 MLX 4 位模型上算（绝对值和云上 bf16 不同，只看版本之间的差）：
1. 验证集（train_dpo 用的同一份：mlx-v3/valid.jsonl 前 45 条）每个 token 的损失，拆成两块：
   -log p(tok) = -log(1 - p_end)          ← 「想停」：在这里就想结束的概率压掉了多少
               + -log p(tok | 不结束)       ← 「内容」：不结束时选对这个字的难度
   最后的 <|im_end|> 单独算（「该停时停」）。
2. 偏好对：每对 chosen / rejected 的对数概率和（减去参考模型的 = DPO 的隐式奖励），
   看留出对上模型选对的，是不是就是「短的那条」。

内存（11-24 跑第一次时整机卡死重启过）：词表 24.8 万，一次前向的整段 logits 约 0.5 GB；
MLX 默认把用完的显存留作缓存，长短不一的序列会让缓存越堆越大 → 8 GB 机器狂换页。
所以关掉缓存、每次前向后清空；外面再套 caffeinate -i（别在跑的时候进睡眠）+ 内存守护。
时间：戏卡开头约 1250 token，本机每段约 3 秒；每个模型 435 段约 22 分钟，4 个约 90 分钟。
要快：只算留出对 + 抽一部分训练对；同卡好 / 坏例子共用开头的 KV 缓存；参考模型结果存下来复用。

用法（仓库根目录，别和考试同时跑）：
  .venv/bin/python scripts/loss_probe.py --ref models/q35-2b-v3-dpo2-s1-mlx4 \
      --models models/q35-2b-v3-dpo3-s0-mlx4,... --pairs data/dpo3/pairs.jsonl --out data/dpo3/loss_probe.json
"""
import argparse
import gc
import math
import json
import random
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

END = "<|im_end|>"


def enc_prompt(tok, msgs):
    p = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    return tok.encode(p, add_special_tokens=False)


def enc_resp(tok, text):
    return tok.encode(text + END + "\n", add_special_tokens=False)


def token_lp(model, p_ids, r_ids, end_id):
    """回答部分：每个 token 的 log p、log p_end（同一位置想结束的概率）。"""
    ids = mx.array(p_ids + r_ids)[None]
    logits = model(ids)[0, len(p_ids) - 1:-1].astype(mx.float32)
    lsm = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    tgt = mx.array(r_ids)
    lp = mx.take_along_axis(lsm, tgt[:, None], axis=-1)[:, 0]
    lend = lsm[:, end_id]
    mx.eval(lp, lend)
    out = lp.tolist(), lend.tolist()
    del ids, logits, lsm, lp, lend
    mx.clear_cache()
    return out


def probe(path, val, pairs, end_id):
    model, tok = load(path)
    out = {"val": [], "pairs": []}
    for msgs in val:
        p, r = enc_prompt(tok, msgs[:-1]), enc_resp(tok, msgs[-1]["content"])
        lp, lend = token_lp(model, p, r, end_id)
        k = r.index(end_id)                      # <|im_end|> 的位置；后面的 \n 不算进「内容」
        stop = sum(-math.log1p(-math.exp(x)) for x in lend[:k])
        content = sum(-(a - math.log1p(-math.exp(b))) for a, b in zip(lp[:k], lend[:k]))
        out["val"].append({"n": len(r), "nll": -sum(lp), "stop": stop, "content": content,
                           "end": -lp[k], "after": -sum(lp[k + 1:]), "body_tokens": k})
    for pr in pairs:
        p = enc_prompt(tok, pr["prompt"])
        c, r = enc_resp(tok, pr["chosen"]), enc_resp(tok, pr["rejected"])
        out["pairs"].append({"id": pr["id"], "c": sum(token_lp(model, p, c, end_id)[0]), "r": sum(token_lp(model, p, r, end_id)[0]),
                             "lc": len(c), "lr": len(r)})
    del model
    gc.collect(); mx.clear_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--val", default="data/v3/mlx-v3/valid.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--beta", type=float, default=0.1)
    a = ap.parse_args()
    mx.set_cache_limit(0)

    val = [json.loads(t)["messages"] for t in Path(a.val).read_text(encoding="utf-8").splitlines() if t.strip()][:45]
    pairs = [json.loads(t) for t in Path(a.pairs).read_text(encoding="utf-8").splitlines() if t.strip()]
    ids = sorted({p["id"] for p in pairs}); random.Random(1).shuffle(ids)          # 和 train_dpo.py 同一种留出法
    hold = set(ids[:max(1, int(len(ids) * 0.1))])
    _, tok = load(a.ref)
    end_id = tok.encode(END, add_special_tokens=False)[0]

    res = {}
    for path in [a.ref] + a.models.split(","):
        name = Path(path).name
        print("==", name, flush=True)
        res[name] = probe(path, val, pairs, end_id)
    Path(a.out).write_text(json.dumps({"hold": sorted(hold), "res": res}, ensure_ascii=False), encoding="utf-8")

    ref = res[Path(a.ref).name]
    print("\n验证集（45 条，每 token 平均；本机 4 位）")
    print("  版本 | 损失 | 想停 | 内容 | 结束符 | 正文 token")
    for name, r in res.items():
        T = sum(v["n"] for v in r["val"])
        s = lambda k: sum(v[k] for v in r["val"]) / T
        print(f"  {name} | {s('nll'):.4f} | {s('stop'):.4f} | {s('content'):.4f} | {s('end'):.4f} | {sum(v['body_tokens'] for v in r['val'])}")

    print("\n偏好对：隐式奖励差 m = β[(c−c_ref) − (r−r_ref)]；「短」= chosen token 更少")
    for name, r in res.items():
        if name == Path(a.ref).name:
            continue
        for lab, sel in (("留出", lambda i: i in hold), ("训练", lambda i: i not in hold)):
            rows = [(x, y) for x, y in zip(r["pairs"], ref["pairs"]) if sel(x["id"])]
            m = [a.beta * ((x["c"] - y["c"]) - (x["r"] - y["r"])) for x, y in rows]
            short = [x["lc"] < x["lr"] for x, _ in rows]
            acc = sum(v > 0 for v in m) / len(m)
            same = sum((v > 0) == s for v, s in zip(m, short)) / len(m)
            acc_s = [v > 0 for v, s in zip(m, short) if s]; acc_l = [v > 0 for v, s in zip(m, short) if not s]
            print(f"  {name} {lab} {len(m)} 对：选对 {acc:.0%}｜和「选短的」一致 {same:.0%}｜"
                  f"chosen 短的对选对 {sum(acc_s)}/{len(acc_s)}，chosen 长的对选对 {sum(acc_l)}/{len(acc_l)}")
        # 每个 token 的平均对数概率变化：chosen / rejected 各自
        dc = sum(x["c"] - y["c"] for x, y in zip(r["pairs"], ref["pairs"])) / sum(x["lc"] for x in r["pairs"])
        dr = sum(x["r"] - y["r"] for x, y in zip(r["pairs"], ref["pairs"])) / sum(x["lr"] for x in r["pairs"])
        print(f"    全部 195 对每 token 对数概率变化：chosen {dc:+.4f}，rejected {dr:+.4f}")


if __name__ == "__main__":
    main()
