"""云上 DPO（2026-11-21）：从 v3 出发，用同卡「好 / 坏」台词对，教模型更偏向「藏住心思且办成」、远离「心里话说出口 / 没办成」。

做法（手写 DPO，不依赖 trl）：
  1. 底座 + v3 的 LoRA 检查点 → 合并成 v3 完整模型 = 参考模型，也是起点；
  2. 在上面再加一层新 LoRA（全部层、rank 16，同 v3），只训这层；参考模型 = 同一个模型把新 LoRA 关掉（省一份显存）；
  3. 损失 = −log σ(β·[(策略 − 参考)(好) − (策略 − 参考)(坏)])，可选再加 α × 好例子的平均负对数似然（防止整体变差）；
  4. 每 eval-every 步：留出的 10% 对看「好比坏得分高」的比例和平均差距，另算 v3 验证集的普通损失（看说话本身有没有变差），存检查点；
  5. 训完按 --merge 选的检查点（默认最后一个）合并到 v3 上，存 out/merged/，之后照 v3b.sh 的 conv 转 MLX 4 位。
提示词拼法和 train_lora.py / run_exam.py 完全一致。

用法：python train_dpo.py --model 底座 --init-lora v3检查点 --pairs pairs.jsonl --sft-val v3验证集.jsonl --out 输出 [--beta 0.1 --lr 5e-6 --epochs 2]
"""
import argparse
import json
import math
import random
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_jsonl(p):
    return [json.loads(t) for t in Path(p).read_text(encoding="utf-8").splitlines() if t.strip()]


def enc_prompt(tok, msgs):
    p = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    return tok(p, add_special_tokens=False)["input_ids"]


def enc_resp(tok, text):
    return tok(text + "<|im_end|>\n", add_special_tokens=False)["input_ids"]


def resp_logp(model, p_ids, r_ids, dev):
    """回答部分每个 token 的对数概率之和、token 数。"""
    ids = torch.tensor(p_ids + r_ids, device=dev)[None]
    logits = model(input_ids=ids).logits[0, len(p_ids) - 1:-1].float()
    tgt = ids[0, len(p_ids):]
    lp = torch.log_softmax(logits, -1).gather(-1, tgt[:, None])[:, 0]
    return lp.sum(), len(r_ids)


@torch.no_grad()
def sft_val(model, val, dev):
    model.eval()
    tot, n = 0.0, 0
    for p_ids, r_ids in val:
        s, k = resp_logp(model, p_ids, r_ids, dev)
        tot += -s.item(); n += k
    model.train()
    return tot / max(n, 1)


@torch.no_grad()
def pair_eval(model, pairs, beta, dev):
    model.eval()
    acc, marg = 0, 0.0
    for p in pairs:
        pc, _ = resp_logp(model, p["p"], p["c"], dev)
        pr, _ = resp_logp(model, p["p"], p["r"], dev)
        m = beta * ((pc.item() - p["rc"]) - (pr.item() - p["rr"]))
        acc += m > 0; marg += m
    model.train()
    return acc / max(len(pairs), 1), marg / max(len(pairs), 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--init-lora", required=True, help="v3 的 LoRA 检查点目录（ckpt-600）")
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--sft-val", required=True, help="v3 的 valid.jsonl（messages 格式），看普通损失")
    ap.add_argument("--out", required=True)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--sft-alpha", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--scale", type=float, default=20.0)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--hold", type=float, default=0.1, help="留出多少对做评估（按卡分，不和训练混）")
    ap.add_argument("--merge", default="last", help="合并哪个检查点：last 或步数")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); random.seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(a.model)
    base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, a.init_lora).merge_and_unload().to(dev)   # = v3
    model.gradient_checkpointing_enable(); model.enable_input_require_grads(); model.config.use_cache = False
    targets = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and "visual" not in n
               and "lm_head" not in n and re.search(r"\.layers\.\d+\.", n)]
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=a.rank * a.scale, lora_dropout=0.0,
                                             target_modules=targets, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    raw = load_jsonl(a.pairs)
    ids = sorted({p["id"] for p in raw}); random.Random(1).shuffle(ids)
    hold_ids = set(ids[:max(1, int(len(ids) * a.hold))])
    data = []
    with torch.no_grad(), model.disable_adapter():      # 参考模型的对数概率先算好
        model.eval()
        for p in raw:
            pi, ci, ri = enc_prompt(tok, p["prompt"]), enc_resp(tok, p["chosen"]), enc_resp(tok, p["rejected"])
            rc, _ = resp_logp(model, pi, ci, dev); rr, _ = resp_logp(model, pi, ri, dev)
            data.append({"id": p["id"], "p": pi, "c": ci, "r": ri, "rc": rc.item(), "rr": rr.item()})
        model.train()
    train = [d for d in data if d["id"] not in hold_ids]
    hold = [d for d in data if d["id"] in hold_ids]
    val = [(enc_prompt(tok, d["messages"][:-1]), enc_resp(tok, d["messages"][-1]["content"])) for d in load_jsonl(a.sft_val)][:45]
    steps = math.ceil(len(train) * a.epochs)
    order = []
    while len(order) < steps:
        idx = list(range(len(train))); random.shuffle(idx); order += idx
    order = order[:steps]
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    log = (out / "log.jsonl").open("w", encoding="utf-8")

    def write(rec):
        print(json.dumps(rec, ensure_ascii=False), flush=True)
        log.write(json.dumps(rec, ensure_ascii=False) + "\n"); log.flush()

    acc0, m0 = pair_eval(model, hold, a.beta, dev)
    write({"step": 0, "hold_acc": round(acc0, 3), "hold_margin": round(m0, 4), "sft_val": round(sft_val(model, val, dev), 4),
           "train_pairs": len(train), "hold_pairs": len(hold), "steps": steps, "beta": a.beta, "lr": a.lr, "sft_alpha": a.sft_alpha})
    t0, run, racc = time.time(), 0.0, 0
    saved = []
    for step, i in enumerate(order, 1):
        d = train[i]
        pc, nc = resp_logp(model, d["p"], d["c"], dev)
        pr, _ = resp_logp(model, d["p"], d["r"], dev)
        z = a.beta * ((pc - d["rc"]) - (pr - d["rr"]))
        loss = -F.logsigmoid(z) + (a.sft_alpha * (-pc / nc) if a.sft_alpha else 0)
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        run += loss.item(); racc += int(z.item() > 0)
        if step % 50 == 0:
            write({"step": step, "train_loss": round(run / 50, 4), "train_acc": round(racc / 50, 3), "sec": round(time.time() - t0, 1),
                   "mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)})
            run, racc = 0.0, 0
        if step % a.eval_every == 0 or step == steps:
            acc, m = pair_eval(model, hold, a.beta, dev)
            model.save_pretrained(out / f"ckpt-{step}"); saved.append(step)
            write({"step": step, "hold_acc": round(acc, 3), "hold_margin": round(m, 4), "sft_val": round(sft_val(model, val, dev), 4)})

    pick = saved[-1] if a.merge == "last" else int(a.merge)
    del model, opt; torch.cuda.empty_cache()
    base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16)
    v3 = PeftModel.from_pretrained(base, a.init_lora).merge_and_unload()
    merged = PeftModel.from_pretrained(v3, out / f"ckpt-{pick}").merge_and_unload()
    merged.save_pretrained(out / "merged", safe_serialization=True); tok.save_pretrained(out / "merged")
    write({"merged": str(out / "merged"), "from": f"ckpt-{pick}"})


if __name__ == "__main__":
    main()
