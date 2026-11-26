"""云上（NVIDIA 显卡）LoRA 训练：16 位底座 + LoRA → 选验证损失最低的检查点 → 合并成完整模型。
本机 8 GB Mac 训不了 Qwen3.5-2B（2026-11-19 试训第 1 步空闲内存掉到 4%），训练搬到云上；合并后的模型
在云上或本机用 mlx_lm.convert 压成 4 位，拿回 Mac 考试。

数据：to_mlx.py 产出的 train.jsonl / valid.jsonl（{"messages": [system, user, assistant]}）。
只算台词（assistant）部分的损失；提示词的拼法和本机考试（run_exam.py）完全一致：
  apply_chat_template(system+user, add_generation_prompt=True, enable_thinking=False) + 台词 + <|im_end|>

默认参数照本机 v1 / v2a（mlx-lm）：LoRA 只加在最后 8 层的全部线性层、rank 8、scale 20（alpha = 160）、
dropout 0、Adam lr 1e-4、batch 1、1 轮、最长 2560 token、每 100 步验证一次（前 45 条）并存检查点。

用法：
  python train_lora.py --model 底座目录 --data 数据目录 --out 输出目录 [--epochs 1] [--layers 8] [--rank 8]
输出：out/ckpt-<步数>/（LoRA）、out/best/（验证最低的那份）、out/merged/（合并后的完整 16 位模型）、out/log.jsonl
"""

import argparse
import json
import math
import random
import re
import time
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_jsonl(p):
    return [json.loads(t) for t in Path(p).read_text(encoding="utf-8").splitlines() if t.strip()]


def encode(tok, msgs, max_len):
    prompt = tok.apply_chat_template(msgs[:-1], add_generation_prompt=True, tokenize=False, enable_thinking=False)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tok(msgs[-1]["content"] + "<|im_end|>\n", add_special_tokens=False)["input_ids"]
    ids = (p_ids + a_ids)[:max_len]
    labels = ([-100] * len(p_ids) + a_ids)[:max_len]
    return torch.tensor(ids), torch.tensor(labels)


def loss_of(model, ids, labels, dev):
    """只在台词位置算输出概率（2026-11-25，为 9B 省显存）：词表 24.8 万，全长 2560 的 logits 连梯度要 5 GB 左右。
    第 p 个位置预测第 p+1 个 token，所以取「第一个台词 token 的前一位」到倒数第二位；
    结果和原来 model(labels=...) 的平均交叉熵在数学上一样（同样升到 float32 再算）。"""
    ids, labels = ids.to(dev), labels.to(dev)
    start = int((labels != -100).nonzero()[0])
    pos = torch.arange(start - 1, len(ids) - 1, device=dev)
    logits = model(input_ids=ids[None], logits_to_keep=pos).logits[0].float()
    return torch.nn.functional.cross_entropy(logits, labels[start:], ignore_index=-100)


@torch.no_grad()
def evaluate(model, val, dev):
    model.eval()
    tot, n = 0.0, 0
    for ids, labels in val:
        k = int((labels != -100).sum()) - 1
        tot += loss_of(model, ids, labels, dev).item() * k
        n += k
    model.train()
    return tot / max(n, 1)


def stack(base, loras):
    """依次把逗号分隔的 LoRA 合并进去（每层都是在前一层合并后的模型上训的；同 train_dpo.stack）。"""
    for path in loras.split(","):
        base = PeftModel.from_pretrained(base, path).merge_and_unload()
    return base


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--layers", type=int, default=8, help="只在最后几层加 LoRA；0 = 全部层")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--scale", type=float, default=20.0, help="mlx-lm 的 scale = alpha / rank")
    ap.add_argument("--max-len", type=int, default=2560)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--val-n", type=int, default=45)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-lora", default="", help="先依次合并这些 LoRA（逗号分隔）再加新 LoRA（11-26 第 4 批：在 v3 + dpoa-s1 + dpo2-s1 上接着训）")
    ap.add_argument("--no-merge", action="store_true", help="训完不合并（9B 合并后 18 GB，云盘放不下；考试直接挂 LoRA）")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16)
    if a.init_lora:
        model = stack(model, a.init_lora)
    model = model.to(dev)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False

    n_layers = model.config.get_text_config().num_hidden_layers
    keep = set(range(n_layers - a.layers, n_layers)) if a.layers else set(range(n_layers))
    # peft 不允许 layers_to_transform + "all-linear"：自己列出最后几层里的全部线性层（不含看图部分和 lm_head）
    targets = []
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear) or "visual" in name or "lm_head" in name:
            continue
        m = re.search(r"\.layers\.(\d+)\.", name)
        if m and int(m.group(1)) in keep:
            targets.append(name)
    print(f"LoRA 加在 {len(targets)} 个线性层（第 {min(keep)}–{max(keep)} 层）")
    cfg = LoraConfig(r=a.rank, lora_alpha=a.rank * a.scale, lora_dropout=0.0, target_modules=targets,
                     task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()

    train = [encode(tok, d["messages"], a.max_len) for d in load_jsonl(Path(a.data) / "train.jsonl")]
    val = [encode(tok, d["messages"], a.max_len) for d in load_jsonl(Path(a.data) / "valid.jsonl")][:a.val_n]
    steps = math.ceil(len(train) * a.epochs)
    order = []
    while len(order) < steps:
        idx = list(range(len(train)))
        random.shuffle(idx)
        order += idx
    order = order[:steps]
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    log = (out / "log.jsonl").open("w", encoding="utf-8")

    def write(rec):
        print(json.dumps(rec, ensure_ascii=False), flush=True)
        log.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log.flush()

    v0 = evaluate(model, val, dev)
    write({"step": 0, "val": round(v0, 4), "train_n": len(train), "val_n": len(val), "steps": steps})
    best = (v0, 0)
    t0, run = time.time(), 0.0
    for step, i in enumerate(order, 1):
        ids, labels = train[i]
        loss = loss_of(model, ids, labels, dev)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        run += loss.item()
        if step % 50 == 0:
            write({"step": step, "train": round(run / 50, 4), "sec": round(time.time() - t0, 1),
                   "mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)})
            run = 0.0
        if step % a.eval_every == 0 or step == steps:
            v = evaluate(model, val, dev)
            model.save_pretrained(out / f"ckpt-{step}")
            write({"step": step, "val": round(v, 4)})
            if v < best[0]:
                best = (v, step)
    write({"best_step": best[1], "best_val": round(best[0], 4), "sec": round(time.time() - t0, 1)})

    # 合并最好的那份：重新加载干净的底座 + 最好的 LoRA → 合并 → 存完整模型
    best_dir = out / f"ckpt-{best[1]}" if best[1] else None
    del model, opt
    torch.cuda.empty_cache()
    if best_dir is None:
        print("验证损失一直没降，不合并")
        return
    (out / "best").unlink(missing_ok=True) if (out / "best").is_symlink() else None
    if not (out / "best").exists():
        (out / "best").symlink_to(best_dir.name)
    if a.no_merge:
        return
    base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16)
    if a.init_lora:
        base = stack(base, a.init_lora)
    merged = PeftModel.from_pretrained(base, best_dir).merge_and_unload()
    merged.save_pretrained(out / "merged", safe_serialization=True)
    tok.save_pretrained(out / "merged")
    write({"merged": str(out / "merged"), "from": best_dir.name})


if __name__ == "__main__":
    main()
