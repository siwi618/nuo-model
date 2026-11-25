"""云上在线蒸馏（On-Policy Distillation，2026-11-25）：学生 dpo2-s1 自己写台词，训后 9B 老师给每个词打分，学生往老师那边靠。

做法照 Thinking Machines《On-Policy Distillation》（参考文献子文档）：
  - 每步取 P 张戏卡，学生（当前策略）每张写 K 条（温度 1、不截断分布：采样分布 = 训练时算的分布）；
  - 老师只对学生写出来的那个词给对数概率 logp_T（不算整个词表，省显存）；
  - 每个词的「优势」A = logp_T − logp_S（= −每词 reverse KL，不打折扣、不往后传），
    损失 = −Σ A.detach() · logp_S / 总词数（策略梯度；一步一采样，完全在线，不用重要性采样）；
  - 学生 = 底座 + 三段 LoRA（v3 → dpoa-s1 → dpo2-s1）在内存里合并，再加一层新 LoRA（全部层，r16）只训这层；
  - 老师 = 9B 底座 + SFT LoRA（ckpt-500）合并。
显存（4090 24 GB）：老师 16 位 18 GB + 学生 4.5 GB 放不下一起，所以两边冻住的权重都在锁页内存里留一份，
  用时把指针换上显卡、用完换回（不往回拷），每步换两次，各约 1–2 秒。
提示词拼法和 train_lora.py / run_exam.py 一致（apply_chat_template + enable_thinking=False；台词以 <|im_end|> 收尾）。

用法（云上 nuo/ 目录）：
  自检（只用学生，不要老师）：python opd.py --check ...（见 main 的参数）
  训练：python opd.py --student-base 2B --student-loras a,b,c --teacher 9B --teacher-lora ckpt-500 --data train.jsonl --hold valid.jsonl --out 输出
输出：out/log.jsonl（每步 reverse KL、长度、各阶段用时）、out/ckpt-<步>/（新 LoRA）、out/samples-<步>.jsonl（抽几条学生写的看）
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


def load_merged(base, loras):
    m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
    for l in [x for x in (loras or "").split(",") if x]:
        m = PeftModel.from_pretrained(m, l).merge_and_unload()
    return m


class Swap:
    """冻住的参数 / 缓冲区：锁页内存里留一份，gpu() 拷上显卡，cpu() 只把指针指回内存那份（不往回拷）。"""

    def __init__(self, model):
        self.items, seen = [], set()   # 去重：2B 的输入 / 输出层共用一份权重（tie_word_embeddings），别拷两遍
        for _, mod in model.named_modules():
            ts = [p for _, p in mod.named_parameters(recurse=False) if not p.requires_grad]
            ts += [b for _, b in mod.named_buffers(recurse=False) if b is not None]
            for t in ts:
                if id(t) not in seen:
                    seen.add(id(t))
                    self.items.append((t, t.data.contiguous().pin_memory()))
        for t, c in self.items:
            t.data = c

    def gpu(self):
        for t, c in self.items:
            t.data = c.to("cuda", non_blocking=True)
        torch.cuda.synchronize()

    def cpu(self):
        for t, c in self.items:
            t.data = c
        torch.cuda.empty_cache()


def enc_prompt(tok, msgs):
    p = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    return tok(p, add_special_tokens=False)["input_ids"]


def resp_logp(model, p_ids, r_ids):
    """bs = 1、不补齐：返回台词每个词的对数概率（只在台词位置算输出概率）。"""
    ids = torch.tensor([p_ids + r_ids], device="cuda")
    pos = torch.arange(len(p_ids) - 1, len(p_ids) + len(r_ids) - 1, device="cuda")
    logits = model(input_ids=ids, logits_to_keep=pos).logits[0].float()
    return torch.log_softmax(logits, -1).gather(1, torch.tensor(r_ids, device="cuda")[:, None])[:, 0]


@torch.no_grad()
def sample(model, tok, prompts, k, max_new, eos, temp, scores=False, bs=64):
    """每张卡写 k 条，多张卡左侧补齐后一起生成（bs 条一批）。返回 [(卡序号, p_ids, r_ids, 采样时的 logp 或 None)]。
    11-25 冒烟：一张卡一批（4 条）每卡 14 秒——没装 fla，线性注意力走 PyTorch 参考实现，一个词一个词地发小算子，
    慢在调度不在算力，所以条数加到 64 几乎不加时间。补齐对不对靠 --check 验（采样 logp vs 不补齐的前向 logp）。"""
    rows = [(i, p) for i, p in prompts for _ in range(k)]
    out = []
    for b in range(0, len(rows), bs):
        chunk = rows[b:b + bs]
        L = max(len(p) for _, p in chunk)
        x = torch.full((len(chunk), L), eos, dtype=torch.long)
        m = torch.zeros((len(chunk), L), dtype=torch.long)
        for j, (_, p) in enumerate(chunk):
            x[j, L - len(p):] = torch.tensor(p)
            m[j, L - len(p):] = 1
        g = model.generate(x.cuda(), attention_mask=m.cuda(), do_sample=True, temperature=temp, top_p=1.0, top_k=0,
                           min_p=None, repetition_penalty=1.0, max_new_tokens=max_new, eos_token_id=eos,
                           pad_token_id=eos, use_cache=True, output_scores=scores, return_dict_in_generate=True)
        seqs = g.sequences[:, L:].tolist()
        for j, s in enumerate(seqs):
            r = s[:s.index(eos) + 1] if eos in s else s
            lp = None
            if scores:
                lp = [torch.log_softmax(g.scores[t][j].float(), -1)[r[t]].item() for t in range(len(r))]
            out.append((chunk[j][0], chunk[j][1], r, lp))
    return out


def lora_targets(model):
    return [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and "visual" not in n
            and "lm_head" not in n and re.search(r"\.layers\.\d+\.", n)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--student-base", required=True)
    ap.add_argument("--student-loras", default="", help="逗号分隔，按顺序合并（v3 ckpt-600,dpoa-s1 ckpt-170,dpo2-s1 ckpt-164）")
    ap.add_argument("--teacher")
    ap.add_argument("--teacher-lora", default="")
    ap.add_argument("--data", required=True, help="训练卡（mlx-v3/train.jsonl，只用 system + user）")
    ap.add_argument("--hold", help="留出卡（valid.jsonl）：每 eval-every 步各写 1 条只算 KL、不训")
    ap.add_argument("--hold-n", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", type=int, default=16, help="每步几张卡")
    ap.add_argument("--k", type=int, default=4, help="每张卡写几条")
    ap.add_argument("--steps", type=int, default=0, help="0 = 训练卡过一遍")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--scale", type=float, default=20.0)
    ap.add_argument("--max-new", type=int, default=300)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true", help="自检：采样时的 logp vs 训练前向的 logp（只用学生），然后退出")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.student_base)
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    cards = [enc_prompt(tok, d["messages"][:-1]) for d in load_jsonl(a.data)]
    hold = [enc_prompt(tok, d["messages"][:-1]) for d in load_jsonl(a.hold)][:a.hold_n] if a.hold else []

    t0 = time.time()
    student = load_merged(a.student_base, a.student_loras)
    student.gradient_checkpointing_enable()
    student.enable_input_require_grads()
    student = get_peft_model(student, LoraConfig(r=a.rank, lora_alpha=a.rank * a.scale, lora_dropout=0.0,
                                                 target_modules=lora_targets(student), task_type="CAUSAL_LM"))
    student.print_trainable_parameters()
    for p in student.parameters():          # 新 LoRA 常驻显卡
        if p.requires_grad:
            p.data = p.data.to("cuda")
    s_swap = Swap(student)
    print(f"学生加载 {time.time() - t0:.0f}s", flush=True)

    if a.check:
        s_swap.gpu()
        student.eval()
        rows = sample(student, tok, [(i, cards[i]) for i in range(4)], 2, a.max_new, eos, a.temp, scores=True)
        print(f"补齐一起生成 4 张 × 2 条（不同提示长度）", flush=True)
        tg = time.time()
        sample(student, tok, [(i, cards[i]) for i in range(16)], 4, a.max_new, eos, a.temp)
        print(f"GENTIME 16 张 × 4 条 = {time.time() - tg:.0f}s", flush=True)
        worst = 0.0
        with torch.no_grad():
            for i, p_ids, r, lp in rows:
                lt = resp_logp(student, p_ids, r).tolist()
                d = max(abs(x - y) for x, y in zip(lp, lt))
                worst = max(worst, d)
                print(f"卡 {i}：{len(r)} 词，采样 logp 合计 {sum(lp):.2f}，前向 {sum(lt):.2f}，单词最大差 {d:.3f}"
                      f" | {tok.decode(r)[:60]!r}", flush=True)
        print(f"CHECK worst={worst:.3f}（bf16 下 ≲ 0.1 算一致；差很大说明采样被截断或拼法不同）")
        return

    t0 = time.time()
    teacher = load_merged(a.teacher, a.teacher_lora).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    t_swap = Swap(teacher)
    print(f"老师加载 {time.time() - t0:.0f}s", flush=True)

    order = list(range(len(cards)))
    random.shuffle(order)
    steps = a.steps or math.ceil(len(cards) / a.prompts)
    opt = torch.optim.Adam([p for p in student.parameters() if p.requires_grad], lr=a.lr)
    log = (out / "log.jsonl").open("a", encoding="utf-8")

    def write(rec):
        print(json.dumps(rec, ensure_ascii=False), flush=True)
        log.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log.flush()

    write({"start": time.strftime("%H:%M:%S"), "cards": len(cards), "steps": steps, "prompts": a.prompts, "k": a.k,
           "lr": a.lr, "rank": a.rank, "scale": a.scale, "temp": a.temp, "seed": a.seed})
    for step in range(1, steps + 1):
        tt = {}
        idx = [order[((step - 1) * a.prompts + j) % len(order)] for j in range(a.prompts)]
        # ① 学生写
        t = time.time()
        s_swap.gpu()
        student.eval()
        rows = sample(student, tok, [(i, cards[i]) for i in idx], a.k, a.max_new, eos, a.temp)
        do_hold = hold and (step == 1 or step % a.eval_every == 0 or step == steps)
        hrows = sample(student, tok, list(enumerate(hold)), 1, a.max_new, eos, a.temp) if do_hold else []
        s_swap.cpu()
        tt["gen"] = time.time() - t
        # ② 老师打分
        t = time.time()
        t_swap.gpu()
        with torch.no_grad():
            lpt = [resp_logp(teacher, p, r) for _, p, r, _ in rows + hrows]
        t_swap.cpu()
        tt["teacher"] = time.time() - t
        # ③ 学生学
        t = time.time()
        s_swap.gpu()
        student.train()
        n_tok = sum(len(r) for _, _, r, _ in rows)
        kl_sum, kls = 0.0, []
        for (_, p, r, _), lt in zip(rows, lpt[:len(rows)]):
            ls = resp_logp(student, p, r)
            adv = (lt - ls).detach()
            (-(adv * ls).sum() / n_tok).backward()
            kl_sum += float((ls.detach() - lt).sum())
            kls.append(float((ls.detach() - lt).mean()))
        gn = float(torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], a.clip))
        opt.step()
        opt.zero_grad(set_to_none=True)
        hold_kl = None
        if hrows:
            student.eval()
            with torch.no_grad():
                hk = [float((resp_logp(student, p, r) - lt).sum()) for (_, p, r, _), lt in zip(hrows, lpt[len(rows):])]
            hold_kl = sum(hk) / sum(len(r) for _, _, r, _ in hrows)
        s_swap.cpu()
        tt["train"] = time.time() - t
        lens = [len(r) for _, _, r, _ in rows]
        write({"step": step, "kl_tok": round(kl_sum / n_tok, 4), "kl_seq_med": round(sorted(kls)[len(kls) // 2], 4),
               "hold_kl_tok": None if hold_kl is None else round(hold_kl, 4), "len": round(sum(lens) / len(lens), 1),
               "cut": sum(r[-1] != eos for _, _, r, _ in rows), "gnorm": round(gn, 3),
               **{k: round(v, 1) for k, v in tt.items()}, "mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)})
        if step % a.save_every == 0 or step == steps:
            student.save_pretrained(out / f"ckpt-{step}")
            with open(out / f"samples-{step}.jsonl", "w", encoding="utf-8") as f:
                for (i, _, r, _), lt in list(zip(rows, lpt))[:8]:
                    f.write(json.dumps({"card": i, "text": tok.decode(r, skip_special_tokens=True),
                                        "teacher_logp_tok": round(float(lt.mean()), 3)}, ensure_ascii=False) + "\n")
    write({"done": time.strftime("%H:%M:%S")})


if __name__ == "__main__":
    main()
