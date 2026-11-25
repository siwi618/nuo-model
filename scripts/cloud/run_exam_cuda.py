"""云上（NVIDIA）考试：run_exam.py 的 CUDA 版，给蒸馏前「老师够不够顺」用（2026-11-25）。

为什么要有：老师 Qwen3.5-9B 本机 8 GB 跑不动，只能在云上考；为了同流程比较，学生 dpo2-s1 也在云上用本脚本
（16 位合并版）再考一遍，不拿本机 MLX 4 位的结果直接比（11-25 LoRA 调淡发现：流程不同，同种子台词只 22% 一样）。

和 run_exam.py 保持一致的地方：
  - 提示词：system = pilot.SYS_LINES，user = render_card(card)，apply_chat_template(enable_thinking=False)
    （9B / 27B 的模板默认开思考，2B 默认关；这里显式关，三者拼出来的提示一样）
  - 重复惩罚：只罚模型自己写出来的最近 rep_context 个 token，不罚名字 / 表情 / 格式符号；罚法同 mlx-lm
  - 采样：温度 + top_p（+ 可选 top_k），每张卡、每个种子单独设种子
  - 输出格式：输出前缀-s<种子>.txt，块头「=== id（说话人）」，可以直接喂 read_judge.py vs / score_exam.py
不一致（无法一致）：随机数发生器不同，同种子和本机 MLX 的台词不会一样；只在本脚本的输出之间配对比较。

用法（仓库根目录）：
  python scripts/cloud/run_exam_cuda.py 模型目录 输出前缀 [--exam exam/考卷.jsonl] [--tier 困难]
      [--seeds 0,1,2] [--temp 0.3 --top-p 0.9 --top-k 0 --rep-penalty 1.15]
"""

import argparse
import ast
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from render_card import render_card  # noqa: E402

# pilot.py 顶上 import 了 deepseek / review（云上用不到、也可能缺包），这里只从源码里读出 SYS_LINES 常量，不执行它
SYS_LINES = next(ast.literal_eval(n.value) for n in ast.parse((HERE.parent / "pilot.py").read_text(encoding="utf-8")).body
                 if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "SYS_LINES" for t in n.targets))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList  # noqa: E402


def format_ids(tok, card):
    """同 run_exam.format_ids：名字、表情、格式符号的 token 每行都要重复写，不能罚。"""
    r = card["rules"]
    words = list(r.get("speakers", [])) + list(r.get("emotions", [])) + ["[", "]", ":", "：", "\n", "（", "）"]
    out = set()
    for w in words:
        for pre in ("", "\n"):
            out.update(tok.encode(pre + w, add_special_tokens=False))
    return sorted(out)


class GenOnlyPenalty(LogitsProcessor):
    """同 run_exam.gen_only_penalty：只罚生成部分最近 context 个 token，豁免 exempt。"""

    def __init__(self, prompt_len, exempt, penalty, context):
        self.prompt_len, self.penalty, self.context = prompt_len, penalty, context
        self.exempt = torch.tensor(exempt, dtype=torch.long)

    def __call__(self, input_ids, scores):
        gen = input_ids[:, self.prompt_len:][:, -self.context:]
        if gen.shape[1] == 0:
            return scores
        hit = torch.zeros_like(scores, dtype=torch.bool)
        hit.scatter_(1, gen, True)
        hit[:, self.exempt.to(scores.device)] = False
        pen = torch.where(scores < 0, scores * self.penalty, scores / self.penalty)
        return torch.where(hit, pen, scores)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--exam", default=str(HERE.parent.parent / "exam" / "考卷.jsonl"))
    ap.add_argument("--adapter", help="LoRA 目录，可逗号分隔叠几段；不落盘，在显存里合并")
    ap.add_argument("--tier",help="只考这一档（公开卷 meta.tier：简单 / 中等 / 困难）")
    ap.add_argument("--ids", help="只考这几张，逗号分隔")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--temp", type=float, default=0.3)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--rep-penalty", type=float, default=1.15)
    ap.add_argument("--rep-context", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=300)
    a = ap.parse_args()

    exam = [json.loads(t) for t in Path(a.exam).read_text(encoding="utf-8").splitlines() if t.strip()]
    if a.tier:
        exam = [e for e in exam if e.get("meta", {}).get("tier") == a.tier]
    if a.ids:
        want = set(a.ids.split(","))
        exam = [e for e in exam if e["id"] in want]
    seeds = [int(s) for s in a.seeds.split(",")]

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda")
    for ad in (a.adapter or "").split(","):  # 可以叠几段 LoRA（逗号分隔，按顺序合并进显存里的模型，不落盘）
        if ad:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, ad).merge_and_unload()
    model = model.eval()
    load_s = time.time() - t0
    eos = [tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id]
    eos = sorted({i for i in eos if i is not None})

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    totals, ntoks = [], []
    for seed in seeds:
        blocks = []
        for e in exam:
            msgs = [{"role": "system", "content": SYS_LINES}, {"role": "user", "content": render_card(e["card"])}]
            prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
            ids = tok.encode(prompt, add_special_tokens=False)
            x = torch.tensor([ids], device="cuda")
            procs = LogitsProcessorList()
            if a.rep_penalty and a.rep_penalty != 1.0:
                procs.append(GenOnlyPenalty(len(ids), format_ids(tok, e["card"]), a.rep_penalty, a.rep_context))
            torch.manual_seed(seed)
            t = time.time()
            with torch.no_grad():
                y = model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=a.max_tokens,
                                   do_sample=a.temp > 0, temperature=a.temp or None, top_p=a.top_p,
                                   top_k=a.top_k or None, repetition_penalty=None, logits_processor=procs,
                                   eos_token_id=eos, pad_token_id=eos[0])
            totals.append(time.time() - t)
            new = y[0, len(ids):].tolist()
            ntoks.append(len(new))
            text = tok.decode(new, skip_special_tokens=True)
            blocks.append(f"=== {e['id']}（{e['card']['speaker']['name']}）\n{text.strip()}")
        Path(f"{out}-s{seed}.txt").write_text("\n".join(blocks) + "\n", encoding="utf-8")
        print(f"seed {seed} done {time.time() - t0:.0f}s", flush=True)

    summary = (f"{len(exam)} 张 × {len(seeds)} 个种子 | 加载 {load_s:.1f}s | 整句 中位 {sorted(totals)[len(totals) // 2]:.2f}s"
               f" | {sum(ntoks) / sum(totals):.0f} tok/s | 显存峰值 {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
    Path(f"{out}.timing").write_text(summary + "\n", encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
