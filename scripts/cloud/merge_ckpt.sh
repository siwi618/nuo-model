#!/bin/bash
# 2026-11-21：把某个 run 的指定检查点在 CPU 上合并 → MLX 4 位 → 打包到 download/（无卡模式能跑）
# 用法：bash merge_ckpt.sh run名 步数 [底座=Qwen3.5-2B]   例：bash merge_ckpt.sh q35-2b-full-heavy-lr2e5 600
# 产物：/root/autodl-tmp/download/<run>-k<步数>-mlx4.tgz，md5 记在 nuo/merge.done
set -e
export PATH=/root/miniconda3/bin:$PATH
T=/root/autodl-tmp; run=$1; step=$2; base=${3:-Qwen3.5-2B}
name=$run-k$step; m=$T/runs/$run/merged-k$step
python - $T/models/$base $T/runs/$run/ckpt-$step $m <<'PY'
import sys, json, shutil, torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
b, ck, out = sys.argv[1:]
torch.set_num_threads(32)
base = AutoModelForCausalLM.from_pretrained(b, dtype=torch.bfloat16)
PeftModel.from_pretrained(base, ck).merge_and_unload().save_pretrained(out, safe_serialization=True)
AutoTokenizer.from_pretrained(b).save_pretrained(out)
p = out + "/config.json"; c = json.load(open(p))
if c.get("model_type") == "qwen3_5_text":
    shutil.copy(p, p.replace("config.json", "config.hf.json")); c["model_type"] = "qwen3_5"; json.dump(c, open(p, "w"), indent=2)
PY
python -m mlx_lm convert --hf-path $m --mlx-path $T/download/$name-mlx4 -q --q-bits 4 > $T/nuo/convert-$name.log 2>&1
cd $T/download && tar -czf $name-mlx4.tgz $name-mlx4 && rm -rf $name-mlx4 $m
md5sum $name-mlx4.tgz >> $T/nuo/merge.done; stat -c %s $name-mlx4.tgz >> $T/nuo/merge.done
