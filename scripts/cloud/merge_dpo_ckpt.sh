#!/bin/bash
# 2026-11-22：DPO 早期检查点合并（DPO 的 LoRA 是加在 v3 合并后的模型上的）→ MLX 4 位 → 打包到 download/
# 只用 CPU（有卡模式约 120 GB 内存；无卡模式只有 2 GB，跑不了）
# 用法：bash merge_dpo_ckpt.sh run名 步数 [更多「run名 步数」...]
#   例：bash merge_dpo_ckpt.sh q35-2b-v3-dpo 100 q35-2b-v3-dpo-s1 100
# 产物：/root/autodl-tmp/download/<run>-k<步数>-mlx4.tgz，md5 和大小记在 nuo/early.done
export PATH=/root/miniconda3/bin:$PATH CUDA_VISIBLE_DEVICES=
T=/root/autodl-tmp
log() { echo "$* $(date +%H:%M)" >> $T/nuo/early.done; }
while [ $# -ge 2 ]; do
  run=$1; step=$2; shift 2
  name=$run-k$step; m=$T/runs/$run/merged-k$step
  python - $T/models/Qwen3.5-2B $T/runs/q35-2b-v3-heavy-lr2e5/ckpt-600 $T/runs/$run/ckpt-$step $m <<'PY'
import sys, json, shutil, torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
b, v3, ck, out = sys.argv[1:]
torch.set_num_threads(32)
base = AutoModelForCausalLM.from_pretrained(b, dtype=torch.bfloat16)
m = PeftModel.from_pretrained(base, v3).merge_and_unload()          # = v3
m = PeftModel.from_pretrained(m, ck).merge_and_unload()             # + DPO 检查点
m.save_pretrained(out, safe_serialization=True)
AutoTokenizer.from_pretrained(b).save_pretrained(out)
p = out + "/config.json"; c = json.load(open(p))
if c.get("model_type") == "qwen3_5_text":
    shutil.copy(p, p.replace("config.json", "config.hf.json")); c["model_type"] = "qwen3_5"; json.dump(c, open(p, "w"), indent=2)
PY
  rc=$?; log "$name merge exit $rc"; [ $rc -ne 0 ] && continue
  python -m mlx_lm convert --hf-path $m --mlx-path $T/download/$name-mlx4 -q --q-bits 4 > $T/nuo/convert-$name.log 2>&1
  log "$name convert exit $?"
  (cd $T/download && tar -czf $name-mlx4.tgz $name-mlx4 && rm -rf $name-mlx4 $m \
    && md5sum $name-mlx4.tgz >> $T/nuo/early.done && stat -c %s $name-mlx4.tgz >> $T/nuo/early.done)
done
log "ALL DONE"
# 合并完等 15 分钟：期间 touch nuo/keep 就不关机（要在同一次开机里加训别的），否则自动关机
for i in $(seq 90); do [ -f $T/nuo/keep ] && { log "有 keep，不关机"; exit 0; }; sleep 10; done
sync; shutdown
