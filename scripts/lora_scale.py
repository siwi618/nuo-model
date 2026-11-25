"""LoRA 调淡：本机合并 底座 + α ×（一串 LoRA 的改动之和）→ MLX 4 位。不用训练、不用显卡。（2026-11-25）

为什么在 16 位上做：LoRA 的改动比 4 位量化的误差还小，直接拿两个 4 位模型插值，插出来的多半是量化噪声。
所以从 16 位底座出发，在 float32 里加改动，转回 bf16，再照 mlx_lm convert 的步骤压 4 位（和云上 merge_ckpt.sh 同一套）。

PEFT 的 LoRA 合并是线性的：W' = W + (alpha / r) · B @ A，一层一层叠也是相加。所以
  dpo2-s1 = 底座 + d(v3 ckpt-600) + d(dpoa-s1 ckpt-170) + d(dpo2-s1 ckpt-164)
  调淡    = 底座 + α · (三段之和)
（云上是每叠一段就转一次 bf16，这里一次加完再转，α = 1.0 和云上版本只差舍入。）

用法：
  python scripts/lora_scale.py merge --alpha 0.8 0.65 0.5 [--alpha 1.0 用来核对]
  python scripts/lora_scale.py check models/q35-2b-v3-dpo2-s1-a100-mlx4 models/q35-2b-v3-dpo2-s1-mlx4
"""
import argparse, json, re, shutil
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parent.parent
M = ROOT / "models"
CHAIN = ["q35-2b-v3-heavy-lr2e5_ckpt-600", "q35-2b-v3-dpoa-s1_ckpt-170", "q35-2b-v3-dpo2-s1_ckpt-164"]
REF = M / "q35-2b-v3-dpo2-s1-mlx4"          # 拿它的 config / 分词器，保证和现在考的模型同一套


def load_chain(names):
    """返回 {底座权重名: [(A, B, scale), ...]}"""
    out = {}
    for n in names:
        d = M / "lora-chain" / n
        cfg = json.load(open(d / "adapter_config.json"))
        assert not cfg.get("use_rslora") and not cfg.get("use_dora")
        s = cfg["lora_alpha"] / cfg["r"]
        w = mx.load(str(d / "adapter_model.safetensors"))
        mods = {}
        for k, v in w.items():
            m = re.match(r"base_model\.model\.(.+)\.lora_([AB])(?:\.default)?\.weight$", k)
            assert m, k
            mods.setdefault(m.group(1), {})[m.group(2)] = v
        for mod, ab in mods.items():
            # 云上底座用 AutoModelForCausalLM 读，层名是 model.layers.N…；原版文件里是 model.language_model.layers.N…
            key = mod.replace("model.layers.", "model.language_model.layers.", 1) + ".weight"
            out.setdefault(key, []).append((ab["A"], ab["B"], s))
        print(f"{n}: {len(mods)} 层，scale {s}")
    return out


def merge(alpha, chain, base_dir, out_name):
    """一个进程里做完，不落 16 位临时文件（磁盘紧）：
    临时目录只放 现在模型的 config（去掉量化）+ 分词器 + 底座权重的软链接 → mlx_lm 懒加载（sanitize 照常：去视觉 / MTP、
    norm +1、conv 转轴、改名）→ 线性层 W += α·Σ scale·B@A（float32 里加，转回 bf16）→ 和 convert 一样转 dtype、压 4 位、保存。"""
    from mlx.utils import tree_flatten, tree_map_with_path
    from mlx_lm.utils import load, quantize_model, save

    tmp = M / f"_tmp-{out_name}"
    shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir()
    cfg = json.load(open(REF / "config.json")); cfg.pop("quantization", None); cfg.pop("quantization_config", None)
    json.dump(cfg, open(tmp / "config.json", "w"), indent=2)
    for f in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"]:
        if (REF / f).exists():
            shutil.copy(REF / f, tmp / f)
    for f in base_dir.glob("model*.safetensors"):
        (tmp / f.name).symlink_to(f.resolve())

    model, tok, config = load(str(tmp), lazy=True, return_config=True)
    params = dict(tree_flatten(model.parameters()))
    # 底座名 model.language_model.layers.N.X.weight → mlx 名 language_model.model.layers.N.X.weight
    todo = {k.replace("model.language_model.", "language_model.model.", 1): v for k, v in chain.items()}
    missing = set(todo) - set(params)
    assert not missing, f"LoRA 层在模型里找不到：{sorted(missing)[:5]}"
    upd = []
    for k, ab in todo.items():                                          # 只建计算图，不求值（省内存）
        d = sum(s * (B.astype(mx.float32) @ A.astype(mx.float32)) for A, B, s in ab)
        upd.append((k, (params[k].astype(mx.float32) + alpha * d).astype(mx.bfloat16)))
    del params
    from mlx.utils import tree_unflatten
    model.update(tree_unflatten(upd)); del upd
    print(f"α={alpha}: 改 {len(todo)} 层 → 压 4 位")

    cast = getattr(model, "cast_predicate", lambda _: True)            # 同 convert：除 A_log 外转 bf16
    model.update(tree_map_with_path(
        lambda k, v: v.astype(mx.bfloat16) if cast(k) and mx.issubdtype(v.dtype, mx.floating) else v, model.parameters()))
    model, config = quantize_model(model, config, 64, 4, mode="affine")
    # 一个张量一个张量求值：16 位中间结果算完就释放（11-25 第一次整体求值峰值 8 GB，挤爆交换区和磁盘）
    mx.set_cache_limit(0)
    for i, (k, v) in enumerate(tree_flatten(model.parameters())):
        mx.eval(v)
        if i % 200 == 0:
            print(f"  {i} 峰值 {mx.get_peak_memory() / 2**30:.2f} GB", flush=True)
    out = M / f"{out_name}-mlx4"
    shutil.rmtree(out, ignore_errors=True)
    save(out, tmp, model, tok, config)
    shutil.rmtree(tmp)
    print("→", out)


def check(a, b):
    """逐张量比两个 4 位模型：4 位码完全一样的比例 + 反量化后的相对差。"""
    wa = {}; wb = {}
    for f in sorted(Path(a).glob("model*.safetensors")): wa.update(mx.load(str(f)))
    for f in sorted(Path(b).glob("model*.safetensors")): wb.update(mx.load(str(f)))
    keys = sorted(set(wa) & set(wb))                                  # 只比两边都有的（原版 4 位带视觉部分）
    same = tot = 0; worst = (0, ""); rels = []
    for k in keys:
        if not k.endswith(".weight") or k[:-7] + ".scales" not in wa:
            continue
        p = k[:-7]
        qa = mx.dequantize(wa[k], wa[p + ".scales"], wa[p + ".biases"], 64, 4)
        qb = mx.dequantize(wb[k], wb[p + ".scales"], wb[p + ".biases"], 64, 4)
        rel = (mx.linalg.norm((qa - qb).astype(mx.float32)) / mx.linalg.norm(qb.astype(mx.float32))).item()
        worst = max(worst, (rel, p)); rels.append(rel)
        same += (wa[k] == wb[k]).sum().item(); tot += wa[k].size
    rels.sort()
    print(f"打包的 4 位码（每格 8 个）完全一样：{same / tot:.2%}；相对差 中位 {rels[len(rels) // 2]:.2e}、"
          f"最大 {worst[0]:.2e}（{worst[1]}）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    m = sp.add_parser("merge"); m.add_argument("--alpha", type=float, nargs="+", required=True)
    m.add_argument("--base", default=str(M / "Qwen3.5-2B-hf"))
    c = sp.add_parser("check"); c.add_argument("a"); c.add_argument("b")
    a = ap.parse_args()
    if a.cmd == "merge":
        chain = load_chain(CHAIN)
        for al in a.alpha:
            merge(al, chain, Path(a.base), f"q35-2b-v3-dpo2-s1-a{round(al * 100):03d}")
    else:
        check(a.a, a.b)
