"""把 .jsonl 里的样本渲染成人看的戏卡（格式见 docs/戏卡格式.md）。

用法：
    python3 scripts/render_card.py examples/梅姨.jsonl            # 打印所有卡片
    python3 scripts/render_card.py examples/梅姨.jsonl --out DIR  # 每条存成 DIR/<id>.txt
"""

import argparse
import json
import sys
from pathlib import Path

USE_NAMES = {"scene": "写整段", "reply": "接一句"}


def line(name, emotion, text):
    tag = f"[{emotion}]" if emotion and emotion != "平常" else ""
    return f"{name}{tag}: {text}"


def field(label, value):
    """有内容才输出一行；列表用「；」连起来。"""
    if isinstance(value, list):
        value = "；".join(v for v in value if v)
    return [f"{label}：{value}"] if value else []


def render_card(card):
    w, s, l, sc, g, r = (card[k] for k in ("world", "speaker", "listener", "scene", "goal", "rules"))
    out = ["【世界】"]
    out += field("腔调", w.get("tone"))
    out += field("称呼", w.get("address"))
    out += field("本场的词", "、".join(t for t in w.get("terms", []) if t))
    out += field("不能出现", w.get("banned"))
    out += field("世界外的事", w.get("offstage"))

    out += ["", f"【说话人】{s['name']}"]
    out += field("一句话", s.get("summary"))
    out += field("性格", s.get("personality"))
    out += field("说话", s.get("speech"))
    out += field("例句", "".join(f"「{e}」" for e in s.get("examples", []) if e))
    out += field("想要", s.get("wants"))
    out += field("怕", s.get("fears"))
    out += field("知道", s.get("knows"))
    out += field("不能说", s.get("secrets") or "无")

    out += ["", f"【对方】{l['name']}"]
    out += field("是谁", l.get("summary"))
    out += field(f"{s['name']}怎么看他", l.get("view"))
    out += field("在场", [f"{o['name']}（{o['view']}）" for o in l.get("others", []) if o.get("name")])

    out += ["", "【场景】"]
    out += field("地点时间", sc.get("place_time"))
    out += field("前情", sc.get("backstory"))
    out += field("已发生的事", sc.get("happened"))

    hist = card.get("history") or []
    if hist:
        out += ["", "【前文】"] + [line(h["name"], h.get("emotion"), h["text"]) for h in hist]
    else:
        out += ["", "【前文】无"]

    out += ["", "【目的】"]
    out += field("要办成", g.get("must"))
    out += field("心里想的", g.get("inner"))
    out += field("别说漏", g.get("dont_leak"))

    lo, hi = r["lines"]
    out += ["", "【要求】"]
    out += [f"用法：{USE_NAMES[r['use']]}"]
    out += [f"句数：{lo}–{hi}" if lo != hi else f"句数：{lo}"]
    out += [f"每句最多：{r['max_chars']} 字"]
    out += [f"表情：{'、'.join(r['emotions'])}"]
    out += [f"动作：{'可以写' if r.get('actions') else '不写'}"]
    out += [f"谁能开口：{'、'.join(r['speakers'])}"]
    return "\n".join(out)


def render_output(sample):
    return "\n".join(line(o["name"], o.get("emotion"), o["text"]) for o in sample["output"])


def render_sample(sample):
    card = render_card(sample["card"])
    return f"{card}\n\n——— 输出 ———\n{render_output(sample)}\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--out")
    args = ap.parse_args()
    samples = [json.loads(t) for t in Path(args.jsonl).read_text(encoding="utf-8").splitlines() if t.strip()]
    for smp in samples:
        text = render_sample(smp)
        card_len = len(render_card(smp["card"]))
        if card_len > 1500:
            print(f"提醒：{smp['id']} 卡片 {card_len} 字，超过 1500", file=sys.stderr)
        if args.out:
            Path(args.out).mkdir(parents=True, exist_ok=True)
            Path(args.out, f"{smp['id']}.txt").write_text(text, encoding="utf-8")
        else:
            print(f"===== {smp['id']}（来源 {smp['source']}，卡片 {card_len} 字）\n{text}")


if __name__ == "__main__":
    main()
