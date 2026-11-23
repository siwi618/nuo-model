"""「语句不通」拆两类补标（2026-11-23）：已被 v4 评审判为语句不通的台词，再问一次 DeepSeek 是哪一类。

  字句不通 = 单看这一句就别扭（用词 / 搭配 / 语法 / 生造词 / 半截话），不看场景也能看出来；
  接不上   = 每句单看通顺，但和场景前情、对方的话、人物关系对不上，或前后句矛盾、跳跃、答非所问。
两类可以都是；都不是 = 补标时觉得其实读得通。
起因：朋友 FOXzzz（玩家视角，不看剧情）判语句不通只有 Ken 的一半，Ken 多判的那些看着像「接不上」。
用途：看第三轮要治的主要是哪一类；以后比版本时也可以分两类报。不改 v4 的结果，单独存文件。

用法（仓库根目录）：
  .venv/bin/python scripts/split_budong.py data/split/split.jsonl \
      --src data/dpo3/judge.jsonl:data/dpo3/cards.jsonl:500 \
      --src "data/cloud-v3/judge-r4.jsonl:exam/考卷.jsonl:0:^(c3|cdpo2s1)r?cut" \
      --human data/human-check/items.json
  --src 是「评审结果:卡文件:抽几条(0 = 全部):文件名正则(可省)」；--human 把人工核对的 40 条单评全标（不管判没判不通），用来和 Ken / 朋友对照。
  可中断，再跑会跳过已标的。
"""
import argparse
import json
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from deepseek import chat_json  # noqa: E402
from render_card import render_card  # noqa: E402

MODEL = "deepseek-flash"

SYS = """你是中文台词审读。下面给你一张戏卡（人物、场景、对方的话等）和这个人物说的台词。请只看「台词读起来通不通」，判断毛病属于哪一类（两类可以都是）：

1. 字句不通：单看这一句就别扭——用词不当、搭配不对、语法错、生造的词或说法、半截话、语序乱、意思含混读不懂。不用看场景也能看出来。
2. 接不上：每句单看都通顺，但放进这张卡里不对——和场景前情、对方刚说的话、人物关系或已知事实对不上；前后句互相矛盾、话题乱跳；答非所问。要读了场景才看得出来。

只看通不通。别管藏没藏住心思、像不像这个人、有没有办成事。括号里的动作和 [表情] 标签不算毛病。
如果你认为其实读得通，两项都填 false。

只输出 JSON：
{"字句不通": true/false, "接不上": true/false, "字句例": "台词里最别扭的那句原文（没有填空串）", "接不上例": "接不上的那句原文（没有填空串）", "说明": "一句话说清毛病"}"""


def label(card, lines):
    body = "\n".join(f"{k}. {s}" for k, s in enumerate(lines, 1)) or "（没写出任何台词）"
    data, u = chat_json(SYS, f"{render_card(card)}\n\n【台词】\n{body}", model=MODEL, thinking=True)
    rec = {k: data.get(k) is True for k in ("字句不通", "接不上")}
    rec.update({k: data.get(k, "") for k in ("字句例", "接不上例", "说明")})
    return rec, u


def load_cards(path):
    return {d["id"]: d["card"] for d in (json.loads(t) for t in Path(path).read_text(encoding="utf-8").splitlines() if t.strip())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--src", action="append", default=[], help="评审结果:卡文件:抽几条:文件名正则")
    ap.add_argument("--human", help="人工核对 items.json（单评题全标）")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    todo = []   # (来源, file, id, card, lines, v4 判了语句不通没有)
    for spec in a.src:
        parts = spec.split(":")
        jpath, cpath, n = parts[0], parts[1], int(parts[2])
        pat = re.compile(parts[3]) if len(parts) > 3 and parts[3] else None
        cards = load_cards(cpath)
        recs = [json.loads(t) for t in Path(jpath).read_text(encoding="utf-8").splitlines() if t.strip()]
        recs = [r for r in recs if r.get("rule") == "v4" and "语句不通" in r["problems"] and r["id"] in cards
                and (pat is None or pat.search(r["file"]))]
        if n:
            recs = random.Random(0).sample(recs, min(n, len(recs)))
        todo += [(jpath, r["file"], r["id"], cards[r["id"]], r["lines"].split("\n"), True) for r in recs]
    if a.human:
        key = json.loads((Path(a.human).parent / "key-v4.json").read_text(encoding="utf-8"))
        exams = {"sp": load_cards(HERE.parent / "exam/专项-心里话.jsonl"), "pub": load_cards(HERE.parent / "exam/考卷.jsonl")}
        for it in json.loads(Path(a.human).read_text(encoding="utf-8")):
            if it["kind"] == "single":   # items.json 里的卡是给人看的格式，原卡按 key 里的 exam / card_id 取
                k = key[it["id"]]
                card = exams["sp" if k["exam"] == "sp" else "pub"][k["card_id"]]
                todo.append((a.human, "human", it["id"], card, it["lines"], "语句不通" in k["problems"]))

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {(d["src"], d["file"], d["id"], d["lines"]) for d in (json.loads(t) for t in out.read_text(encoding="utf-8").splitlines() if t.strip())}
    todo = [t for t in todo if (t[0], t[1], t[2], "\n".join(t[4])) not in done]
    print(f"要标 {len(todo)} 条（已标 {len(done)} 条跳过），模型 {MODEL}", flush=True)

    lock = threading.Lock()
    usage = {"in": 0, "out": 0}
    n = [0]

    def one(t):
        src, fname, i, card, lines, v4bad = t
        r, u = label(card, lines)
        rec = {"src": src, "file": fname, "id": i, "lines": "\n".join(lines), "v4语句不通": v4bad, **r, "model": MODEL}
        with lock:
            with out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            usage["in"] += u.get("prompt_tokens", 0)
            usage["out"] += u.get("completion_tokens", 0)
            n[0] += 1
            if n[0] % 50 == 0:
                print(f"  {n[0]}/{len(todo)}  token 入 {usage['in']} 出 {usage['out']}", flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        for f in [ex.submit(one, t) for t in todo]:
            try:
                f.result()
            except Exception as e:
                print("失败：", e, flush=True)
    print(f"完成 {n[0]} 条；token 入 {usage['in']} 出 {usage['out']}", flush=True)


if __name__ == "__main__":
    main()
