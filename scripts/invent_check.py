"""卡外编造专项评审（2026-11-24，Ken 定边界：①②③ 禁、④ 放）。

接游戏后，NPC 不能跟玩家提游戏里不存在、玩家会当真去找 / 去做的东西：
  ① 约定、指路、时间安排（「明早卯时老槐树底下见」「老细明早七点来巡查」）
  ② 能买、能拿、能得到的东西（「我这儿有几头良驹，随我问问价」）
  ③ 有名有姓的新地点、新人物、新组织（「崇义门口那家裁缝铺」）
  ④ 氛围细节、泛称、自己的往事和感受 —— 放行，不报。
卡里任何一栏（世界 / 档案 / 场景 / 前文 / 目的）出现过的都不算编。

用法：
  .venv/bin/python scripts/invent_check.py model  data/invent/model.jsonl   # dpo2-s1 公开卷种子 0–2 + 专项卷种子 0–1
  .venv/bin/python scripts/invent_check.py train  data/invent/train.jsonl --n 60   # v3 训练集里 DeepSeek 写的台词
  .venv/bin/python scripts/invent_check.py report data/invent/model.jsonl
"""
import argparse
import json
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from deepseek import chat_json  # noqa: E402
from judge_exam import MODEL, number_lines  # noqa: E402
from render_card import render_card  # noqa: E402

KINDS = ["约定指路", "东西", "专名"]

SYS = """你是游戏编剧的审稿人。下面是一张「戏卡」和 NPC 按它写的台词。这段台词要放进游戏里说给玩家听。
游戏里只存在戏卡里写到的东西（任何一栏都算：世界、专有词、档案里的「知道的事」、场景、前文、目的）。
你只查一件事：**台词有没有编出卡里没有、玩家会当真去找或去做的东西**。分三类：
① 约定指路：约好的时间地点（「明早卯时老槐树底下见」）、别人的时间安排（「掌柜明早七点来巡查」）、指玩家去某个具体地方找某人某物、许诺将来会发生的具体事情。
② 东西：玩家能买、能拿、能得到的货物、奖励、服务（「我这儿有几头良驹，随我问问价」「去后院领两斤米」）。
③ 专名：有名有姓的新地点、新人物、新组织、新店铺（「崇义门口那家裁缝铺」「王二麻子」「青龙帮」）。

**不报**（这些是好的细节，放行）：
- 卡里出现过的东西，或者从卡里直接能推出来的（卡说在茶棚里，提「这壶茶」不算编）。
- 泛称、不能去找的说法：「路口那几家」「街坊」「当官的」「前头那条街」「有人」。
- 说话人自己的往事、感受、身体状况、对天气和眼前景象的描写：「我年轻时跑过马」「今儿风大」「喝了两盅」。
- 俗话、比喻、口头禅。
- 卡的【目的】要求做约定，但没给具体时间地点时，台词补了具体时间地点：**照样报①**，另外把「卡缺」写成 true（说明是卡没给够，不全是演员的错）。

逐句看，每处写：哪一句（〔n〕里的号）、原话（从台词里一字不改地照抄，越短越好）、类（约定指路 / 东西 / 专名）、卡缺（true / false）、说明（一句话：玩家会怎么当真）。
没有就给空列表。只输出 JSON：
{"编造": [{"句": 1, "原话": "…", "类": "专名", "卡缺": false, "说明": "…"}]}"""


def check_one(card_text, lines, model=MODEL):
    shown, _ = number_lines(lines)
    data, u = chat_json(SYS, f"{card_text}\n\n【台词】\n{shown}", model=model, thinking=True)
    items = data.get("编造", []) if isinstance(data, dict) else []
    body = "\n".join(lines)
    for e in items:
        e["quote_ok"] = bool(e.get("原话")) and e["原话"] in body
    return items, u


def model_items():
    sys.path.insert(0, str(HERE))
    from human_check_build import load_exam
    exams = {"judge-r4.jsonl": load_exam(ROOT / "exam" / "考卷.jsonl"),
             "judge-sp9-r4.jsonl": load_exam(ROOT / "exam" / "专项-心里话.jsonl")}
    want = {"judge-r4.jsonl": ["cdpo2s1cut-pen-s0.txt", "cdpo2s1cut-pen-s1.txt", "cdpo2s1cut-pen-s2.txt"],
            "judge-sp9-r4.jsonl": ["sp9-cdpo2s1cut-pen-s0.txt", "sp9-cdpo2s1cut-pen-s1.txt"]}
    out = []
    for fn, files in want.items():
        for t in (ROOT / "data" / "cloud-v3" / fn).read_text(encoding="utf-8").splitlines():
            r = json.loads(t)
            if r["file"] in files and r["lines"].strip():
                out.append({"src": r["file"], "card": r["id"], "lines": r["lines"].strip().split("\n"),
                            "card_text": render_card(exams[fn][r["id"]])})
    return out


def train_items(n, seed=20261124):
    rows = [json.loads(t) for t in (ROOT / "data" / "v3" / "mlx-v3" / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    man = json.loads((ROOT / "data" / "v3" / "manifest.json").read_text(encoding="utf-8"))["mlx-v3"]
    idx = random.Random(seed).sample(range(len(rows)), n)
    out = []
    for i in idx:
        m = {x["role"]: x["content"] for x in rows[i]["messages"]}
        out.append({"src": man[i] if i < len(man) else "?", "card": f"train-{i}",
                    "lines": m["assistant"].strip().split("\n"), "card_text": m["user"]})
    return out


def run(items, path):
    use = {"in": 0, "out": 0}

    def one(it):
        try:
            e, u = check_one(it["card_text"], it["lines"])
        except Exception as ex:  # 坏一条不拖垮整批
            return {**{k: it[k] for k in ("src", "card", "lines")}, "error": str(ex)}, {}
        return {**{k: it[k] for k in ("src", "card", "lines")}, "编造": e}, u
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(one, items))
    for _, u in res:
        use["in"] += u.get("prompt_tokens", 0)
        use["out"] += u.get("completion_tokens", 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r, _ in res), encoding="utf-8")
    print(f"查完 {len(res)} 条，token 入 {use['in']} 出 {use['out']}")


def report(path):
    R = [json.loads(t) for t in path.read_text(encoding="utf-8").splitlines()]
    bad = [r for r in R if "error" in r]
    R = [r for r in R if "error" not in r]
    E = [e for r in R for e in r["编造"]]
    hit = [r for r in R if r["编造"]]
    print(f"{path.name}：{len(R)} 条（出错 {len(bad)}），有编造 {len(hit)} 条 = {len(hit) / len(R):.0%}；共 {len(E)} 处，引文对得上 {sum(e['quote_ok'] for e in E)}")
    for k in KINDS:
        rk = [r for r in R if any(e.get("类") == k for e in r["编造"])]
        print(f"  {k}：{len(rk)} 条（{len(rk) / len(R):.0%}），{sum(e.get('类') == k for e in E)} 处")
    print(f"  其中「卡缺」（卡要约定却没给细节）：{sum(bool(e.get('卡缺')) for e in E)} 处")
    print("  类别外：", Counter(e.get("类") for e in E if e.get("类") not in KINDS))
    for r in hit:
        for e in r["编造"]:
            print(f"   {r['card']}〔{e.get('句')}〕{e.get('类')}{'·卡缺' if e.get('卡缺') else ''}「{e.get('原话')}」— {e.get('说明')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["model", "train", "report"])
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    p = Path(a.out)
    if a.what == "report":
        return report(p)
    items = model_items() if a.what == "model" else train_items(a.n)
    run(items[:a.limit] if a.limit else items, p)
    report(p)


if __name__ == "__main__":
    main()
