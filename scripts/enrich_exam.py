"""素材卷：给考卷卡的「知道」加常设素材（DeepSeek 写，Claude 不写）。

11-25 小试的 6 张（data/enrich/专项-加素材.jsonl）+ 这里新加的 6 张 = 素材卷 12 张。
新加的照第 4 批的写法：5–8 条、每条 ≤25 字，只有 1–2 条和这一场有关。

用法：
  .venv/bin/python scripts/enrich_exam.py write     # 写新 6 张 → data/enrich/素材卷-新6.jsonl
  .venv/bin/python scripts/enrich_exam.py merge     # 合并 12 张 → data/enrich/素材卷.jsonl
"""

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deepseek import chat_json                     # noqa: E402
from render_card import render_card                # noqa: E402
from review import PUNCT, longest_common           # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/enrich"
MODEL = "deepseek-flash"
# 老师赢学生 5–6 / 6、输在平淡 / 漏信息 / 称呼 / 和场景打架的卡（data/cloud-9b/rj2-*.jsonl），覆盖 T1–T3
NEW = ["exam-H01", "exam-H02", "exam-H05", "inner-07", "inner-10", "inner-14"]
ECHO = 10

SYS = """你在给一个「演员模型」的考试卡补设定。卡里「知道」一栏是说话人此刻「提得到的」常设事实，游戏会按人物身份填进去。
你只补「知道」，不改卡上任何别的东西，也不写台词。"""

NEED = """下面是一张戏卡。请给说话人补 5–8 条「提得到的」常设素材，输出 JSON：{"knows_more": [...], "relevant": [...]}
- 每条一件具体的事（职业流程、物品、价目、作息、地点、熟人的习惯、最近发生过的小事……），每条不超过 25 字，写成事实，不写成台词。
- **其中只有 1–2 条和这一场有关**，其余合理、符合他的身份和世界，但和这一场无关。relevant 里原样抄出和这一场有关的那 1–2 条。
- 不能把「要办成」的话、要报的数 / 暗号 / 原话写进素材；不能和档案里已有的「知道」重复；不能透露他「不能说」的秘密；不能和卡上已有的事实矛盾。
- 用这个世界里的词，不出现世界外的词。

【戏卡】
"""


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def exam_cards():
    cards = {}
    for f in ("exam/考卷.jsonl", "exam/专项-心里话.jsonl"):
        for s in load(ROOT / f):
            cards[s["id"]] = s
    return cards


def write():
    cards, out, cost = exam_cards(), [], [0, 0]
    for i in NEW:
        s = copy.deepcopy(cards[i])
        base = s["card"]["speaker"].get("knows") or ""
        base = "；".join(base) if isinstance(base, list) else base
        for attempt in range(3):
            data, u = chat_json(SYS, NEED + render_card(s["card"]), model=MODEL)
            cost[0] += u.get("prompt_tokens", 0)
            cost[1] += u.get("completion_tokens", 0)
            more = [str(x).strip().rstrip("。；;") for x in data.get("knows_more") or [] if str(x).strip()]
            more = [m for m in more if longest_common(PUNCT.sub("", m), PUNCT.sub("", base)) < ECHO]
            if 5 <= len(more) <= 8 and all(len(m) <= 30 for m in more):
                break
        else:
            print(i, "三次都不合格", more)
        s["card"]["speaker"]["knows"] = "；".join([base.rstrip("。；;")] + more) + "。" if base else "；".join(more) + "。"
        s["meta"] = {**s.get("meta", {}), "enrich": {"knows_more": more, "relevant": data.get("relevant", []),
                                                    "by": MODEL, "date": "2026-11-26"}}
        s["source"] = s.get("source", "") + "+素材"
        out.append(s)
        print(f"## {i}  有关：{' / '.join(data.get('relevant', []))}")
        for m in more:
            print("  -", m)
    with open(OUT / "素材卷-新6.jsonl", "w", encoding="utf-8") as f:
        for s in out:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"token 入 {cost[0]} 出 {cost[1]}")


def merge():
    old = load(OUT / "专项-加素材.jsonl")
    new = load(OUT / "素材卷-新6.jsonl")
    with open(OUT / "素材卷.jsonl", "w", encoding="utf-8") as f:
        for s in old + new:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"素材卷 {len(old) + len(new)} 张：", " ".join(s["id"] for s in old + new))


if __name__ == "__main__":
    {"write": write, "merge": merge}[sys.argv[1]]()
