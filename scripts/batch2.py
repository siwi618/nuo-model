"""第二批（batch-2，2026-11-19 Ken 定三条改进一起做）。

和 batch-1 一样的世界、角色档案（直接从 data/batch-1 拷，不重写），一样的步骤和给 DeepSeek 的说明，
只加三样：
1. 审稿（scripts/review.py）：戏卡写完先审「目的和秘密打架 / 自相矛盾 / 和档案对不上 / 关系称呼打架」，
   台词过了机器检查再审「心里话说出口 / 说漏秘密 / 没办成 / 称呼人物错」，有问题就带着问题让 DeepSeek 重写；
   另外台词和「心里想的」连续 6 个字一样直接算说出口（机器检查里）。
2. 补少的：按 batch-1 每人留下的条数分配名额，留得少的多做（目标是两批合起来每人差不多）。
3. 卡的比例：「动作：可以写」从 25% 降到 15%（v1 考卷上不该写动作时还在写）；
   「写整段」里对方也能开口从 15% 提到 35%（多人开口的卡太少）。

用法：NUO_N=20 python3 scripts/batch2.py [plan / cards / lines / all]
     NUO_TOTAL=600 改总条数（默认 600）。结果在 data/batch-2/（不进 git）。
"""

import json
import math
import os
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import batch  # noqa: E402  （导入时会把 batch-1 的世界、角色、每卡随机 12 个词的规矩装进 pilot）
import pilot  # noqa: E402
from review import review_card, review_lines  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RUN = os.environ.get("NUO_RUN", "batch-2")
pilot.OUT = ROOT / "data" / RUN
pilot.DATE = "2026-11-19"
PREV = ROOT / "data" / "batch-1"
LIMIT = int(os.environ.get("NUO_N", "20"))
TOTAL = int(os.environ.get("NUO_TOTAL", "600"))
KEEP_RATE = 0.8           # 加了审稿后预计能留下的比例（batch-1 是 75.5%）
MIN_EACH, MAX_EACH = 6, 40
P_ACTIONS = 0.15
P_SPEAKERS_EXTRA = 0.35

pilot.CARD_REVIEW = lambda card: review_card(card, pilot.MODEL, pilot.USAGE)
pilot.LINES_REVIEW = lambda card, out: review_lines(card, out, pilot.MODEL, pilot.USAGE)


def seed_prev():
    """世界、档案沿用 batch-1（考卷 S 组的老角色、名字标签都已对齐）。"""
    for name in ("worlds.json", "people.json"):
        if not (pilot.OUT / name).exists():
            pilot.save(name, json.loads((PREV / name).read_text(encoding="utf-8")))


def quotas():
    """每人做几条：让两批合起来每人留下的条数差不多。"""
    samples = json.loads((PREV / "samples.json").read_text(encoding="utf-8"))
    kept = Counter(s["meta"]["plan"]["speaker"] for s in samples if s["meta"].get("keep"))
    people = list(pilot.PEOPLE)

    def plan_for(target):
        return {p: min(MAX_EACH, max(MIN_EACH, math.ceil((target - kept[p]) / KEEP_RATE))) for p in people}
    target = 0
    while sum(plan_for(target + 1).values()) <= TOTAL:
        target += 1
    q = plan_for(target)
    return q, kept, target


def make_plan(seed=20261119):
    rnd = random.Random(seed)
    q, _, _ = quotas()
    first, rest = [], []
    for who, n in q.items():
        items = batch.per_person(rnd, who, n)
        for it in items:
            it["actions"] = rnd.random() < P_ACTIONS
            it["speakers_extra"] = it["use"] == "scene" and rnd.random() < P_SPEAKERS_EXTRA
        first += items[:1]
        rest += items[1:]
    rnd.shuffle(rest)
    plan = first + rest
    for i, it in enumerate(plan):
        it["id"] = f"b2-{i + 1:03d}"
    return plan


def main():
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    plan = pilot.load("plan.json")
    if not plan:
        plan = make_plan()
        pilot.save("plan.json", plan)
    if step == "plan":
        q, kept, target = quotas()
        print(f"共 {len(plan)} 条；目标每人两批合计约留 {target} 条")
        for p in sorted(q, key=lambda p: kept[p]):
            print(f"  {p}：batch-1 留 {kept[p]}，这批做 {q[p]}")
        w = Counter(it["world"] for it in plan)
        print("按世界：", dict(w.most_common()))
        print("动作可以写：", sum(it["actions"] for it in plan), "；对方也能开口：", sum(it["speakers_extra"] for it in plan))
        return
    plan = plan[:LIMIT]
    seed_prev()
    worlds = pilot.load("worlds.json")
    people = pilot.load("people.json")
    cards = pilot.step_cards(worlds, people, plan)
    if step == "cards":
        return
    samples = pilot.step_lines(worlds, people, plan, cards)
    pilot.step_check(samples)
    tin = sum(u.get("prompt_tokens", 0) for u in pilot.USAGE)
    tout = sum(u.get("completion_tokens", 0) for u in pilot.USAGE)
    print(f"本次调用 {len(pilot.USAGE)} 次，输入 {tin} token，输出 {tout} token")


if __name__ == "__main__":
    main()
