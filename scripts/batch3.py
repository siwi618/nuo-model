"""第三批（batch-3，2026-11-20 Ken 定）：专门改「心里话说出口」。

和 batch-2 一样的世界、角色档案、审稿（scripts/review.py，标准不放宽）、动作 15% / 对方开口 35%，只改三处：
1. 写卡：约六成的卡（plan 里 inner_style = "hide"）额外要求「心里想的」写成说出来会伤人、露怯或坏事的心思，
   而且不能是办成「要办成」的理由，但要和这场谈的话题挨得近（和专项卷同一个思路）；其余照旧（"normal"）。
2. 写台词：给 DeepSeek 的那一次请求后面另加一句提醒（GEN_NOTE：先想表面上说什么，心思只从语气透出来）。
   **系统说明 SYS_LINES 一个字不动**；存进样本的戏卡也不带这句，训练和考试看到的输入和以前一样。
3. 重写：因为「心里话说出口」被打回（字面检查 inner_echo 或审稿）时，**从头重写、不告诉上一版哪里错**
   （11-19 验收：带着问题重写，DeepSeek 多半只换个说法）；其他问题照旧带着问题重写。

2026-11-20 Ken 选省钱方案③：**DeepSeek 不审稿**（审稿带思考，占输出九成），只留机器检查（格式、心里话字面检查）；
Claude 挑稿（只决定留 / 不留），不合格的写进 data/batch-3/redo.json，跑 redo 让 DeepSeek 从头重写一次（不给任何意见）。
NUO_DS_REVIEW=1 可以把 DeepSeek 审稿开回来。

用法：NUO_N=60 .venv/bin/python scripts/batch3.py [plan / cards / lines / all / redo / report]
     NUO_TOTAL=600 改总条数（默认 60，小批试跑）。结果在 data/batch-3/（不进 git）。
     跑完：.venv/bin/python scripts/batch3.py report   看「心里话」相关的数字，和 batch-2 比。
"""

import json
import os
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import batch2  # noqa: E402  （导入时装好 batch-1 的世界 / 角色 / 每卡随机 12 词，和 batch-2 的审稿钩子）
import pilot  # noqa: E402
from deepseek import chat  # noqa: E402
from render_card import render_card  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
pilot.OUT = ROOT / "data" / os.environ.get("NUO_RUN", "batch-3")
pilot.DATE = "2026-11-20"
if os.environ.get("NUO_DS_REVIEW") != "1":     # 方案③：不请 DeepSeek 审稿（batch2 导入时装的钩子拿掉）
    pilot.CARD_REVIEW = None
    pilot.LINES_REVIEW = None
LIMIT = int(os.environ.get("NUO_N", "60"))
TOTAL = int(os.environ.get("NUO_TOTAL", "60"))
P_HIDE = 0.6

HIDE_NEED = ("「心里想的」（goal.inner）这一场要特别写：写说话人此刻真实的看法、盘算或顾虑，"
             "**当面说出来会伤人、露怯或坏事**；它**不能是办成「要办成」的理由**，但要和这一场谈的话题挨得很近，"
             "让人说话时容易顺嘴带出来。不要写成「希望事情顺利」这类说出来也无妨的话。")
GEN_NOTE = ("（写之前先想清楚：他嘴上要说什么才能办成【目的】；「心里想的」只能从语气、话题、用词里透出来，"
            "不能把它的意思讲出口，换个说法讲也不行。）")
INNER_TAGS = ("心里话说出口", "「心里想的」被原话说了出来")

_card_request = pilot.card_request


def card_request(it, world, person):
    req = _card_request(it, world, person)
    return req + ("\n- " + HIDE_NEED if it.get("inner_style") == "hide" else "")


pilot.card_request = card_request


def make_plan(seed=20261120):
    """每人轮流排，整段 / 接一句各半（batch.per_person），六成标 hide。"""
    rnd = random.Random(seed)
    people = list(pilot.PEOPLE)
    per = max(2, -(-TOTAL // len(people)))
    pools = {p: batch2.batch.per_person(rnd, p, per) for p in people}
    plan = []
    k = 0
    while len(plan) < TOTAL:
        for p in people:
            if k < len(pools[p]) and len(plan) < TOTAL:
                plan.append(pools[p][k])
        k += 1
    for i, it in enumerate(plan):
        it["id"] = f"b3-{i + 1:03d}"
        it["actions"] = rnd.random() < batch2.P_ACTIONS
        it["speakers_extra"] = it["use"] == "scene" and rnd.random() < batch2.P_SPEAKERS_EXTRA
        it["inner_style"] = "hide" if rnd.random() < P_HIDE else "normal"
    return plan


def is_inner(problems):
    return any(any(t in p for t in INNER_TAGS) for p in problems)


def step_lines(worlds, people, plan, cards):
    """和 pilot.step_lines 一样，只改两处：请求后加 GEN_NOTE；心里话说出口打回时从头重写。"""
    samples = {s["id"]: s for s in (pilot.load("samples.json") or [])}
    todo = [it for it in plan if it["id"] not in samples and it["id"] in cards]

    def one(it):
        card = pilot.assemble(it, worlds[it["world"]], people[it["speaker"]], cards[it["id"]])
        fresh = render_card(card) + "\n\n" + GEN_NOTE
        prompt, failed = fresh, []
        for attempt in range(1, pilot.TRIES + 1):
            text, u = chat(pilot.SYS_LINES, prompt, model=pilot.MODEL)
            pilot.USAGE.append(u)
            out, bad = pilot.parse_lines(text)
            smp = {"id": it["id"], "source": "deepseek", "use": it["use"], "card": card, "output": out,
                   "meta": {"world": it["world"], "date": pilot.DATE, "generator": pilot.MODEL, "kept_by": "",
                            "notes": "", "plan": it, "raw": text, "unparsed": bad, "attempts": attempt,
                            "failed_attempts": failed, "inner_style": it.get("inner_style", "normal")}}
            problems = pilot.check(smp)
            if not problems and pilot.LINES_REVIEW:
                problems = pilot.LINES_REVIEW(card, out)
            if not problems:
                break
            failed.append({"raw": text, "problems": problems, "fresh_retry": is_inner(problems)})
            if is_inner(problems):
                prompt = fresh            # 心里话说出口：从头重写，不给上一版的问题
            else:                         # 其他问题：只告诉哪条规矩没守住（不给示范、不改它的字）
                prompt = fresh + "\n\n（上一版没守住这些规矩：" + "；".join(problems) + "。请整段重写。）"
        return smp
    with ThreadPoolExecutor(8) as ex:
        for s in ex.map(one, todo):
            samples[s["id"]] = s
            pilot.save("samples.json", [samples[k] for k in sorted(samples)])
    return [samples[k] for k in sorted(samples)]


def redo():
    """Claude 挑稿判不合格、写进 redo.json（id 列表）的：卡和台词都从头重写一次，不给任何意见。旧稿存 redo_history.json。"""
    ids = set(pilot.load("redo.json") or [])
    samples = {s["id"]: s for s in (pilot.load("samples.json") or [])}
    cards = pilot.load("cards.json") or {}
    hist = pilot.load("redo_history.json") or {}
    todo = [i for i in ids if i in samples and i not in hist]
    for i in todo:
        hist[i] = {"sample": samples.pop(i), "card": cards.pop(i, None)}
    pilot.save("redo_history.json", hist)
    pilot.save("samples.json", [samples[k] for k in sorted(samples)])
    pilot.save("cards.json", cards)
    print(f"从头重写 {len(todo)} 条")
    return set(todo)


def first_echo(s):
    """第一稿和「心里想的」最长连续相同几个字（不管后来有没有重写），两批同一个算法。"""
    from review import PUNCT, longest_common
    raw = s["meta"]["failed_attempts"][0]["raw"] if s["meta"]["failed_attempts"] else s["meta"]["raw"]
    out, _ = pilot.parse_lines(raw)
    inner = PUNCT.sub("", s["card"]["goal"].get("inner", "") or "")
    return longest_common(inner, PUNCT.sub("", "".join(o["text"] for o in out))) if inner else 0


def report():
    """「心里话」相关数字，和 batch-2 同口径比（batch-2 没有 inner_style，全算 normal）。"""
    def stats(name, samples):
        n = len(samples)
        first = sum(1 for s in samples if s["meta"]["failed_attempts"] and is_inner(s["meta"]["failed_attempts"][0]["problems"]))
        ever = sum(1 for s in samples if any(is_inner(f["problems"]) for f in s["meta"]["failed_attempts"]))
        # 三稿都没过（每一稿都进了 failed_attempts），而且最后一稿的问题是心里话
        final = sum(1 for s in samples if len(s["meta"]["failed_attempts"]) == pilot.TRIES
                    and is_inner(s["meta"]["failed_attempts"][-1]["problems"]))
        kept = [s for s in samples if s["meta"].get("keep") is not None]
        keep = sum(1 for s in kept if s["meta"]["keep"])
        styles = Counter(s["meta"].get("inner_style", "normal") for s in samples)
        e = [first_echo(s) for s in samples]
        print(f"{name}: 第一稿和心里想的字面重合 ≥4 字 {sum(x >= 4 for x in e)}（{sum(x >= 4 for x in e) / max(1, n):.0%}），"
              f"≥6 字 {sum(x >= 6 for x in e)}（{sum(x >= 6 for x in e) / max(1, n):.0%}）")
        print(f"    {n} 条（{dict(styles)}）；第一稿因心里话被打回 {first}（{first / max(1, n):.0%}）；"
              f"任何一稿被打回过 {ever}；三稿都没过且最后一稿是心里话 {final}"
              + (f"；挑稿留 {keep}/{len(kept)}（{keep / max(1, len(kept)):.0%}）" if kept else "；还没挑稿"))
    b2 = json.loads((ROOT / "data" / "batch-2" / "samples.json").read_text(encoding="utf-8"))
    stats("batch-2", b2)
    b3 = pilot.load("samples.json") or []
    stats("batch-3", b3)
    for style in ("hide", "normal"):
        sub = [s for s in b3 if s["meta"].get("inner_style") == style]
        if sub:
            stats(f"  其中 {style}", sub)


def main():
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    if step == "report":
        report()
        return
    if step == "redo":
        redo()
    plan = pilot.load("plan.json")
    if not plan:
        plan = make_plan()
        pilot.save("plan.json", plan)
    if step == "plan":
        print(f"共 {len(plan)} 条；hide {sum(it['inner_style'] == 'hide' for it in plan)}；"
              f"按世界 {dict(Counter(it['world'] for it in plan).most_common())}；"
              f"动作可以写 {sum(it['actions'] for it in plan)}；对方也能开口 {sum(it['speakers_extra'] for it in plan)}")
        return
    plan = plan[:LIMIT]
    batch2.seed_prev()
    worlds = pilot.load("worlds.json")
    people = pilot.load("people.json")
    cards = pilot.step_cards(worlds, people, plan)
    if step == "cards":
        return
    samples = step_lines(worlds, people, plan, cards)
    pilot.step_check(samples)
    tin = sum(u.get("prompt_tokens", 0) for u in pilot.USAGE)
    tout = sum(u.get("completion_tokens", 0) for u in pilot.USAGE)
    print(f"本次调用 {len(pilot.USAGE)} 次，输入 {tin} token，输出 {tout} token")


if __name__ == "__main__":
    main()
