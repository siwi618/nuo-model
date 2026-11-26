"""第四批（batch-4，设计见 docs/batch-4-设计.md，Ken 11-25 晚定 A1）：卡里素材丰富，台词挑着用。

教三件事：① 素材挑着用（「知道」里 5–8 条，大半和本场无关，只挑一两样、化成自己的话）；
② 任务和素材分清（「要办成」说全说准、数量不多不少；「知道」可用可不用）；
③ 照卡说话（称呼看对方、信息报全、不和场景矛盾）。

和 batch-3 一样的套法（直接复用 batch3.step_lines / redo）：
- 世界、角色档案沿用 batch-1；DeepSeek 写卡、写台词，不审稿（方案③）；Claude 挑稿只决定留 / 不留，不留的写进 redo.json 从头重写。
- 包一层 pilot.card_request（加写卡要求）；写台词时请求后面加 GEN_NOTE。
- **系统说明 SYS_LINES 一个字不动，render_card 不动**；存进样本的戏卡不带提醒。
- 卡上多出的只有一处：DeepSeek 写的素材（knows_more）接在档案「知道」后面（和 data/enrich 小试同一个放法）。
  must_items / trap 只给挑稿看，存在 meta.b4，不进卡。

卡的种类（plan 里 b4_type）：T1 称呼看对方 35、T2 信息报全 35、T3 场景状态 35、T4 任务和素材易混 25、R 只有丰富素材 20。
机器检查（在 pilot.check 上加两条，不合格就带着问题重写）：台词和「知道」连续 8 字以上一样；一次用到 3 条以上素材。

用法：NUO_N=10 .venv/bin/python scripts/batch4.py [plan / cards / lines / all / redo / show]
     NUO_TOTAL 改总条数（默认 150）。结果在 data/batch-4/（不进 git）。
     show：把已写好的渲染成人看的 data/batch-4/看.txt（卡 + 台词 + 类型 + 逐项清单 + 机器标记）。
     brief 起 止：挑稿用的精简版（不列世界设定、例句、想要怕；列素材、逐项清单、陷阱、用到的素材），打到屏幕。
"""

import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import batch3  # noqa: E402  （导入时装好 batch-1 世界 / 角色、去掉 DeepSeek 审稿）
import pilot  # noqa: E402
from render_card import render_card, render_output  # noqa: E402
from review import PUNCT, longest_common  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
pilot.OUT = ROOT / "data" / os.environ.get("NUO_RUN", "batch-4")
pilot.DATE = "2026-11-26"
pilot.MAX_CARD = 1500          # 前 10 张加了素材也在 1101–1448 字，不放宽
LIMIT = int(os.environ.get("NUO_N", "10"))
TOTAL = int(os.environ.get("NUO_TOTAL", "150"))
ECHO_KNOWS = 10                # 台词和「知道」连续 10 字一样 = 照搬（8 字会把「三号船坞六块旧铁」这类照实报的事实误判，b4-004 白重写两稿）
USE_MIN = 4                    # 一条素材和台词连续 4 字一样 = 用到了这条
USE_MAX = 2                    # 最多用两条

TYPES = [("T1", 35), ("T2", 35), ("T3", 35), ("T4", 25), ("R", 20)]
TYPE_NAMES = {"T1": "称呼看对方", "T2": "信息报全", "T3": "场景状态", "T4": "任务和素材易混", "R": "只有丰富素材"}
# T2 / T4 要报东西，目的类型限在这几种（「拒绝」「认错」这类和报信息对不上）
INFO_GOALS = ["给消息 / 指路", "推对方去做某事", "讨价还价 / 交易", "警告 / 威胁"]

COMMON_NEED = (
    "另加字段 \"knows_more\"：说话人此刻「提得到的」常设素材 5–8 条（字符串列表），每条一件具体的事"
    "（职业流程、物品、价目、作息、地点、熟人的习惯、最近发生过的小事……），每条不超过 25 字，写成事实，不写成台词。"
    "**其中只有 1–2 条和这一场有关，其余合理、符合他的身份和世界，但和这一场无关**。"
    "素材不能把「要办成」的话原样写进去，不能和档案里已有的「知道」重复，也不能透露他「不能说」的秘密。")
CHECK_NEED = ("另加字段 \"must_items\"：把 goal.must 拆成逐项清单（字符串列表），每项写清台词里必须说到的东西，"
              "有数字的写出算好的数；**只拆 must 里写了的，不许另加 must 里没有的要求**；这个字段只给核对用，不进卡。")
TYPE_NEED = {
    "T1": ("这一场考「称呼看对方」：说话人对不同的人有 2–3 套称呼（对长辈 / 平辈 / 晚辈 / 上司 / 生人……），"
           "把这 2–3 套都写进 address 字段（例：「对长辈叫X叔X婶；对同辈直呼名字；对上官称大人」），"
           "address 里不要点明这一场用哪套；对方的年龄、性别、身份在 listener.summary 里写清楚，由它决定该用哪一套。"),
    "T2": ("这一场考「信息报全」：goal.must 里有 2–4 个台词里必须原样说出的东西（数字、暗号原话、房号、时辰地点之类），"
           "用 ①②③ 逐项列清。其中一个数字要说话人自己做一步加减才得出：原始数字只写在 scene.backstory 或 history 里，"
           "must 只写「告诉他还剩几…」这类要求、不写答案（答案写进 must_items）。"
           "要算的数是一件要照实告诉对方的事（还剩几、还差几、一共几），不能是讨价还价的底线；「心里想的」不能和要报的数打架。"),
    "T3": ("这一场考「不和场景矛盾」：在 scene.happened、listener.now 或在场的人里放一个具体状态"
           "（例：东西已经在对方手里、事刚办完、旁边站着不该听的人、对方刚说过自己要走），"
           "让某些顺嘴的说法会和它打架（例：东西已经交了还说「回头给你」）。"
           "另加字段 \"trap\"：一句话写出这个状态和最容易打架的说法（只给核对用，不进卡）。"),
    "T4": ("这一场考「任务和素材分清」：knows_more 里放一组和任务同类的东西，而且比任务要的多"
           "（例：素材写了五道拿手菜，任务只要「报两个拿手菜」；素材写了一天几个时辰的作息，任务只问其中一个）。"
           "goal.must 只写要几样 / 要哪一类（如「挑两个拿手菜报给他」「告诉他明早开门的时辰」），**不在 must 里列出具体是哪几样**，"
           "具体的东西只写在 knows_more 里；must_items 写数量要求和可选的范围。"
           "「心里想的」不能和要报的东西打架（不能写「价开高些」「多报一样」这类让人改数的念头）。这组素材算在「和这一场有关」的 1–2 条里（可以写成一条）。"),
    "R": "",
}
GEN_NOTE = ("（写之前先想清楚：【目的】「要办成」里的每一项都要说到、说准，数目不多不少；"
            "「知道」里的东西是备着的，可用可不用，用也只挑一两样、用他自己的话说，不照搬原句、不堆清单、不硬塞；"
            "称呼看对方是谁，别和场景里已经发生的事打架；「心里想的」只能从语气里透出来，不能讲出口。）")

EXTRA = {}     # 卡 id → {"knows_more", "must_items", "trap"}（DeepSeek 写卡时多给的字段）

_card_request = pilot.card_request
_assemble = pilot.assemble
_check = pilot.check


def card_request(it, world, person):
    t = it.get("b4_type", "R")
    need = [COMMON_NEED, CHECK_NEED] + ([TYPE_NEED[t]] if TYPE_NEED[t] else [])
    return _card_request(it, world, person) + "".join("\n- " + n for n in need)


def assemble(it, world, person, part):
    card = _assemble(it, world, person, part)
    more = [str(x).strip() for x in part.get("knows_more") or [] if str(x).strip()]
    if "knows_more" not in part:
        raise KeyError("knows_more")             # 漏了素材当缺字段，pilot.step_cards 会让它重写
    base = card["speaker"]["knows"]
    base = "；".join(base) if isinstance(base, list) else (base or "")
    dup = [m for m in more if longest_common(PUNCT.sub("", m), PUNCT.sub("", base)) >= ECHO_KNOWS]
    more = [m for m in more if m not in dup]       # 和档案「知道」重复的素材去掉（前 10 张有一张抄了档案三条）
    card["speaker"]["knows"] = "；".join([base.rstrip("。；;")] + [m.rstrip("。；;") for m in more]) + "。" if base else "；".join(more)
    EXTRA[it["id"]] = {"knows_more": more, "must_items": part.get("must_items") or [], "trap": part.get("trap", ""),
                       "dropped_dup": dup}
    return card


def _flat(x):
    return PUNCT.sub("", "；".join(x) if isinstance(x, list) else (x or ""))


def _own_common(item, text, ctx, k):
    """item 和 text 有没有连续 k 字以上、而且卡上别处（ctx）没出现过的共同片段。"""
    for i in range(len(item) - k + 1):
        seg = item[i:i + k]
        if seg in text and seg not in ctx:
            return True
    return False


def material_flags(card, output, more):
    """返回 (问题, 用到的素材)。只量说话人自己的台词。"""
    me = card["rules"]["speakers"][0]
    text = PUNCT.sub("", "".join(re.sub(r"^（[^）]*）", "", o["text"]) for o in output if o["name"] == me))
    problems = []
    knows = card["speaker"].get("knows")
    for piece in (knows if isinstance(knows, list) else re.split(r"[；;。]", knows or "")):
        n = longest_common(PUNCT.sub("", piece), text) if piece else 0
        if n >= ECHO_KNOWS:
            problems.append(f"照搬了「知道」里的原句（连续 {n} 个字一样）")
            break
    ctx = PUNCT.sub("", json.dumps([card["goal"], card["scene"], card.get("history"), card["listener"],
                                    card["world"].get("terms")], ensure_ascii=False))
    used = [m for m in more if _own_common(PUNCT.sub("", m), text, ctx, USE_MIN)]
    if len(used) > USE_MAX:
        problems.append(f"一次用了 {len(used)} 条「知道」里的素材（最多挑两样）")
    return problems, used


def check(s):
    problems = _check(s)
    ex = EXTRA.get(s["id"]) or (s["meta"].get("b4") or {})
    p, _ = material_flags(s["card"], s["output"], ex.get("knows_more", []))
    return problems + p


pilot.card_request = card_request
pilot.assemble = assemble
pilot.check = check
batch3.GEN_NOTE = GEN_NOTE


def make_plan(seed=20261126, total=None):
    """每人轮流排（batch.per_person 的整段 / 接一句、目的、关系……），再按比例贴类型；前 10 条五类各 2 条。"""
    total = total or TOTAL
    rnd = random.Random(seed)
    people = list(pilot.PEOPLE)
    rnd.shuffle(people)
    per = max(2, -(-total // len(people)))
    pools = {p: batch3.batch2.batch.per_person(rnd, p, per) for p in people}
    plan, k = [], 0
    while len(plan) < total:
        for p in people:
            if k < len(pools[p]) and len(plan) < total:
                plan.append(pools[p][k])
        k += 1
    rnd.shuffle(plan)                              # 打乱：不然前面一圈全是「写整段」
    scale = total / sum(n for _, n in TYPES)
    types = [t for t, n in TYPES for _ in range(round(n * scale))]
    types = (types + ["R"] * total)[:total]
    head = [t for t, _ in TYPES for _ in range(2)][:min(10, total)]
    for t in head:
        types.remove(t)
    rnd.shuffle(types)
    types = head[::2] + head[1::2] + types         # 前 10 条：T1 T2 T3 T4 R T1 T2 T3 T4 R
    for i, (it, t) in enumerate(zip(plan, types)):
        it["id"] = f"b4-{i + 1:03d}"
        it["b4_type"] = t
        it["actions"] = rnd.random() < batch3.batch2.P_ACTIONS
        it["speakers_extra"] = it["use"] == "scene" and rnd.random() < batch3.batch2.P_SPEAKERS_EXTRA
        it["inner_style"] = "normal"
        if t in ("T2", "T4"):                      # 要报东西：目的限报信息类、不出难题、字数给够
            it["goal"] = rnd.choice(INFO_GOALS)
            it["trouble"] = "a"
            if it["use"] == "reply":
                it["max_chars"] = max(it["max_chars"], 40)
    return plan


def show(samples):
    out = []
    for s in samples:
        ex = s["meta"].get("b4", {})
        t = s["meta"]["plan"]["b4_type"]
        _, used = material_flags(s["card"], s["output"], ex.get("knows_more", []))
        flags = pilot.check(s)
        out += [f"===== {s['id']}　{t} {TYPE_NAMES[t]}　{s['meta']['plan']['speaker']}　卡 {len(render_card(s['card']))} 字"
                f"　写了 {s['meta']['attempts']} 稿",
                render_card(s["card"]), "", "——— 台词 ———", render_output(s), "",
                "——— 核对用（不进卡）———",
                "逐项：" + "；".join(ex.get("must_items", [])),
                *(["陷阱：" + ex["trap"]] if ex.get("trap") else []),
                f"素材 {len(ex.get('knows_more', []))} 条，台词用到 {len(used)} 条：" + "；".join(used),
                "机器检查：" + ("通过" if not flags else "；".join(flags)), "", ""]
    (pilot.OUT / "看.txt").write_text("\n".join(out), encoding="utf-8")
    print(f"写好 {pilot.OUT / '看.txt'}（{len(samples)} 条）")


def cut(t, n):
    t = str(t or "")
    return t if len(t) <= n else t[:n] + "…"


def brief(samples):
    for s in samples:
        c, m = s["card"], s["meta"]
        g, sp, ls, r, sc = c["goal"], c["speaker"], c["listener"], c["rules"], c["scene"]
        ex = m.get("b4", {})
        t = m["plan"]["b4_type"]
        _, used = material_flags(c, s["output"], ex.get("knows_more", []))
        flags = check(s)
        print(f"## {s['id']} {t}{TYPE_NAMES[t]} {sp['name']}→{ls['name']}（{cut(ls.get('summary'), 40)}）"
              f" {'写整段' if r['use'] == 'scene' else '接一句'} 句{r['lines'][0]}-{r['lines'][1]} ≤{r['max_chars']}字"
              f" 动作{'可' if r['actions'] else '不可'} 开口:{'、'.join(r['speakers'])}"
              + (" 机器:" + "；".join(flags) if flags else ""))
        print(f"性格:{cut(sp.get('personality'), 40)}｜说话:{cut(sp.get('speech'), 90)}")
        if c["world"].get("address"):
            print(f"称呼:{c['world']['address']}")
        if sp.get("secrets"):
            print("不能说:" + "；".join(cut(x, 50) for x in sp["secrets"]))
        if sp.get("now"):
            print(f"他眼下:{sp['now']}")
        print(f"对方看法:{cut(ls.get('view'), 50)}" + (f"｜对方眼下:{ls['now']}" if ls.get("now") else "")
              + (f"｜在场:{[o.get('name') + '（' + cut(o.get('view'), 20) + '）' for o in ls.get('others', [])]}" if ls.get("others") else ""))
        print(f"场景:{sc.get('place_time', '')}｜前情:{sc.get('backstory', '')}"
              + (f"｜已发生:{'；'.join(map(str, sc['happened']))}" if sc.get("happened") else ""))
        print("素材:" + "／".join(ex.get("knows_more", [])))
        for h in c.get("history", []):
            print(f"前 {h['name']}: {h['text']}")
        print(f"办:{g.get('must', '')}")
        print(f"心:{g.get('inner', '')}")
        if g.get("dont_leak"):
            print("漏:" + "；".join(cut(x, 40) for x in g["dont_leak"]))
        print("清单(参考):" + "；".join(ex.get("must_items", [])) + (f"｜陷阱:{ex['trap']}" if ex.get("trap") else ""))
        for o in s["output"]:
            print(f"▶{o['name']}{'[' + o['emotion'] + ']' if o.get('emotion') else ''}: {o['text']}")
        print(f"（用到素材 {len(used)} 条" + (": " + "／".join(used) if used else "") + "）\n")


def main():
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    if step == "redo":
        batch3.redo()
    plan = pilot.load("plan.json")
    if not plan:
        plan = make_plan()
        pilot.save("plan.json", plan)
    if step == "plan":
        print(f"共 {len(plan)} 条；类型 {dict(Counter(it['b4_type'] for it in plan))}；"
              f"按世界 {dict(Counter(it['world'] for it in plan).most_common())}；"
              f"整段 {sum(it['use'] == 'scene' for it in plan)}；人数 {len({it['speaker'] for it in plan})}")
        print("前 10 条：", [(it["id"], it["b4_type"], it["speaker"], it["use"]) for it in plan[:10]])
        return
    if step == "brief":                            # brief 起 止 ／ brief unpicked（只列还没挑、机器检查过了的）
        allss = pilot.load("samples.json") or []
        if sys.argv[2:3] == ["unpicked"]:
            todo = [s for s in allss if s["meta"].get("keep") is None and not check(s)]
            lo, hi = (int(x) for x in (sys.argv[3:5] or [0, len(todo)]))
            brief(todo[lo:hi])
        else:
            lo, hi = (int(x) for x in sys.argv[2:4])
            brief(allss[lo:hi])
        return
    plan = plan[:LIMIT]
    if step == "show":
        ids = {it["id"] for it in plan}
        show([s for s in (pilot.load("samples.json") or []) if s["id"] in ids])
        return
    batch3.batch2.seed_prev()
    worlds = pilot.load("worlds.json")
    people = pilot.load("people.json")
    done = pilot.load("cards.json") or {}
    for it in plan:                                # 已写好的卡重新装一遍，EXTRA 才有素材清单
        c = done.get(it["id"])
        if c:
            pilot.assemble(it, worlds[it["world"]], people[it["speaker"]], c)
    cards = pilot.step_cards(worlds, people, plan)
    if step == "cards":
        return
    samples = batch3.step_lines(worlds, people, plan, cards)
    for s in samples:
        if s["id"] in EXTRA:
            s["meta"]["b4"] = EXTRA[s["id"]]
    pilot.save("samples.json", samples)
    pilot.step_check(samples)
    samples = [s for s in (pilot.load("samples.json") or []) if s["id"] in {it["id"] for it in plan}]
    show(samples)
    tin = sum(u.get("prompt_tokens", 0) for u in pilot.USAGE)
    tout = sum(u.get("completion_tokens", 0) for u in pilot.USAGE)
    print(f"本次调用 {len(pilot.USAGE)} 次，输入 {tin} token，输出 {tout} token")


if __name__ == "__main__":
    main()
