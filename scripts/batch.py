"""第一批扩量（batch-1，2026-11-18 Ken 定）：600 条，10 个世界、31 个练习角色，全用 DeepSeek Flash。

和试跑（pilot.py）共用全部步骤和给 DeepSeek 的说明，只换「设计」：世界 / 角色的一句话设定和抽样计划。
- 老 4 个世界、老 5 个角色的档案直接沿用 data/pilot-001（考卷 S 组用的就是这几份），名字标签对齐考卷。
- 新 6 个世界、26 个新角色：一句话设定由 Claude 给（Ken 过了名单），世界说明 / 档案由 DeepSeek 写。
- 每人前 2 条（1 整段 + 1 接一句）排在最前面：先跑 NUO_N=62 看质量，再跑 NUO_N=600 接着做完（已做的会跳过）。

用法：NUO_N=62 python3 scripts/batch.py [plan / worlds / people / cards / lines / all]
结果在 data/batch-1/（不进 git）。
"""

import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pilot  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RUN = os.environ.get("NUO_RUN", "batch-1")
pilot.OUT = ROOT / "data" / RUN
LIMIT = int(os.environ.get("NUO_N", "62"))
SEED_RUN = ROOT / "data" / "pilot-001"          # 老世界、老角色档案（考卷 S / N 组也取自这里）
OLD_NAMES = {"老铁匠": "老铁匠", "牙人": "牙人", "语文老师": "陈老师", "法师学徒": "法师学徒", "维修机器人": "维修机器人"}

# ---------- 新世界（一句话设定，Claude 给，Ken 同意 2026-11-18） ----------
NEW_WORLDS = {
    "江湖": {"brief": "南方水乡的江湖：镖局、漕帮、码头、水寨，靠船和规矩吃饭。说话是武侠小说式的半白话，讲江湖规矩和客套。门派、帮会、地名、人名都要自己新编，不能用金庸、古龙等小说里的名字。",
             "emotions": ["平常", "生气", "冷笑", "惊讶"]},
    "民国": {"brief": "民国年间北方一座新编的小城，有戏园子、茶馆、车站、当铺，新旧交替。说话是老北方口语夹一点新名词。城名、店名、人名都要自己新编，不能用真实历史人物和真实事件。",
             "emotions": ["平常", "高兴", "生气", "无奈"]},
    "山区小镇": {"brief": "当代中国西南山区的一个小镇，逢五赶集，年轻人多在外打工，有留守老人、小学、卫生院、刚通的快递点。普通话口语带一点西南味，但要让各地人都看得懂。镇名、村名自己新编。",
             "emotions": ["平常", "高兴", "着急", "生气"]},
    "校园": {"brief": "当代中国南方一所普通高中，有高三、晚自习、食堂、小卖部、篮球场。学生说话是年轻人口语，老师说话是老师腔。校名、人名自己新编。",
             "emotions": ["平常", "得意", "生气", "紧张", "无语"]},
    "废土": {"brief": "一场大灾变几十年后的世界，人们住在用旧时代废墟搭起来的聚居地，靠以物易物、拾荒和净水过日子，旧时代的东西只剩传说。说话粗糙、简短、实用。地名、帮派名自己新编。",
             "emotions": ["平常", "警惕", "生气", "疲惫"]},
    "动物森林": {"brief": "一片会说话的动物住的森林，像童话寓言：有河、橡树、集市树墩、冬天要囤粮。动物们按自己的习性说话做事，不知道人类世界。",
             "emotions": ["平常", "高兴", "生气", "害怕"]},
}

# ---------- 新角色（一句话设定，Claude 给，Ken 过了名单 2026-11-18） ----------
NEW_PEOPLE = {
    "说书先生": {"world": "古风", "brief": "茶馆里说书的先生，爱夸张、爱卖关子，三句话不离评书腔"},
    "小乞丐": {"world": "古风", "brief": "十来岁的小乞丐，机灵、消息灵通，嘴甜，开口就要钱"},
    "县令千金": {"world": "古风", "brief": "县令家被宠坏的千金小姐，任性、嘴不饶人，但心眼不坏"},
    "网红主播": {"world": "现代", "brief": "二十出头的网红主播，满嘴网络梗，镜头前热情、镜头后冷淡"},
    "中年程序员": {"world": "现代", "brief": "刚被裁员的中年程序员，话短、爱用术语，心里憋着火"},
    "广场舞领队": {"world": "现代", "brief": "广场舞队的领队阿姨，爱张罗、好面子、说话一套一套"},
    "吟游诗人": {"world": "奇幻", "brief": "四处流浪的吟游诗人，浮夸、爱押韵、爱吹嘘自己的冒险"},
    "古龙": {"world": "奇幻", "brief": "活了上千年的古龙，傲慢、说话很慢，看人类像看虫子"},
    "小修女": {"world": "奇幻", "brief": "修道院里的小修女，善良、胆小，爱引经文给自己壮胆"},
    "叛逆少年": {"world": "科幻", "brief": "飞船上第六代出生的少年，冲、不服管，一心想下船去看真正的星球"},
    "老档案员": {"world": "科幻", "brief": "管船上档案室的老人，记得很多地球的老故事，爱讲古，有点糊涂"},
    "总镖头": {"world": "江湖", "brief": "镖局的总镖头，豪爽、重信用，最讲走镖的规矩"},
    "漕帮小头目": {"world": "江湖", "brief": "漕帮码头上的小头目，狠、护地盘，开口就是帮规"},
    "江湖郎中": {"world": "江湖", "brief": "走街串巷的江湖郎中，医术半真半假，满嘴偏方和吉利话"},
    "戏班名角": {"world": "民国", "brief": "戏班里唱青衣的名角，清高、讲究、爱摆角儿的架子"},
    "茶馆跑堂": {"world": "民国", "brief": "茶馆的跑堂伙计，嘴快、会来事，城里的事什么都知道一点"},
    "女学生": {"world": "民国", "brief": "留洋回来的女学生，新派、爱讲道理，看不惯旧规矩"},
    "留守奶奶": {"world": "山区小镇", "brief": "独自在家带孙子的留守老奶奶，耳背、爱唠叨，什么话都能绕到孙子身上"},
    "返乡青年": {"world": "山区小镇", "brief": "从城里回乡做直播卖山货的青年，满腔热情，老被乡亲泼冷水"},
    "小镇兽医": {"world": "山区小镇", "brief": "镇上唯一的兽医，话少、实在、脾气急，心疼牲口胜过心疼人"},
    "校霸": {"world": "校园", "brief": "高二的校霸，成绩差、嘴硬、爱逞强，但讲义气"},
    "学霸班长": {"world": "校园", "brief": "高三的学霸班长，认真、守规矩，最怕让别人失望"},
    "拾荒者": {"world": "废土", "brief": "独来独往的拾荒者，疑心重、只认交换，不信任何人"},
    "疯老头": {"world": "废土", "brief": "从灾变前活下来的老人，半疯，说话颠三倒四，却偶尔一语说中"},
    "老乌龟": {"world": "动物森林", "brief": "森林里最老的乌龟，说话慢吞吞，爱讲大道理"},
    "松鼠": {"world": "动物森林", "brief": "急性子的松鼠，说话飞快、坐不住，满脑子囤粮过冬"},
}

pilot.WORLDS = {**pilot.WORLDS, **NEW_WORLDS}
pilot.PEOPLE = {**pilot.PEOPLE, **NEW_PEOPLE}
OLD_PEOPLE = list(OLD_NAMES)
OTHER_TROUBLES = ["a", "d", "e", "f", "g", "h", "i"]


def per_person(rnd, who, n):
    """一个人的 n 条：整段 / 接一句各半；接一句里难题 b、c 各约 1/4；目的轮着用。前 2 条 = 1 整段 + 1 接一句。"""
    half = n // 2
    k = max(1, round(half / 4))
    troubles = ["b"] * k + ["c"] * k
    others = OTHER_TROUBLES[:]
    rnd.shuffle(others)
    while len(troubles) < half:
        troubles.append(others[(len(troubles) - 2 * k) % len(others)])
    rnd.shuffle(troubles)
    goals = []
    while len(goals) < n:
        g = pilot.GOALS[:]
        rnd.shuffle(g)
        goals += g
    items = []
    for j in range(n):
        use = "scene" if j % 2 == 0 else "reply"
        it = {"speaker": who, "world": pilot.PEOPLE[who]["world"], "use": use, "goal": goals[j],
              "relation": rnd.choice(pilot.RELATIONS), "first_meet": rnd.random() < 0.4,
              "happened": rnd.random() < 0.35, "others": rnd.random() < 0.2,
              "listener_now": rnd.choice(pilot.LISTENER_NOW) if rnd.random() < 0.25 else "",
              "speaker_now": rnd.choice(pilot.SPEAKER_NOW) if rnd.random() < 0.2 else ""}
        if use == "reply":
            it["trouble"] = troubles.pop()
            it["lines"], it["max_chars"] = (1, 2), rnd.choice([20, 40, 40, 60])
        else:
            it["trouble"] = "a"
            it["lines"], it["max_chars"] = rnd.choice(pilot.FORMATS[1:4])
        it["actions"] = rnd.random() < 0.25
        it["speakers_extra"] = use == "scene" and rnd.random() < 0.15
        items.append(it)
    return items


def make_plan(seed=20261118):
    """600 条：老 5 人各 30、新 26 人各 18（截到 600）。每人前 2 条排最前（共 62 条，先跑这些看质量）。"""
    rnd = random.Random(seed)
    first, rest = [], []
    for who in pilot.PEOPLE:
        items = per_person(rnd, who, 30 if who in OLD_NAMES else 18)
        first += items[:2]
        rest += items[2:]
    rnd.shuffle(rest)
    plan = (first + rest)[:600]      # 新角色其实是 26 个（名单表里数错成 25），共 618 条，截到 600
    for i, it in enumerate(plan):
        it["id"] = f"b1-{i + 1:03d}"
    return plan


# ---------- 世界词扩到 30 个（2026-11-18 Ken 定）：原 10 个不动，DeepSeek 补 20 个 ----------
TERMS_TARGET = 30
SYS_MORE_TERMS = """你在为一个「演员模型」准备训练资料。下面是一个世界的说话规则，其中 terms 是这个世界特有的词。
请再补 20 个这个世界特有的词：地名、店铺、行当、物件、吃食、钱、规矩、俗语、称呼等都可以，覆盖面要广，和已有的词不能重复、不能是近义词。
每个写成「词（简短解释）」。只输出 JSON：{"terms": [...]}。全部用中文，用普通话写。"""


def step_more_terms(worlds):
    todo = [w for w, d in worlds.items() if len(d["terms"]) < TERMS_TARGET]
    for w in todo:
        data, u = pilot.chat_json(SYS_MORE_TERMS, json.dumps(worlds[w], ensure_ascii=False), model=pilot.MODEL)
        pilot.USAGE.append(u)
        have = {t.split("（")[0] for t in worlds[w]["terms"]}
        new = [t for t in data.get("terms", []) if t.split("（")[0] not in have]
        worlds[w]["terms"] += new[:TERMS_TARGET - len(worlds[w]["terms"])]
        pilot.save("worlds.json", worlds)
        print(f"{w}：词表 {len(worlds[w]['terms'])} 个")
    return worlds


_card_request = pilot.card_request


def card_request(it, world, person):
    """每张戏卡只给 DeepSeek 看随机 12 个世界词（按 id 固定），免得总挑那几个常用词。"""
    w = dict(world)
    w["terms"] = random.Random(it["id"]).sample(world["terms"], min(12, len(world["terms"])))
    return _card_request(it, w, person)


pilot.card_request = card_request


def seed_old():
    """老世界、老角色档案从 pilot-001 拷来（不重写，考卷 S 组考的就是这几份）。"""
    if not (pilot.OUT / "worlds.json").exists():
        pilot.save("worlds.json", json.loads((SEED_RUN / "worlds.json").read_text(encoding="utf-8")))
    if not (pilot.OUT / "people.json").exists():
        people = json.loads((SEED_RUN / "people.json").read_text(encoding="utf-8"))
        for k, name in OLD_NAMES.items():
            people[k]["name"] = name      # 只改名字标签，和考卷对齐
        pilot.save("people.json", people)


def main():
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    plan = pilot.load("plan.json")
    if not plan:
        plan = make_plan()
        pilot.save("plan.json", plan)
    if step == "plan":
        c = Counter((it["use"], it["trouble"]) for it in plan)
        print(len(plan), "条；", dict(sorted(c.items())))
        print("眼下不平常：", sum(1 for it in plan if it["listener_now"] or it["speaker_now"]))
        return
    plan = plan[:LIMIT]
    seed_old()
    worlds = step_more_terms(pilot.step_worlds())
    if step == "worlds":
        return
    people = pilot.step_people(worlds)
    if step == "people":
        return
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
