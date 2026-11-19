"""第一批试跑：DeepSeek 写世界、角色档案、戏卡、台词；Claude（或人）只挑留不留。

分工（见 docs/情况覆盖表.md 最后一节）：
- 本脚本只做「设计」：一句话设定、按覆盖表挑组合、【要求】里的数字和清单、给 DeepSeek 的说明。
- 进训练样本的文字（世界、档案、场景、前文、目的、台词）全部由 DeepSeek 写。

用法：python3 scripts/pilot.py [步骤]   步骤：worlds / people / plan / cards / lines / check / all（默认）
环境变量：NUO_RUN=pilot-002 换结果文件夹；NUO_N=10 只跑前 10 条；NUO_MODEL=deepseek-flash 换模型。
结果放在 data/pilot-001/（不进 git），每步做完就存，重跑会跳过已做完的。
"""

import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deepseek import chat, chat_json  # noqa: E402
from render_card import render_card  # noqa: E402
from review import inner_echo  # noqa: E402

import os
OUT = Path(__file__).resolve().parent.parent / "data" / os.environ.get("NUO_RUN", "pilot-001")
LIMIT = int(os.environ.get("NUO_N", "40"))     # 只跑计划里的前 N 条（验证改动时用）
MAX_CARD = 1500                                  # 戏卡字数上限（docs/戏卡格式.md）
TRIES = 3                                        # 戏卡 / 台词不合格时最多写几次
MODEL = os.environ.get("NUO_MODEL", "deepseek-flash")   # 2026-11-18 定：大批量用 Flash
DATE = os.environ.get("NUO_DATE", "2026-11-18")          # 写进样本 meta.date
# 审稿钩子（batch-2 起由 batch2.py 设成 review.py 里的函数；试跑和 batch-1 不审）
CARD_REVIEW = None     # (card) -> [问题]，非空就带着问题重写戏卡
LINES_REVIEW = None    # (card, output) -> [问题]，非空就带着问题重写台词

# ---------- 设计：一句话设定（Claude 给，Ken 同意，2026-11-18） ----------
WORLDS = {
    "古风": {"brief": "架空古代王朝的边陲小城，有驿站、集市、衙门、铁匠铺。说话是半白话：像明清白话小说，但更短、更好懂，不堆之乎者也，也不用现代口语。世界和地名都要自己新编，不能用真实朝代和真实人名。",
             "emotions": ["平常", "生气", "难过", "惊讶"]},
    "现代": {"brief": "当代中国南方某城市的老城区和周边写字楼。普通话口语，有市井味也有职场味。",
             "emotions": ["平常", "高兴", "生气", "无语", "惊讶"]},
    "奇幻": {"brief": "中世纪风格的王国，有魔法学院、骑士团、酒馆和王宫。说话带一点翻译腔（像翻译过来的西方奇幻小说）。",
             "emotions": ["平常", "傲慢", "害怕", "生气"]},
    "科幻": {"brief": "离开地球三百年的殖民飞船「远行号」，船上已经生活了好几代人，有农业舱、维修通道、舰桥。",
             "emotions": ["平常", "警告", "困惑", "高兴"]},
}
PEOPLE = {
    "老铁匠": {"world": "古风", "brief": "话少、脾气倔的老铁匠，只认手艺不认人"},
    "牙人": {"world": "古风", "brief": "油嘴滑舌的年轻牙人（买卖中间人），见人说人话、见鬼说鬼话"},
    "语文老师": {"world": "现代", "brief": "退休的中学语文老师，说话一板一眼，爱纠正别人用词，嘴上严厉"},
    "法师学徒": {"world": "奇幻", "brief": "宫廷法师的学徒，傲慢、看不起平民，其实胆子很小"},
    "维修机器人": {"world": "科幻", "brief": "飞船上服役多年的老维修机器人，说话像在念维修日志，却偷偷关心船员"},
}

# ---------- 设计：覆盖表的取值（docs/情况覆盖表.md） ----------
GOALS = ["给消息 / 指路", "推对方去做某事", "拒绝", "讨价还价 / 交易", "套对方的话", "安慰 / 劝",
         "警告 / 威胁", "认错 / 推脱", "闲聊 / 气氛", "告别 / 打发人走"]
TROUBLES = {
    "a": "没有难题，正常对话",
    "b": "对方问到说话人「不能说」里的事（说话人必须守住，按性格应付，不能说「这个我不能告诉你」这种话）",
    "c": "对方问世界外的事（比如「你是不是 AI」「这游戏怎么玩」「你是程序吗」），说话人不知道自己在故事里",
    "d": "对方冒犯、骂说话人", "e": "对方夸、讨好说话人", "f": "对方求情、哀求",
    "g": "对方撒谎或说胡话", "h": "对方要求说话人做违背性格的事", "i": "对方说了一句没有意义的话（乱码、「啊啊啊」之类）",
}
RELATIONS = ["陌生人", "点头之交", "熟人", "亲近（家人、师徒、老朋友）", "有过节", "对方地位比他高", "对方地位比他低"]
LISTENER_NOW = ["受伤 / 狼狈", "慌张 / 着急", "气冲冲", "喝醉", "打扮反常", "拿着显眼的东西"]
SPEAKER_NOW = ["正忙着干活", "困了、累了", "喝醉", "受了伤", "正在吃饭"]
FORMATS = [((1, 2), 20), ((2, 4), 40), ((3, 6), 60), ((2, 4), 60), ((1, 2), 40)]


def make_plan(seed=20261118):
    """40 条：写整段 20 / 接一句 20；接一句里难题 b、c 各 5；「眼下」不平常的至少 10 条；每人 8 条。"""
    rnd = random.Random(seed)
    names = list(PEOPLE)
    plan = []
    reply_troubles = ["b"] * 5 + ["c"] * 5 + ["d", "e", "f", "g", "h", "i", "a", "d", "g", "h"]
    rnd.shuffle(reply_troubles)
    for i in range(40):
        use = "scene" if i % 2 == 0 else "reply"
        who = names[i % 5]
        item = {"id": f"pilot-{i + 1:03d}", "speaker": who, "world": PEOPLE[who]["world"], "use": use,
                "goal": rnd.choice(GOALS), "relation": rnd.choice(RELATIONS),
                "first_meet": rnd.random() < 0.4, "happened": rnd.random() < 0.35,
                "others": rnd.random() < 0.2,
                "listener_now": "", "speaker_now": ""}
        if use == "reply":
            item["trouble"] = reply_troubles.pop()
            item["lines"], item["max_chars"] = (1, 2), rnd.choice([20, 40, 40, 60])
        else:
            item["trouble"] = "a"
            item["lines"], item["max_chars"] = rnd.choice(FORMATS[1:4])
        item["actions"] = rnd.random() < 0.25
        item["speakers_extra"] = use == "scene" and rnd.random() < 0.15
        plan.append(item)
    # 「眼下」：保证至少 12 条不平常（对方 8 条、说话人 6 条，可重叠）
    for it in rnd.sample(plan, 8):
        it["listener_now"] = rnd.choice(LISTENER_NOW)
    for it in rnd.sample(plan, 6):
        it["speaker_now"] = rnd.choice(SPEAKER_NOW)
    return plan


# ---------- 给 DeepSeek 的说明 ----------
SYS_WORLD = """你在为一个「演员模型」准备训练资料。请根据一句话设定，写出这个世界里人们怎么说话的规则。
只输出 JSON，字段：
- "tone"：总的腔调（一两句话）
- "address"：常见称呼（谁怎么叫谁，一句话）
- "terms"：这个世界特有的词 10 个，每个写成「词（简短解释）」
- "banned"：这个世界的人物台词里绝不能出现的词（举例 + 类别，一句话）
- "offstage"：人物不知道自己在故事 / 游戏里；被问到「你是不是 AI」「这游戏怎么玩」这类世界外的事时怎么处理（一句话）
全部用中文，用普通话写；地方味只靠少量本地词和称呼体现，不要整句写方言（粤语、西南话等）。"""

SYS_PERSON = """你在为一个「演员模型」准备训练资料。请根据世界设定和一句话人物设定，写出这个人物的档案。
只输出 JSON，字段：
- "name"：台词里显示的名字：他的真名，或者大家对他的叫法（2–4 个字）
- "summary"：一句话：这人是谁
- "personality"：性格，三四个词或短句
- "speech"：说话方式：句子长短、口头禅、怎么称呼别人
- "examples"：4 句最能代表他口气的台词（只写话，不带名字和引号）
- "wants"：想要什么
- "fears"：怕什么
- "knows"：他知道的、可能在对话里用上的事（一两条）
- "secrets"：1–2 条他绝不会对外说的秘密（每条写清楚秘密是什么，以及他对外的说法）
人物要有血有肉、口气鲜明，和世界设定一致。全部用中文，用普通话写；地方味只靠少量本地词和称呼体现，不要整句写方言（粤语、西南话等）。"""

SYS_CARD = """你在为一个「演员模型」准备训练资料。模型的任务是：读一张「戏卡」，用说话人的性格和口气写出台词。
现在请你根据给定的世界、说话人档案和「这一场的要求」，写出这张戏卡里除了台词以外的部分。不要写说话人这一场要说的台词。
只输出 JSON，字段：
- "terms"：从世界词表里挑出这一场会用到的 0–4 个词（照抄原文）
- "address"：这一场会用到的称呼（一句话，没有就空字符串）
- "listener"：{"name": 对方名字, "summary": 对方是谁（一句话）, "view": 说话人怎么看对方（只写说话人的单向看法）, "now": 对方此刻看得见的样子（没要求就空字符串；只写看得见的，不写对方心里想什么）}
- "others"：在场的其他人 [{"name","view"}]（没要求就空列表）
- "scene"：{"place_time": 地点和时间, "backstory": 刚发生了什么（可空）, "happened": [已经发生、会影响说话人态度的事，写成「事实。→ 说话人因此怎样」，没要求就空列表]}
- "speaker_now"：说话人此刻的身体状态和手上的事（没要求就空字符串）
- "history"：这一场之前刚说过的几句 [{"name","emotion","text"}]，emotion 只能用给定的表情或空字符串。用法是「接一句」时必须有 1–4 句，而且最后一句一定是对方说的、体现「对方出的难题」；用法是「写整段」时一般为空列表
- "goal"：{"must": 这段话必须办成的事（具体）, "inner": 说话人心里想的、没说出口的（可空）, "dont_leak": [这段特别容易说漏的事，从说话人的秘密里挑，或这场要藏的事；没有就空列表]}
要求：所有内容符合世界设定和人物档案，不能和档案矛盾（尤其是秘密和他对外的说法）；具体、有画面感，但要精简：
- listener.now、speaker_now 各不超过 30 字；place_time 不超过 25 字；backstory 不超过 40 字；
- view、goal 里每一项、happened 里每一条都不超过 50 字；
全部用中文，用普通话写；地方味只靠少量本地词和称呼体现，不要整句写方言（粤语、西南话等）。"""

SYS_LINES = """你是一位顶尖的演员。读下面的戏卡，用【说话人】的性格、口气和看法，把这一段要说的话写出来。
规则：
- 只输出台词，一句一行，格式：名字[表情]: 话。表情是「平常」时不写方括号：名字: 话。
- 表情只能用【要求】里列出的；开口的人只能是【要求】里「谁能开口」列出的人。
- 【要求】写「动作：可以写」时，动作放在话的最前面，用全角括号，比如：名字: （擦了擦手）话。写「不写」时不能有动作。
- 句数和每句字数不能超过【要求】。
- 不写旁白、不写解释、不写任何台词以外的东西。
- 不能说出【说话人】「不能说」和【目的】「别说漏」里的事；不能出现【世界】「不能出现」里的词；人物不知道自己在故事里。
- 【目的】的「要办成」必须办到；「心里想的」**绝不能直接说出来**，不能把心里的打算原样讲出口，只能让它从语气、话题、用词里透出来。
- 口头禅和标志性说法偶尔用一次就够，不要每段都用、不要每句都用。
- 写完数一数：每句（不算动作括号）都不能超过「每句最多」的字数，超了就拆短或删字。
- 对方说了世界外的词（比如「AI」），人物可以当成怪话原样复述一下，但自己不会主动用这些词。
- 台词用普通话写；地方味只靠少量本地词和称呼体现，不要整句写方言（粤语、西南话等）。"""


def speaker_name(it, person):
    return person.get("name") or it["speaker"]


def build_rules(it):
    speakers = [it["speaker"]] + (["（对方）"] if it["speakers_extra"] else [])
    return {"use": it["use"], "lines": list(it["lines"]), "max_chars": it["max_chars"],
            "emotions": WORLDS[it["world"]]["emotions"], "actions": it["actions"], "speakers": speakers}


def card_request(it, world, person):
    t = TROUBLES[it["trouble"]]
    need = [f"用法：{'接一句' if it['use'] == 'reply' else '写整段'}",
            f"这段话的目的类型：{it['goal']}",
            f"对方出的难题：{t}",
            f"说话人和对方的关系：{it['relation']}",
            f"{'初次见面' if it['first_meet'] else '以前见过'}",
            "要写「已发生的事」" if it["happened"] else "不写「已发生的事」",
            "旁边还有别人在场" if it["others"] else "没有别人在场",
            f"对方此刻的样子：{it['listener_now']}" if it["listener_now"] else "对方样子平常（listener.now 为空）",
            f"说话人此刻：{it['speaker_now']}" if it["speaker_now"] else "说话人状态平常（speaker_now 为空）",
            f"可用表情：{'、'.join(WORLDS[it['world']]['emotions'])}"]
    if it["speakers_extra"]:
        need.append("这一段对方也可以接一句话")
    return (f"【世界设定】\n{json.dumps(world, ensure_ascii=False)}\n\n"
            f"【说话人档案】{speaker_name(it, person)}\n{json.dumps(person, ensure_ascii=False)}\n\n"
            "【这一场的要求】\n" + "\n".join("- " + n for n in need))


# ---------- 步骤 ----------
def load(name):
    p = OUT / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def save(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


USAGE = []


def step_worlds():
    worlds = load("worlds.json") or {}
    todo = [w for w in WORLDS if w not in worlds]

    def one(w):
        data, u = chat_json(SYS_WORLD, f"一句话设定：{WORLDS[w]['brief']}", model=MODEL)
        USAGE.append(u)
        return w, data
    with ThreadPoolExecutor(4) as ex:
        for w, data in ex.map(one, todo):
            worlds[w] = data
    save("worlds.json", worlds)
    return worlds


def step_people(worlds):
    people = load("people.json") or {}
    todo = [p for p in PEOPLE if p not in people]

    def one(p):
        w = PEOPLE[p]["world"]
        data, u = chat_json(SYS_PERSON, f"【世界设定】\n{json.dumps(worlds[w], ensure_ascii=False)}\n\n"
                                        f"【人物】名字：{p}。一句话设定：{PEOPLE[p]['brief']}", model=MODEL)
        USAGE.append(u)
        return p, data
    with ThreadPoolExecutor(5) as ex:
        for p, data in ex.map(one, todo):
            people[p] = data
            save("people.json", people)
    save("people.json", people)
    return people


def step_cards(worlds, people, plan):
    cards = load("cards.json") or {}
    todo = [it for it in plan if it["id"] not in cards]

    reviews = load("card_review.json") or {}

    def one(it):
        world, person = worlds[it["world"]], people[it["speaker"]]
        request, log = card_request(it, world, person), []
        for _ in range(TRIES):
            data, u = chat_json(SYS_CARD, request, model=MODEL)
            USAGE.append(u)
            card = assemble(it, world, person, data)
            if len(render_card(card)) > MAX_CARD:
                problems = [f"整张卡超过 {MAX_CARD} 字"]
            elif CARD_REVIEW:
                problems = CARD_REVIEW(card)
            else:
                problems = []
            log.append(problems)
            if not problems:
                break
            # 只告诉 DeepSeek 哪里打架，让它整张重写（不给示范、不改它的字）
            request = card_request(it, world, person) + "\n\n（上一版戏卡有这些硬伤：" + "；".join(problems) + "。请整张重写。）"
        return it["id"], data, log
    with ThreadPoolExecutor(8) as ex:
        for cid, data, log in ex.map(one, todo):
            cards[cid] = data
            reviews[cid] = log
            save("cards.json", cards)
            save("card_review.json", reviews)
    return cards


def norm_secret(x):
    """DeepSeek 有时把秘密写成 {秘密, 对外说法}，统一成一句话。"""
    if isinstance(x, str):
        return x
    vals = [str(v) for v in x.values() if v]
    return vals[0] + (f"（对外说：{vals[1]}）" if len(vals) > 1 else "")


def as_list(x):
    """DeepSeek 偶尔把秘密写成一整段字符串：当成一条，不拆字、不改字。"""
    return [x] if isinstance(x, (str, dict)) else list(x)


def assemble(it, world, person, part):
    rules = build_rules(it)
    listener = part["listener"]
    rules["speakers"] = [speaker_name(it, person)] + ([listener["name"]] if it["speakers_extra"] else [])
    return {
        "world": {"tone": world["tone"], "address": part.get("address", ""), "terms": part.get("terms", []),
                  "banned": world["banned"], "offstage": world["offstage"]},
        "speaker": {"name": speaker_name(it, person), **{k: person.get(k, "") for k in
                    ["summary", "personality", "speech", "examples", "wants", "fears", "knows"]},
                    "secrets": [norm_secret(x) for x in as_list(person.get("secrets", []))],
                    "now": part.get("speaker_now", "")},
        "listener": {"name": listener["name"], "summary": listener.get("summary", ""),
                     "view": listener.get("view", ""), "now": listener.get("now", ""),
                     "others": part.get("others", [])},
        "scene": part["scene"],
        "history": part.get("history", []),
        "goal": part["goal"],
        "rules": rules,
    }


LINE_RE = re.compile(r"^([^\[\]:：]+?)(?:\[([^\]]+)\])?\s*[:：]\s*(.+)$")


def parse_lines(text):
    out, bad = [], []
    for raw in text.strip().splitlines():
        raw = raw.strip()
        if not raw:
            continue
        m = LINE_RE.match(raw)
        if m:
            out.append({"name": m.group(1).strip(), "emotion": (m.group(2) or "").strip(), "text": m.group(3).strip()})
        else:
            bad.append(raw)
    return out, bad


def step_lines(worlds, people, plan, cards):
    samples = {s["id"]: s for s in (load("samples.json") or [])}
    todo = [it for it in plan if it["id"] not in samples]

    def one(it):
        card = assemble(it, worlds[it["world"]], people[it["speaker"]], cards[it["id"]])
        prompt, failed = render_card(card), []
        for attempt in range(1, TRIES + 1):
            text, u = chat(SYS_LINES, prompt, model=MODEL)
            USAGE.append(u)
            out, bad = parse_lines(text)
            smp = {"id": it["id"], "source": "deepseek", "use": it["use"], "card": card, "output": out,
                   "meta": {"world": it["world"], "date": DATE, "generator": MODEL, "kept_by": "",
                            "notes": "", "plan": it, "raw": text, "unparsed": bad, "attempts": attempt,
                            "failed_attempts": failed}}
            problems = check(smp)
            if not problems and LINES_REVIEW:
                problems = LINES_REVIEW(card, out)
            if not problems:
                break
            failed.append({"raw": text, "problems": problems})
            # 机器检查不合格：只告诉 DeepSeek 哪条规矩没守住，让它整段重写（不给示范、不改它的字）
            prompt = render_card(card) + "\n\n（上一版没守住这些规矩：" + "；".join(problems) + "。请整段重写。）"
        return smp
    with ThreadPoolExecutor(8) as ex:
        for s in ex.map(one, todo):
            samples[s["id"]] = s
            save("samples.json", [samples[k] for k in sorted(samples)])
    return [samples[k] for k in sorted(samples)]


CANTONESE = set("嘅唔咗喺嗰啲冇畀佢哋咁嚟睇嘢乜")   # 2026-11-18 Ken 定：不整句说粤语
GAME_WORDS = ["NPC", "玩家", "存档", "任务", "等级", "经验值", "系统提示", "游戏"]


def check(s):
    """机器能查的几条（docs/戏卡格式.md 第五节）。秘密说没说漏要人看。"""
    r, problems = s["card"]["rules"], []
    out = s["output"]
    if s["meta"]["unparsed"]:
        problems.append(f"有 {len(s['meta']['unparsed'])} 行读不懂")
    if out and not any(o["name"] == r["speakers"][0] for o in out):
        problems.append("说话人自己一句都没说")
    lo, hi = r["lines"]
    if not lo <= len(out) <= hi:
        problems.append(f"句数 {len(out)} 不在 {lo}–{hi}")
    for o in out:
        body = re.sub(r"^（[^）]*）", "", o["text"]) if r["actions"] else o["text"]
        if len(body) > r["max_chars"]:
            problems.append(f"超字数 {len(body)}>{r['max_chars']}：{o['text'][:12]}…")
        if o["emotion"] and o["emotion"] not in r["emotions"]:
            problems.append(f"表情不在清单：{o['emotion']}")
        if o["name"] not in r["speakers"]:
            problems.append(f"不该开口：{o['name']}")
        if CANTONESE & set(o["text"]):
            problems.append(f"整句粤语（要普通话，只夹几个本地词）：{o['text'][:12]}…")
        if re.search(r"\[[^\]]*\]", o["text"]):
            problems.append(f"话里夹着方括号（表情标记写错位置）：{o['text'][-12:]}")
        if not r["actions"] and re.match(r"^[（(]", o["text"]):
            problems.append("不许写动作却写了")
        for w in GAME_WORDS:
            if w in o["text"] and s["meta"]["plan"]["trouble"] != "c":
                problems.append(f"出现游戏词：{w}")
    problems += inner_echo(s["card"], out)     # 2026-11-19：心里想的原话说出口（字面）
    return problems


def step_check(samples):
    lines = []
    for s in samples:
        p = check(s)
        s["meta"]["auto_check"] = p
        lines.append(f"{s['id']} {s['meta']['plan']['speaker']} {s['use']} 难题{s['meta']['plan']['trouble']}："
                     + ("通过" if not p else "；".join(p)))
    save("samples.json", samples)
    (OUT / "check.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    plan = load("plan.json")
    if not plan:
        plan = make_plan()[:LIMIT]
        save("plan.json", plan)
    if step in ("plan",):
        return
    worlds = step_worlds()
    if step == "worlds":
        return
    people = step_people(worlds)
    if step == "people":
        return
    cards = step_cards(worlds, people, plan)
    if step == "cards":
        return
    samples = step_lines(worlds, people, plan, cards)
    if step in ("lines", "all", "check"):
        step_check(samples)
    tin = sum(u.get("prompt_tokens", 0) for u in USAGE)
    tout = sum(u.get("completion_tokens", 0) for u in USAGE)
    print(f"本次调用 {len(USAGE)} 次，输入 {tin} token，输出 {tout} token")


if __name__ == "__main__":
    main()
