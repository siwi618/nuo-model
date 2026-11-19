"""人写的台词（公有领域小说 / 剧本）→ 训练样本（2026-11-19 Ken 定小试，约 100 条）。

和 DeepSeek 写的样本不同：**答案（output）是书里的原句，一个字不改**（只做繁转简、统一几个异体字、
去掉剧本里的括号舞台说明）。DeepSeek 只做两件事：标每句是谁说的（不碰原文），按片段写档案 / 戏卡（输入）。
人物档案里的「例句」小试先留空（不让 DeepSeek 编，也不冒险把别人的话安到他头上）。

书单（Ken 2026-11-19 定，中美两边都是公有领域，见 KB「公有领域书单核实」）：
红楼梦、儒林外史、官场现形记、水浒传（120 回本）、鲁迅《呐喊》《彷徨》、朱生豪译莎士比亚（仲夏夜之梦、汉姆莱脱）。
底本：维基文库（zh.wikisource.org），样本 meta.ref 里记书名、篇章、作者 / 译者、链接。

防「满嘴之乎者也」（Ken：别把模型调成清朝老头）：
- 答案句和前文里文言虚词（之乎者矣焉哉兮欤汝吾尔曰）多的片段不要；
- 戏卡全部用现代白话写，【世界】写清年代，让模型学「口气跟着卡走」。

用法：.venv/bin/python scripts/novel.py [fetch / segment / label / cards / all]
     NUO_N=每本书最多几条（默认 24）。结果在 data/novel-1/（不进 git）。
     第二批：NUO_RUN=novel-2 NUO_PREFIX=nv2 NUO_EXCLUDE=novel-1 NUO_N=14 …（不重复第一批用过的句子）
"""

import html
import json
import os
import random
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pilot  # noqa: E402
from deepseek import chat_json  # noqa: E402
from render_card import render_card  # noqa: E402
from review import inner_echo, review_card, review_lines  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / os.environ.get("NUO_RUN", "novel-1")
PER_BOOK = int(os.environ.get("NUO_N", "24"))
PREFIX = os.environ.get("NUO_PREFIX", "nv")        # 第二批起换前缀（nv2），免得编号撞
MODEL = pilot.MODEL
API = "https://zh.wikisource.org/w/api.php"
TRIES = 3
EMOTIONS = ["平常", "生气", "害怕", "难过", "惊讶", "傲慢", "冷笑", "无语", "高兴"]

BOOKS = {
    "红楼梦": {"author": "曹雪芹 / 高鹗", "translator": "", "world": "清代京城贵族大宅",
             # 第 040 回大半是牙牌令（答案多是诗句、牌名），11-19 挑稿后换成第 041 回
             "pages": ["紅樓夢/第006回", "紅樓夢/第039回", "紅樓夢/第041回", "紅樓夢/第044回", "紅樓夢/第055回", "紅樓夢/第068回"]},
    "儒林外史": {"author": "吴敬梓", "translator": "", "world": "清代江南乡绅、读书人",
              "pages": ["儒林外史/第03回", "儒林外史/第04回", "儒林外史/第05回", "儒林外史/第06回", "儒林外史/第13回",
                        "儒林外史/第02回", "儒林外史/第07回", "儒林外史/第08回", "儒林外史/第11回"]},
    "官场现形记": {"author": "李宝嘉（李伯元）", "translator": "", "world": "晚清官场",
               "pages": ["官場現形記/01", "官場現形記/02", "官場現形記/03", "官場現形記/04", "官場現形記/05",
                         "官場現形記/06", "官場現形記/07", "官場現形記/17", "官場現形記/18", "官場現形記/24"]},
    "水浒传": {"author": "施耐庵 / 罗贯中", "translator": "", "world": "北宋江湖、市井",
            "pages": ["水滸傳 (120回本)/第003回", "水滸傳 (120回本)/第024回", "水滸傳 (120回本)/第025回", "水滸傳 (120回本)/第038回", "水滸傳 (120回本)/第045回"]},
    "鲁迅小说": {"author": "鲁迅", "translator": "", "world": "民国初年江南乡镇",
             "pages": ["孔乙己", "阿Ｑ正傳", "故鄉", "風波", "明天", "祝福", "離婚", "肥皂", "高老夫子",
                       "藥", "端午節", "社戲", "頭髮的故事", "在酒樓上", "孤獨者", "弟兄", "長明燈"]},
    "莎士比亚": {"author": "威廉·莎士比亚", "translator": "朱生豪", "world": "西洋古代宫廷",
             "pages": ["漢姆萊脫/第一幕", "漢姆萊脫/第二幕", "漢姆萊脫/第三幕", "漢姆萊脫/第四幕", "漢姆萊脫/第五幕",
                       "莎士比亞戲劇全集 (朱生豪)/仲夏夜之夢/第一幕", "莎士比亞戲劇全集 (朱生豪)/仲夏夜之夢/第二幕",
                       "莎士比亞戲劇全集 (朱生豪)/仲夏夜之夢/第三幕", "莎士比亞戲劇全集 (朱生豪)/仲夏夜之夢/第五幕"],
             "play": True},
}

ARCHAIC = set("之乎者矣焉哉兮欤汝吾尔曰")
# 内心独白 / 自言自语（11-19 挑稿：「心里不喜道」「心下发急说」「自言自语的说」都漏过）
THOUGHT = re.compile(r"(想道|暗想|自想|寻思|忖道|自忖|暗道|心[里中下内头][^。！？」』]{0,6}(道|说|想)|自言自语[^。！？」』]{0,4}(道|说)?|想着)[^。」』]{0,4}[：:，,]?$")
VARIANTS = {"嬾": "懒", "淸": "清", "偸": "偷", "鎗": "枪", "粧": "妆", "湏": "须", "呌": "叫"}
KEEP_ZHU = ("著名", "著作", "著述", "显著", "著称", "卓著", "土著", "原著", "名著", "巨著", "编著")


def load(name):
    p = OUT / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def save(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


# ---------- 1. 取原文 ----------
def fetch_page(page):
    """用 curl（这台机器上 Python 直接联网解析域名不稳）；空回复 / 限速就等一下重试。"""
    for attempt in range(6):
        out = subprocess.run(["curl", "-s", "--max-time", "60", "--retry", "3", "-A", "nuo-model-data-script/0.1", "-G", API,
                              "--data-urlencode", "action=parse", "--data-urlencode", f"page={page}",
                              "--data-urlencode", "prop=text", "--data-urlencode", "format=json",
                              "--data-urlencode", "formatversion=2"], capture_output=True, text=True).stdout
        try:
            h = json.loads(out)["parse"]["text"]
            break
        except (json.JSONDecodeError, KeyError):
            if attempt == 5:
                raise SystemExit(f"取不到 {page}：{out[:200]}")
            time.sleep(10 * (attempt + 1))
    time.sleep(1)
    h = re.sub(r"<(style|script|sup)[^>]*>.*?</\1>", "", h, flags=re.S)
    h = re.sub(r"<br\s*/?>|</p>|</div>|</tr>|</li>|</h\d>|</dd>", "\n", h)
    t = html.unescape(re.sub(r"<[^>]+>", "", h))
    return t


def normalize(t):
    import opencc
    t = opencc.OpenCC("t2s").convert(t)
    t = re.sub(r"〈（[^〉]*）〉", "", t)            # 官场现形记的注释
    t = re.sub(r"\[[a-z] ?\d+\]", "", t)           # 校勘标记
    t = t.replace("​", "")
    for a, b in VARIANTS.items():
        t = t.replace(a, b)
    keep = {w: f"\0{i}\0" for i, w in enumerate(KEEP_ZHU)}
    for w, k in keep.items():
        t = t.replace(w, k)
    t = t.replace("著", "着")
    for w, k in keep.items():
        t = t.replace(k, w)
    return "\n".join(l.strip() for l in t.splitlines() if l.strip())


def step_fetch():
    raw = load("raw.json") or {}
    for book, b in BOOKS.items():
        for page in b["pages"]:
            if page not in raw:
                raw[page] = normalize(fetch_page(page))
                save("raw.json", raw)
                print(f"取到 {book} {page}：{len(raw[page])} 字")
    return raw


# ---------- 2. 切片段 ----------
OPEN, CLOSE = "「『“", "」』”"


def events_novel(text):
    """把一回拆成 [叙述 / 引语] 的序列；只取最外层引号。"""
    ev, buf, depth, start, qbuf = [], "", 0, None, ""
    for ch in text:
        if depth == 0 and ch in OPEN:
            if buf.strip():
                ev.append({"k": "n", "t": buf})
            buf, depth, qbuf, start = "", 1, "", ch
        elif depth > 0:
            if ch in OPEN:
                depth += 1
                qbuf += ch
            elif ch in CLOSE:
                depth -= 1
                if depth == 0:
                    ev.append({"k": "q", "t": squash(qbuf.strip())})
                else:
                    qbuf += ch
            else:
                qbuf += ch
        else:
            buf += ch
    if buf.strip():
        ev.append({"k": "n", "t": buf})
    # 标出内心独白（「心中想道：「……」」）；只认「说出口的话」：前面紧挨着「道：」「说，」之类，或后面紧跟「他说」
    for i, e in enumerate(ev):
        if e["k"] == "q":
            prev = ev[i - 1]["t"] if i and ev[i - 1]["k"] == "n" else ""
            nxt = ev[i + 1]["t"] if i + 1 < len(ev) and ev[i + 1]["k"] == "n" else ""
            e["thought"] = bool(THOUGHT.search(prev.strip()[-12:]))
            e["speech"] = bool(SAY_BEFORE.search(prev.strip()[-10:]) or SAY_AFTER.search(nxt.strip()[:10]))
    return ev


SAY_BEFORE = re.compile(r"(道|说|问|答|叫|喊|骂|嚷|笑|叹|劝|回|吩咐|嘱咐|喝|告诉)[^。！？」』]{0,3}[：:，,]$")
SAY_AFTER = re.compile(r"^[^。！？「『]{0,6}(道|说|问|答|叫|喊|嚷)")


PLAY_LINE = re.compile(r"^([^\s　（(]{1,6})[　 ]+(.+)$")


def events_play(text):
    ev = []
    for line in text.splitlines():
        m = PLAY_LINE.match(line)
        if m and not line.startswith(("（", "(", "第")):
            speech = squash(re.sub(r"（[^）]*）|\([^)]*\)", "", m.group(2)).strip())
            sung ="唱" in m.group(2) or any(c in m.group(1) for c in "合同众二两齐")   # 唱的歌、几个人齐声说的不要
            if speech:
                ev.append({"k": "q", "t": speech, "abbr": m.group(1), "thought": False, "speech": not sung})
        else:
            ev.append({"k": "n", "t": line})
    # 扫描本丢了括号的舞台说明（「普下」「勒倒地」「向普」「同下」「旁白」）混进台词：整句不要；
    # 两个单字缩写拼成的（「罗基」）= 两人齐声，不要
    singles = {e["abbr"] for e in ev if e["k"] == "q" and len(e["abbr"]) == 1}
    if singles:
        stage = re.compile("(" + "|".join(map(re.escape, singles)) + r")[^\s，。！？]{0,2}(下|上|倒地|向)|同下|旁白|下场|比赛")
        for e in ev:
            if e["k"] == "q" and (stage.search(e["t"]) or (len(e["abbr"]) == 2 and set(e["abbr"]) <= singles)):
                e["speech"] = False
    return ev


def squash(s):
    """台词里扫描本留下的排版空格（「后 面」）去掉；只动台词内部，不动「缩写　台词」的分隔。"""
    return re.sub(r"(?<=[一-鿿])[ 　]+(?=[一-鿿])", "", s)


def archaic(s):
    return sum(1 for c in s if c in ARCHAIC)


def windows(book, page, ev):
    """候选片段：答案 = 1–2 句连续引语（同一人，后面 label 再核），前文 = 之前 1–4 句引语。"""
    out = []
    allq = [i for i, e in enumerate(ev) if e["k"] == "q"]
    qi = [i for i, e in enumerate(ev) if e["k"] == "q" and not e["thought"] and e.get("speech")]
    for j in range(2, len(qi)):
        for n_t in (1, 2):
            tgt = qi[j - n_t + 1: j + 1] if n_t == 2 else [qi[j]]
            if len(tgt) < n_t:
                continue
            hist = [i for i in qi[max(0, j - n_t - 3): j - n_t + 1]][-4:]
            if not hist:
                continue
            texts = [ev[i]["t"] for i in tgt]
            if any(len(t) > 60 or len(t) < 2 for t in texts) or sum(len(t) for t in texts) < 8:
                continue
            gap = sum(len(ev[i]["t"]) for i in range(hist[0], tgt[-1]) if ev[i]["k"] == "n")
            if gap > 200:                      # 中间叙述太长：多半已经换了场景
                continue
            # 前文必须是连着的一段对话（11-19 挑稿：跳场、隔夜、前文缺行是第一大毛病）：
            # 中间每段叙述不超过 60 字；中间不能夹着没选进来的引语（报条、唱词、心里话），否则前文会缺行
            if any(ev[i]["k"] == "n" and len(ev[i]["t"]) > 60 for i in range(hist[0], tgt[-1])):
                continue
            if any(hist[0] < i < tgt[-1] and i not in hist and i not in tgt for i in allq):
                continue
            if any(len(ev[i]["t"]) > 90 for i in hist):
                continue
            arch = sum(archaic(ev[i]["t"]) for i in tgt + hist)
            if arch > 1 or any(archaic(t) for t in texts):
                continue
            lo = max(0, hist[0] - 3)
            ctx = ev[lo: tgt[-1] + 2]
            out.append({"book": book, "page": page, "hist": hist, "tgt": tgt, "lo": lo,
                        "ctx_end": tgt[-1] + 2, "arch": arch})
    return out


def used_before():
    """之前几批（NUO_EXCLUDE=novel-1,...）用过的句子：答案和前文都不再选，免得重复。"""
    used = set()
    for run in filter(None, os.environ.get("NUO_EXCLUDE", "").split(",")):
        p = ROOT / "data" / run / "samples.json"
        for s in json.loads(p.read_text(encoding="utf-8")) if p.exists() else []:
            used |= {o["text"] for o in s["output"]} | {h["text"] for h in s["card"]["history"]}
    return used


def step_segment(raw):
    rnd = random.Random(20261119)
    segs = {}
    used_texts = used_before()
    for book, b in BOOKS.items():
        cands = []
        for page in b["pages"]:
            ev = (events_play if b.get("play") else events_novel)(raw[page])
            cands += [(w, ev) for w in windows(book, page, ev)
                      if not any(ev[i]["t"] in used_texts for i in w["hist"] + w["tgt"])]
        rnd.shuffle(cands)
        chosen, used = [], set()
        for w, ev in cands:                    # 同一处对话只取一个片段，免得前文重叠
            key = (w["page"], tuple(w["tgt"]))
            near = any((w["page"], t) in used for t in range(w["hist"][0], w["tgt"][-1] + 1))
            if near:
                continue
            chosen.append(w)
            used |= {(w["page"], t) for t in range(w["hist"][0], w["tgt"][-1] + 1)}
            if len(chosen) >= PER_BOOK * 2:   # 多留一倍，标说话人后会筛掉一些
                break
        for k, w in enumerate(chosen):
            segs[f"{PREFIX}-{list(BOOKS).index(book) + 1}{k + 1:03d}"] = w
        print(f"{book}：候选 {len(cands)}，取 {len(chosen)}")
    save("segments.json", segs)
    return segs


def passage(seg, ev):
    """给 DeepSeek 看的片段：叙述原样，引语编号 [Q1]…；剧本行写成「缩写：话」。"""
    lines, qno = [], {}
    for i in range(seg["lo"], min(seg["ctx_end"], len(ev))):
        e = ev[i]
        if e["k"] == "n":
            lines.append(e["t"].strip())
        else:
            qno[i] = len(qno) + 1
            who = f"{e['abbr']}：" if e.get("abbr") else ""
            lines.append(f"{who}[Q{qno[i]}]「{e['t']}」")
    return "\n".join(lines), qno


# ---------- 3. 标说话人（不碰原文） ----------
SYS_LABEL = """下面是一段小说或剧本原文，引语已编号为 [Q1] [Q2] …（剧本行前面的是角色名缩写）。
请只做标注，不要改写、不要概括原文。只输出 JSON：
{"speakers": {"Q1": "说话人全名", ...}, "to": {"Q1": "这句是当面对哪一个人说的（全名）", ...}}
说话人只能是原文里出现的人物；从上下文判断不了的写 "?"。剧本里的缩写要换成全名（按上下文和剧中人物判断）。
"to" 只写一个当面听这句话的人；自言自语、心里想的、对着一群人说的、念出来的（诗、牌名、帖子、报条）、判断不了的，一律写空字符串。"""


def label_one(sid, seg, ev):
    text, qno = passage(seg, ev)
    data, u = chat_json(SYS_LABEL, text, model=MODEL)
    pilot.USAGE.append(u)
    sp = {i: (data.get("speakers", {}) or {}).get(f"Q{n}", "?") for i, n in qno.items()}
    to = {i: (data.get("to", {}) or {}).get(f"Q{n}", "") for i, n in qno.items()}
    return sid, {str(i): {"speaker": sp[i], "to": to[i]} for i in qno}


def step_label(raw, segs):
    labels = load("labels.json") or {}
    evs = {}
    todo = []
    for sid, seg in segs.items():
        if sid in labels:
            continue
        b = BOOKS[seg["book"]]
        key = seg["page"]
        if key not in evs:
            evs[key] = (events_play if b.get("play") else events_novel)(raw[key])
        todo.append((sid, seg, evs[key]))
    with ThreadPoolExecutor(8) as ex:
        for sid, lab in ex.map(lambda a: label_one(*a), todo):
            labels[sid] = lab
            save("labels.json", labels)
    return labels


# ---------- 4. 写戏卡 ----------
SYS_NOVEL_CARD = """你在为一个「演员模型」准备训练资料。模型的任务是：读一张「戏卡」，用说话人的性格和口气写出台词。
下面给你一段文学原著的片段，其中【答案】几句是说话人这一场真正说的话（原文）。请根据整个片段，写出这张戏卡里除了台词以外的部分，让一个没读过原著的演员读了卡就能演出这几句。
铁律：
- 戏卡里任何地方都**不能引用或转述【答案】里的句子**，也不能透露答案的具体用词；只写情境、人物、目的。
- 全部用**现代白话**写（普通话、今天的人读得懂），不要用文言、不要用原著的旧式说法来写说明文字。人名、称呼、物件可以照原著。
- 只写片段里有、或从片段能合理推出的东西，不要凭空加剧情。
只输出 JSON，字段：
- "world"：{"tone": 这个世界的说话腔调（一两句，点明年代和阶层）, "address": 这一场用到的称呼（一句话）, "terms": 这一场会出现的 0–4 个世界特有的词（每个写成「词（简短解释）」）, "banned": 这个世界不能出现的词 3–6 个（现代词、外来词之类）, "offstage": 人物遇到世界外的话（比如「你是不是 AI」）会怎么反应（一句话）}
- "speaker"：{"summary": 一句话说他是谁, "personality": 性格（一两句）, "speech": 说话习惯（一两句）, "wants": 他想要什么, "fears": 他怕什么, "knows": 他知道的相关的事（一两句）, "secrets": [他藏着、不愿让对方知道的事，片段里看不出就空列表]}
- "listener"：{"name": 对方名字, "summary": 对方是谁（一句话）, "view": 说话人怎么看对方, "now": 对方此刻看得见的样子（没有就空字符串）}
- "others"：在场的其他人 [{"name","view"}]（没有就空列表）
- "scene"：{"place_time": 地点和时间（不超过 25 字）, "backstory": 刚发生了什么（不超过 40 字）, "happened": [已经发生、会影响说话人态度的事，写成「事实。→ 说话人因此怎样」，没有就空列表]}
- "speaker_now"：说话人此刻的身体状态和手上的事（没有就空字符串，不超过 30 字）
- "goal"：{"must": 这几句话要办成的事（具体，但不引用原话）, "inner": 说话人心里想的、没说出口的（潜台词，不能和答案字面相近）, "dont_leak": [这场要藏住、不能说出口的事；没有就空列表]}
- "emotions"：[【答案】每一句的表情，只能从这些里选：平常、生气、害怕、难过、惊讶、傲慢、冷笑、无语、高兴]"""


def card_one(sid, seg, lab, ev, page_ev_all):
    b = BOOKS[seg["book"]]
    text, qno = passage(seg, ev)
    tgt = [str(i) for i in seg["tgt"]]
    names = {lab[i]["speaker"] for i in tgt}
    speaker = lab[tgt[0]]["speaker"]
    hist = [str(i) for i in seg["hist"]]
    if len(names) != 1 or speaker in ("?", "") or any(lab[h]["speaker"] in ("?", "") for h in hist):
        return sid, None, ["说话人不清楚或答案不是同一个人说的"]
    # 对方 = DeepSeek 标的「这句是对谁说的」（11-19 挑稿：拿前文最后一个说话人当对方，9 条标错人）；
    # 标不出明确对象（自言自语、心里想、对众人嚷）的不要
    listener = (lab[tgt[0]].get("to") or "").strip()
    if not listener or listener in ("?", speaker) or any(c in listener for c in "、，,和众"):
        return sid, None, ["没有明确的对方（自言自语 / 对众人 / 标不出）"]
    # 例句：小试先留空（同一人在书里有好几个叫法，靠名字认人容易安错；以后要用就整回标说话人再取）
    examples = []
    answer = "\n".join(f"{speaker}: {ev[int(i)]['t']}" for i in tgt)
    req = (f"【原著】{seg['book']}（{b['world']}）\n【片段】\n{text}\n\n"
           f"【答案】（{'、'.join('Q' + str(qno[int(i)]) for i in tgt)}，说话人：{speaker}，对方：{listener}）\n{answer}\n\n"
           f"【前文】{'、'.join('Q' + str(qno[int(i)]) for i in hist)}（会原样放进戏卡，不用你写）")
    rules = {"use": "reply", "lines": [1, max(2, len(tgt))],
             "max_chars": next(m for m in (20, 40, 60, 80) if m >= max(len(ev[int(i)]["t"]) for i in tgt)),
             "emotions": EMOTIONS, "actions": False, "speakers": [speaker]}
    log, request = [], req
    for _ in range(TRIES):
        part, u = chat_json(SYS_NOVEL_CARD, request, model=MODEL)
        pilot.USAGE.append(u)
        emos = part.get("emotions", [])
        output = [{"name": speaker, "emotion": (emos[k] if k < len(emos) and emos[k] in EMOTIONS and emos[k] != "平常" else ""),
                   "text": ev[int(i)]["t"]} for k, i in enumerate(tgt)]
        card = {
            "world": {**{k: part.get("world", {}).get(k, "") for k in ("tone", "address", "terms", "banned", "offstage")}},
            "speaker": {"name": speaker, **{k: part.get("speaker", {}).get(k, "") for k in
                        ("summary", "personality", "speech", "wants", "fears", "knows")},
                        "examples": examples, "secrets": pilot.as_list(part.get("speaker", {}).get("secrets", [])),
                        "now": part.get("speaker_now", "")},
            "listener": {"name": listener, **{k: part.get("listener", {}).get(k, "") for k in ("summary", "view", "now")},
                         "others": part.get("others", [])},
            "scene": part.get("scene", {}),
            "history": [{"name": lab[h]["speaker"], "emotion": "", "text": ev[int(h)]["t"]} for h in hist],
            "goal": part.get("goal", {}),
            "rules": rules,
        }
        problems = leak_answer(card, output) + inner_echo(card, output)
        if not problems:
            problems = review_card(card, MODEL, pilot.USAGE)
        if not problems:
            problems = [p for p in review_lines(card, output, MODEL, pilot.USAGE)]
        log.append(problems)
        if not problems:
            break
        request = req + "\n\n（上一版戏卡有这些问题：" + "；".join(problems) + "。答案是原著原文不能改，请改戏卡，整张重写。）"
    smp = {"id": sid, "source": "human", "use": "reply", "card": card, "output": output,
           "meta": {"world": b["world"], "date": "2026-11-19", "generator": f"原著原文；戏卡 {MODEL}", "kept_by": "",
                    "notes": "", "card_review": log, "arch": seg["arch"],
                    "ref": {"book": seg["book"], "page": seg["page"], "author": b["author"], "translator": b["translator"],
                            "url": "https://zh.wikisource.org/wiki/" + quote(seg["page"].replace(" ", "_")),
                            "license": "公有领域（中国、美国）"}}}
    return sid, smp, log[-1]


def leak_answer(card, output):
    """戏卡（除前文外）不能抄答案：任何 8 字以上和答案一样就算抄。"""
    body = json.dumps({k: v for k, v in card.items() if k not in ("history", "rules")}, ensure_ascii=False)
    for o in output:
        t = re.sub(r"[，。！？、；：…—\s]", "", o["text"])
        for k in range(0, max(0, len(t) - 7)):
            if t[k:k + 8] in body:
                return [f"戏卡里抄了答案原话「{t[k:k + 8]}」"]
    return []


def step_cards(raw, segs, labels):
    samples = {s["id"]: s for s in (load("samples.json") or [])}
    evs = {}
    todo = []
    per_book = {}
    for sid, seg in segs.items():
        b = BOOKS[seg["book"]]
        if seg["page"] not in evs:
            evs[seg["page"]] = (events_play if b.get("play") else events_novel)(raw[seg["page"]])
        if sid in samples or sid not in labels:
            continue
        per_book.setdefault(seg["book"], 0)
        have = sum(1 for s in samples.values() if s["meta"]["ref"]["book"] == seg["book"])
        if have + per_book[seg["book"]] >= PER_BOOK:
            continue
        lab = labels[sid]
        tgt = [str(i) for i in seg["tgt"]]
        if len({lab[i]["speaker"] for i in tgt}) != 1 or lab[tgt[0]]["speaker"] in ("?", ""):
            continue
        per_book[seg["book"]] += 1
        todo.append((sid, seg, lab, evs[seg["page"]], evs[seg["page"]]))
    with ThreadPoolExecutor(8) as ex:
        for sid, smp, last in ex.map(lambda a: card_one(*a), todo):
            if smp:
                smp["meta"]["auto_check"] = last
                samples[sid] = smp
                save("samples.json", [samples[k] for k in sorted(samples)])
    lines = [f"{s['id']} {s['meta']['ref']['book']} {s['card']['speaker']['name']}："
             + ("通过" if not s["meta"]["auto_check"] else "；".join(s["meta"]["auto_check"])) for s in samples.values()]
    (OUT / "check.txt").write_text("\n".join(sorted(lines)) + "\n", encoding="utf-8")
    return samples


def main():
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    raw = step_fetch()
    if step == "fetch":
        return
    segs = load("segments.json") or step_segment(raw)
    if step == "segment":
        return
    labels = step_label(raw, segs)
    if step == "label":
        return
    samples = step_cards(raw, segs, labels)
    ok = sum(1 for s in samples.values() if not s["meta"]["auto_check"])
    print(f"样本 {len(samples)} 条，审稿全过 {ok} 条")
    tin = sum(u.get("prompt_tokens", 0) for u in pilot.USAGE)
    tout = sum(u.get("completion_tokens", 0) for u in pilot.USAGE)
    print(f"本次调用 {len(pilot.USAGE)} 次，输入 {tin} token，输出 {tout} token")


if __name__ == "__main__":
    main()
