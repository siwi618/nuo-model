"""小说新取法（novel-3，2026-11-20 Ken 定）：用作者写出来的心思当「心里想的」，后面真正说出口的话当答案。

为什么：「心里话说出口」是所有版本共有的毛病（专项卷 22–38%）。小说里常见「叙述写出心思 → 人物开口说了别的话」：
  周瑞家的……心中难却其意，二则也要显弄自己的体面。听如此说，便笑说道：「姥姥你放心……」
心思由作者写明，答案天然不会把它说破——这是 DeepSeek 编不出来的训练信号。

和 novel.py 的区别（抓原文、繁转简、切引语、标说话人都照用 novel.py）：
1. 取段：答案前面必须有心思——
   A. 紧挨着答案的叙述里有心思词（心中 / 心下 / 暗想 / 寻思 / 想道……），取含心思词的那一句；
   B. 答案前面紧挨着一句内心独白（「心中想道：『……』」），中间的叙述不超过 20 字。
   前文 0–4 句（可以没有）；答案 1–2 句，两句中间不能夹叙述（11-19「小说样本的坑」④）。
2. 写卡：把【心思原文】交给 DeepSeek，由它用现代白话转述成「心里想的」（不能照抄、不能用答案的词）；
   心思如果不是答案说话人的（是对方或旁人的），DeepSeek 标出来，这条不要。
3. 修 11-19「小说样本的坑」扩量前要做的几条：写卡说明里明确禁止把答案的词写进【说话】【本场的词】【要办成】、
   视角 / 性别 / 称谓弄反、凭空编人名、把答案之后的事写进卡；片段里标出「答案之后的内容只供理解」；
   字面检查：答案里任意 4 字出现在【说话】【本场的词】【要办成】【心里想的】就重写（原来只查 8 字）。

书单（Ken 2026-11-20 定）：官场现形记（加回目）、红楼梦（加回目）、二十年目睹之怪现状（新书）；
二十年目睹试切 10 回只取到 2 条（第一人称，少写心思词），换成儒林外史（加回目）。
答案是原文一个字不改（source = human）；Claude 只写代码和挑稿（留 / 不留）。

用法：NUO_N=每本书最多几条（默认 40）.venv/bin/python scripts/novel3.py [fetch / segment / label / cards / all]
     结果在 data/novel-3/（不进 git）。会跳过 novel-1、novel-2 用过的句子。
"""

import json
import os
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

os.environ.setdefault("NUO_RUN", "novel-3")
os.environ.setdefault("NUO_PREFIX", "nv3")
os.environ.setdefault("NUO_EXCLUDE", "novel-1,novel-2")
os.environ.setdefault("NUO_N", "40")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import novel as N  # noqa: E402
import pilot  # noqa: E402
from deepseek import chat_json  # noqa: E402
from review import inner_echo, review_card, review_lines  # noqa: E402

DATE = "2026-11-20"


def pages(fmt, nums):
    return [fmt.format(n) for n in nums]


# 回目：避开 novel-1 / novel-2 用过的；红楼梦避开牙牌令、诗社、灯谜一类（答案多是诗句）
N.BOOKS = {
    "官场现形记": {"author": "李宝嘉（李伯元）", "translator": "", "world": "晚清官场",
               "pages": pages("官場現形記/{:02d}", [8, 9, 10, 11, 12, 13, 19, 20, 21, 22])},
    "红楼梦": {"author": "曹雪芹 / 高鹗", "translator": "", "world": "清代京城贵族大宅",
            "pages": pages("紅樓夢/第{:03d}回", [7, 16, 24, 25, 33, 34, 35, 45, 46, 57, 60, 61])},
    # 二十年目睹之怪现状（11-20 试过 10 回）：第一人称叙述，很少写「心中 / 暗想」，10 回只取到 2 条 → 换儒林外史（加回目）
    "儒林外史": {"author": "吴敬梓", "translator": "", "world": "清代江南乡绅、读书人",
              "pages": pages("儒林外史/第{:02d}回", [9, 10, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])},
}

# 叙述里的心思词（11-20 统计用的同一组）
MIND = re.compile(r"(心中|心下|心里|心内|心头|心上|肚里|肚内|暗想|暗忖|寻思|自思|自忖|暗道|想道|心想)")
SENT_END = "。！？；"


# 吴语（官场现形记里妓女、娘姨的苏白）：Ken 11-18 定不整句说方言，答案里有一个就不要
WU = re.compile(r"倪|勿|覅|嘸|阿是|哉|俚|伲|耐格|耐末|格末|故歇|晏歇|末事|搭仔|说仔|来哚|勒浪|哚")


def wu(s):
    return len(WU.findall(s))


def thought_from_narration(text):
    """叙述里含心思词的那一句（到句号为止）；去掉句尾的「便笑道：」之类。没有就返回 None。"""
    t = text.strip()
    ms = list(MIND.finditer(t))
    if not ms:
        return None
    m = ms[-1]
    start = max(t.rfind(c, 0, m.start()) for c in SENT_END) + 1
    ends = [t.find(c, m.end()) for c in SENT_END if t.find(c, m.end()) >= 0]
    sent = t[start:(min(ends) + 1) if ends else len(t)].strip()
    sent = re.sub(r"[，,]?[^，,。]{0,8}(道|说|问|答|笑)[：:，,]?$", "", sent).strip("，, ")
    return sent if 6 <= len(sent) <= 90 else None


def windows(book, page, ev):
    """答案前有心思的片段。返回的结构和 novel.windows 一样，另加 thought / anchor。"""
    out = []
    allq = [i for i, e in enumerate(ev) if e["k"] == "q"]
    speech = [i for i, e in enumerate(ev) if e["k"] == "q" and not e["thought"] and e.get("speech")]
    for t in speech:
        for n_t in (1, 2):
            if n_t == 2:
                k = speech.index(t)
                if k == 0:
                    continue
                t0 = speech[k - 1]
                between = "".join(ev[i]["t"] for i in range(t0 + 1, t) if ev[i]["k"] == "n")
                if t - t0 > 2 or len(between.strip()) > 6:      # 两句中间夹了叙述（动作 / 耳语）：不要
                    continue
                tgt = [t0, t]
            else:
                tgt = [t]
            first = tgt[0]
            # 心思：A 紧挨着的叙述里有心思词；B 紧挨着一句内心独白（中间叙述 ≤ 20 字）
            thought, anchor = None, None
            pre = ev[first - 1] if first >= 1 else None
            if pre and pre["k"] == "n":
                thought, anchor = thought_from_narration(pre["t"]), "叙述"
                if not thought and first >= 2 and ev[first - 2]["k"] == "q" and ev[first - 2]["thought"] \
                        and len(pre["t"].strip()) <= 20:
                    thought, anchor = ev[first - 2]["t"], "独白"
            elif pre and pre["k"] == "q" and pre["thought"]:
                thought, anchor = pre["t"], "独白"
            if not thought or len(thought) < 6:
                continue
            texts = [ev[i]["t"] for i in tgt]
            if any(len(x) > 80 or len(x) < 2 for x in texts) or sum(len(x) for x in texts) < 8:
                continue
            if any(wu(x) for x in texts):
                continue
            # 前文：答案之前（心思之前）最多 4 句说出口的话，叙述间隔照 novel.py 的规矩（锚点那段叙述放宽到 150 字）
            anchor_i = first - 1 if anchor == "叙述" else max(i for i in allq if i < first)
            prev = [i for i in speech if i < anchor_i][-4:]
            hist = []
            for i in reversed(prev):
                gap = "".join(ev[x]["t"] for x in range(i + 1, (hist[0] if hist else anchor_i)) if ev[x]["k"] == "n")
                if len(gap) > 60 or any(True for q in allq if i < q < (hist[0] if hist else anchor_i) and q not in prev):
                    break
                hist.insert(0, i)
            if anchor == "叙述":                 # 心思句要离开口近（160 字内）；前面的叙述长没关系，会当上下文给 DeepSeek
                pre_t = ev[first - 1]["t"].strip()
                if pre_t.rfind(thought[:6]) < len(pre_t) - 160:
                    continue
            if sum(wu(ev[i]["t"]) for i in hist) >= 2:
                continue
            if any(len(ev[i]["t"]) > 90 for i in hist):
                continue
            arch = sum(N.archaic(ev[i]["t"]) for i in tgt + hist)
            if arch > 1 or any(N.archaic(x) for x in texts):
                continue
            lo = max(0, (hist[0] if hist else anchor_i) - 3)
            out.append({"book": book, "page": page, "hist": hist, "tgt": tgt, "lo": lo,
                        "ctx_end": tgt[-1] + 2, "arch": arch, "thought": thought, "anchor": anchor})
    return out


def passage(seg, ev):
    """和 novel.passage 一样，另在答案之后插一行提示：之后的内容只供理解，不能写进卡（坑③）。"""
    lines, qno = [], {}
    for i in range(seg["lo"], min(seg["ctx_end"], len(ev))):
        if i == seg["tgt"][-1] + 1:
            lines.append("（——以下是答案之后发生的事，只供理解；戏卡里不能用，也不能写进「已发生的事」「眼下」「知道」——）")
        e = ev[i]
        if e["k"] == "n":
            lines.append(e["t"].strip())
        else:
            qno[i] = len(qno) + 1
            lines.append(f"[Q{qno[i]}]「{e['t']}」")
    return "\n".join(lines), qno


def step_segment(raw):
    """照 novel.step_segment；前文可以没有，所以片段的起点 = 前文第一句，没有前文就从心思那里算。"""
    rnd = N.random.Random(20261120)
    segs, used_texts = {}, N.used_before()
    for book, b in N.BOOKS.items():
        cands = []
        for page in b["pages"]:
            ev = N.events_novel(raw[page])
            cands += [w for w in windows(book, page, ev)
                      if not any(ev[i]["t"] in used_texts for i in w["hist"] + w["tgt"])]
        rnd.shuffle(cands)
        chosen, used = [], set()
        for w in cands:                        # 同一处对话只取一个片段
            span = range((w["hist"] or [w["tgt"][0] - 2])[0], w["tgt"][-1] + 1)
            if any((w["page"], t) in used for t in span):
                continue
            chosen.append(w)
            used |= {(w["page"], t) for t in span}
            if len(chosen) >= N.PER_BOOK * 2:
                break
        for k, w in enumerate(chosen):
            segs[f"{N.PREFIX}-{list(N.BOOKS).index(book) + 1}{k + 1:03d}"] = w
        anchors = Counter(w["anchor"] for w in chosen)
        print(f"{book}：候选 {len(cands)}，取 {len(chosen)}（{dict(anchors)}）")
    N.save("segments.json", segs)
    return segs


N.windows = windows
N.passage = passage

SYS_CARD3 = N.SYS_NOVEL_CARD.replace(
    '- "goal"：{"must": 这几句话要办成的事（具体，但不引用原话）, "inner": 说话人心里想的、没说出口的（潜台词，不能和答案字面相近）, "dont_leak": [这场要藏住、不能说出口的事；没有就空列表]}',
    '- "goal"：{"must": 这几句话要办成的事（具体，但不引用原话、不逐项复述答案）, "inner": 根据【心思原文】用现代白话转述说话人此刻的心思（不能照抄原文，不能用答案里的词）, "dont_leak": [这场要藏住、不能说出口的事；没有就空列表]}\n'
    '- "thought_is_speaker"：【心思原文】写的是不是【答案】说话人自己的心思（是 true；是对方或旁人的、看不出来是谁的，false）')
assert SYS_CARD3 != N.SYS_NOVEL_CARD
SYS_CARD3 += """
另外几条（以前常错）：
- 【说话】【本场的词】【要办成】里不能出现答案的词句；口头禅不能拿答案里的说法来编。
- 「X 怎么看他」是说话人看对方，别写反；人物的性别、辈分、称谓关系照原著，别弄反；不要编片段里没有的人名。
- 片段里标了「答案之后发生的事」的部分只供理解：不能写进「已发生的事」「眼下」「知道」，也不能写成要办成的事。"""


def leak4(card, output):
    """答案里任意 4 字出现在【说话】【本场的词】【要办成】【心里想的】：算抄答案。"""
    s, g, w = card["speaker"], card["goal"], card["world"]
    fields = {"说话": s.get("speech", ""), "本场的词": "、".join(w.get("terms") or []),
              "要办成": g.get("must", ""), "心里想的": g.get("inner", "")}
    for o in output:
        t = re.sub(r"[，。！？、；：…—「」『』\s]", "", o["text"])
        for k in range(0, max(0, len(t) - 3)):
            for name, body in fields.items():
                if t[k:k + 4] in body:
                    return [f"【{name}】里用了答案的词「{t[k:k + 4]}」"]
    return []


def card_one(sid, seg, lab, ev):
    """照 novel.card_one，改三处：请求里给【心思原文】；多查 4 字抄答案；心思不是说话人的就不要。"""
    b = N.BOOKS[seg["book"]]
    text, qno = passage(seg, ev)
    tgt = [str(i) for i in seg["tgt"]]
    names = {lab[i]["speaker"] for i in tgt}
    speaker = lab[tgt[0]]["speaker"]
    hist = [str(i) for i in seg["hist"]]
    if len(names) != 1 or speaker in ("?", "") or any(lab[h]["speaker"] in ("?", "") for h in hist):
        return sid, None, ["说话人不清楚或答案不是同一个人说的"]
    listener = (lab[tgt[0]].get("to") or "").strip()
    if not listener or listener in ("?", speaker) or any(c in listener for c in "、，,和众"):
        return sid, None, ["没有明确的对方（自言自语 / 对众人 / 标不出）"]
    answer = "\n".join(f"{speaker}: {ev[int(i)]['t']}" for i in tgt)
    req = (f"【原著】{seg['book']}（{b['world']}）\n【片段】\n{text}\n\n"
           f"【答案】（{'、'.join('Q' + str(qno[int(i)]) for i in tgt)}，说话人：{speaker}，对方：{listener}）\n{answer}\n\n"
           f"【心思原文】{seg['thought']}\n\n"
           f"【前文】{'、'.join('Q' + str(qno[int(i)]) for i in hist) or '无'}（会原样放进戏卡，不用你写）")
    rules = {"use": "reply" if hist else "scene", "lines": [1, max(2, len(tgt))],
             "max_chars": next(m for m in (20, 40, 60, 80) if m >= max(len(ev[int(i)]["t"]) for i in tgt)),
             "emotions": N.EMOTIONS, "actions": False, "speakers": [speaker]}
    log, request, card, output = [], req, None, None
    for _ in range(N.TRIES):
        part, u = chat_json(SYS_CARD3, request, model=N.MODEL)
        pilot.USAGE.append(u)
        if part.get("thought_is_speaker") is False:
            log.append(["心思不是答案说话人的"])
            return sid, None, log[-1]
        emos = part.get("emotions", [])
        output = [{"name": speaker, "emotion": (emos[k] if k < len(emos) and emos[k] in N.EMOTIONS and emos[k] != "平常" else ""),
                   "text": ev[int(i)]["t"]} for k, i in enumerate(tgt)]
        card = {
            "world": {k: part.get("world", {}).get(k, "") for k in ("tone", "address", "terms", "banned", "offstage")},
            "speaker": {"name": speaker, **{k: part.get("speaker", {}).get(k, "") for k in
                        ("summary", "personality", "speech", "wants", "fears", "knows")},
                        "examples": [], "secrets": pilot.as_list(part.get("speaker", {}).get("secrets", [])),
                        "now": part.get("speaker_now", "")},
            "listener": {"name": listener, **{k: part.get("listener", {}).get(k, "") for k in ("summary", "view", "now")},
                         "others": part.get("others", [])},
            "scene": part.get("scene", {}),
            "history": [{"name": lab[h]["speaker"], "emotion": "", "text": ev[int(h)]["t"]} for h in hist],
            "goal": part.get("goal", {}),
            "rules": rules,
        }
        problems = N.leak_answer(card, output) + leak4(card, output) + inner_echo(card, output)
        if not (card["goal"].get("inner") or "").strip():
            problems.append("「心里想的」是空的")
        if not problems:
            problems = review_card(card, N.MODEL, pilot.USAGE)
        if not problems:
            problems = review_lines(card, output, N.MODEL, pilot.USAGE)
        log.append(problems)
        if not problems:
            break
        request = req + "\n\n（上一版戏卡有这些问题：" + "；".join(problems) + "。答案是原著原文不能改，请改戏卡，整张重写。）"
    smp = {"id": sid, "source": "human", "use": rules["use"], "card": card, "output": output,
           "meta": {"world": b["world"], "date": DATE, "generator": f"原著原文；戏卡 {N.MODEL}", "kept_by": "",
                    "notes": "", "card_review": log, "arch": seg["arch"], "method": "心思锚定",
                    "thought": seg["thought"], "anchor": seg["anchor"],
                    "ref": {"book": seg["book"], "page": seg["page"], "author": b["author"], "translator": b["translator"],
                            "url": "https://zh.wikisource.org/wiki/" + quote(seg["page"].replace(" ", "_")),
                            "license": "公有领域（中国、美国）"}}}
    return sid, smp, log[-1]


def step_cards(raw, segs, labels):
    samples = {s["id"]: s for s in (N.load("samples.json") or [])}
    dropped = N.load("dropped.json") or {}
    evs, todo, per_book = {}, [], {}
    for sid, seg in segs.items():
        if seg["page"] not in evs:
            evs[seg["page"]] = N.events_novel(raw[seg["page"]])
        if sid in samples or sid in dropped or sid not in labels:
            continue
        have = sum(1 for s in samples.values() if s["meta"]["ref"]["book"] == seg["book"])
        if have + per_book.get(seg["book"], 0) >= N.PER_BOOK:
            continue
        per_book[seg["book"]] = per_book.get(seg["book"], 0) + 1
        todo.append((sid, seg, labels[sid], evs[seg["page"]]))
    with ThreadPoolExecutor(8) as ex:
        for sid, smp, last in ex.map(lambda a: card_one(*a), todo):
            if smp:
                smp["meta"]["auto_check"] = last
                samples[sid] = smp
                N.save("samples.json", [samples[k] for k in sorted(samples)])
            else:
                dropped[sid] = last
                N.save("dropped.json", dropped)
    lines = [f"{s['id']} {s['meta']['ref']['book']} {s['card']['speaker']['name']}（{s['meta']['anchor']}）："
             + ("通过" if not s["meta"]["auto_check"] else "；".join(s["meta"]["auto_check"])) for s in samples.values()]
    (N.OUT / "check.txt").write_text("\n".join(sorted(lines)) + "\n", encoding="utf-8")
    return samples


def main():
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    raw = N.step_fetch()
    if step == "fetch":
        return
    segs = N.load("segments.json") or step_segment(raw)
    if step == "segment":
        return
    labels = N.step_label(raw, segs)
    if step == "label":
        return
    samples = step_cards(raw, segs, labels)
    ok = sum(1 for s in samples.values() if not s["meta"]["auto_check"])
    dropped = N.load("dropped.json") or {}
    print(f"样本 {len(samples)} 条，审稿全过 {ok} 条；不要的 {len(dropped)} 条")
    tin = sum(u.get("prompt_tokens", 0) for u in pilot.USAGE)
    tout = sum(u.get("completion_tokens", 0) for u in pilot.USAGE)
    print(f"本次调用 {len(pilot.USAGE)} 次，输入 {tin} token，输出 {tout} token")


if __name__ == "__main__":
    main()
