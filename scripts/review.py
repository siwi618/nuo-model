"""batch-2 起加的两道「审稿」（2026-11-19 Ken 定 batch-2 三条改进一起做）。

batch-1 挑稿时丢得最多的是：心里想的被说出口（53 / 147）、戏卡自相矛盾（14）、说漏秘密（12）、
称呼错 / 凭空造人（10）。这些靠字面比对抓不住（多半是换了说法），所以让 DeepSeek 当审稿员：
- review_card：戏卡写完先审，有硬伤就带着问题让 DeepSeek 重写戏卡；
- review_lines：台词通过机器检查后再审，有问题就带着问题让 DeepSeek 重写台词。
审稿员只给「哪里不对」，不改字、不写示范；审稿结果只用来决定重写，不进训练数据。
最后留不留仍由人 / Claude 挑。

另有一条不花钱的字面检查 inner_echo：台词和「心里想的」有连续 6 个字以上一样，直接算说出口
（batch-1 上校准：抓到被丢的 14 / 53，误伤留下的 12 / 453）。
"""

import re

from deepseek import chat_json
from render_card import render_card

PUNCT = re.compile(r"[，。！？、；：,.!?;:“”\"'‘’（）()…—\s]")
ECHO_MIN = 6


def longest_common(a, b):
    best = 0
    for i in range(len(a)):
        j = i + best + 1
        while j <= len(a) and a[i:j] in b:
            best = j - i
            j += 1
    return best


def inner_echo(card, output):
    """台词里有没有一大段和「心里想的」字面一样。返回问题列表（空 = 没问题）。"""
    inner = PUNCT.sub("", card.get("goal", {}).get("inner", "") or "")
    text = PUNCT.sub("", "".join(o["text"] for o in output))
    n = longest_common(inner, text) if inner else 0
    return [f"「心里想的」被原话说了出来（连续 {n} 个字一样）"] if n >= ECHO_MIN else []


SYS_CARD_REVIEW = """你是戏卡审稿员。下面是一张「戏卡」：演员读了它要写出【说话人】这一场的台词。戏卡是别人写的，常有自相矛盾的地方，你的工作就是把它们找出来。
请逐条对照检查（每条都要真的去比一比，不要只看大意）：
1. 要办成 × 秘密：「要办成」要做的事，是否需要说出、承认、暗示，或者顺口提起【说话人】「不能说」/ 秘密 /【目的】「别说漏」里的东西？（例：秘密是替山背货拉客，要办成却是「勾一句山背货生意」→ 打架）
2. 要办成 × 别说漏：两者是否要求相反的事？（例：要办成「说自己听不明白」，别说漏却是「别承认听不懂」→ 打架）
3. 心里想的 × 秘密：「心里想的」是否直接写进了秘密或「别说漏」的内容，或者和「别说漏」说法冲突（地点、人、事对不上）？
4. 心里想的 × 要办成：方向是否相反到演不出来？（例：要办成约他明早见，心里却盼他白跑一趟）
5. 要办成 × 场景：「要办成」里的人、物、谁给谁，和【场景】【前文】【对方】写的事实是否对得上？（例：浆果酱是对方拿来换的，要办成却让对方「带走一罐浆果酱」→ 打架）
6. 称呼和关系：【对方】的名字 / 身份，和「要办成」「前文」「场景」里对他的称呼是否一致？关系是否前后矛盾（例：写亲近 / 老朋友，又写头回见面）？
7. 戏卡 × 档案：戏卡里的事实是否和【说话人】档案的对外说法、经历冲突？
口是心非、话里有话、人物故意撒谎都是正常的戏，不算问题。每个问题都必须能引用卡里两处原话证明它们打架；引不出原话的不要报。
只输出 JSON：{"problems": [{"type": "上面七条之一的名字", "why": "一句话，引用两处原话"}]}，都没问题就输出 {"problems": []}。"""

SYS_LINES_REVIEW = """你是一个严格但公道的审稿员。下面是一张「戏卡」和演员按它写的台词。请检查台词有没有这四类问题：
1. 心里话说出口：【目的】里「心里想的」是潜台词，只能从语气、话题、用词里透出来。如果某句台词把它的意思直接讲了出来（换了说法、意思一样也算），就是问题。只是让人隐约感觉到，不算。
2. 说漏秘密：台词说出、承认或明显暗示了【说话人】「不能说」或【目的】「别说漏」里的事（包括对方没问、自己先提起）。
3. 没办成：台词没有去办【目的】「要办成」的事（连提都没提，或者办的是另一件事）。「接一句」只要朝这个方向推进了就算办了。
4. 称呼或人物错：叫错【对方】的名字或身份，凭空提到卡里没有、又和卡冲突的人或事，或者句子前后说不通。
拿不准的不要报。只输出 JSON：{"problems": [{"type": "上面四类之一", "line": 第几句（从 1 数，整体问题写 0）, "why": "一句话，引用台词原话"}]}，没有问题就输出 {"problems": []}。"""


def _ask(system, user, model, usage):
    data, u = chat_json(system, user, model=model)
    if usage is not None:
        usage.append(u)
    probs = data.get("problems", []) if isinstance(data, dict) else []
    return [f"{p.get('type', '')}：{p.get('why', '')}" for p in probs if isinstance(p, dict)]


def review_card(card, model, usage=None):
    return _ask(SYS_CARD_REVIEW, render_card(card), model, usage)


def review_lines(card, output, model, usage=None):
    lines = "\n".join(f"{i}. {o['name']}{'[' + o['emotion'] + ']' if o.get('emotion') else ''}: {o['text']}"
                      for i, o in enumerate(output, 1))
    return _ask(SYS_LINES_REVIEW, f"{render_card(card)}\n\n【台词】\n{lines}", model, usage)
