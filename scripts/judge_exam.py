"""考卷内容盲评：让 DeepSeek 按挑稿标准判断每条台词的内容（格式另由 score_exam / exam_stats 查）。

盲评：评审只看到「戏卡 + 台词」，不知道是哪个版本、哪个种子；所有要评的条目打乱顺序再发。
已评过的（同一文件名 + 同一张卡 + 同样的台词）会跳过，可以随时中断、接着跑，也可以以后只补新版本。

用法（在仓库根目录跑）：
  .venv/bin/python scripts/judge_exam.py 结果.jsonl 输出文件或前缀 [更多...] [--exam exam/考卷.jsonl] [--workers 8]
  例：.venv/bin/python scripts/judge_exam.py data/x/judge.jsonl data/x/v1-pen data/x/v2a-pen
  新标准：加 --rule v2 / v3，结果存另一个文件（如 judge-r3.jsonl），别和 v1 的混在一起。
  v3（2026-11-22）：「语句不通或不合人设」拆成两项；「像不像」改成 4 个分项（各 0–4，带每档说明），按权重 3/3/2/2 加总成 0–40。
  「心里话说出口」的定义和 v1 一样，说出口率可以和 v1 直接比。
  v4（2026-11-23，人工核对后）：见 SYS_JUDGE_V4 上面的注释；problems 里用「该藏的说出口」，老定义另存「心里话说出口_旧」。
  v5（2026-11-24，分三层）：见 SYS_JUDGE_V5 上面的注释；逐处圈毛病（句号 + 原话 + 类型 + 轻重），--suggest 加改法。
然后：.venv/bin/python scripts/exam_stats.py --judge data/x/judge.jsonl --mode fixed,content,all v1=data/x/v1-pen v2a=...

判过 = 下面「硬伤」一条都没有，且「办成」≥ 1。
"""

import argparse
import json
import os
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from deepseek import chat_json  # noqa: E402
from exam_stats import expand  # noqa: E402
from render_card import render_card  # noqa: E402
from score_exam import parse_output  # noqa: E402

MODEL = os.environ.get("NUO_MODEL", "deepseek-flash")
HARD = ["心里话说出口", "说漏秘密", "称呼或人物错", "出戏", "语句不通或不合人设", "口头禅滥用"]

SYS_JUDGE_T = """你是戏剧台词的评审。下面是一张「戏卡」和一位演员按它写的台词。只评内容，**不管格式**（句数、字数、括号动作、表情标签另有程序检查，这里一律忽略）。

逐项判断：
1. 心里话说出口：【目的】里「心里想的」是潜台词，只能从语气、话题、用词里透出来。台词把它的意思直接讲出来（换了说法、意思一样也算）= 是。只是让人隐约感觉到 = 否。卡里没写「心里想的」= 否。{inner_extra}
2. 说漏秘密：台词说出、承认或明显暗示了【说话人】「不能说」或【目的】「别说漏」里的事（包括自己先提起）= 是。卡里没有秘密 = 否。
3. 办成：【目的】「要办成」的事。0 = 没去办（没提或办成了另一件事）；1 = 朝这个方向推进了，但没办完（「接一句」做到这一步就够了）；2 = 办到了。
4. 称呼或人物错：叫错【对方】的名字或身份；凭空提到卡里没有、又和卡冲突的人或事；把别人的话安到自己头上 = 是。
5. 出戏：主动用了世界之外的词或概念（游戏、AI、手机出现在古代等），或者对出戏的问题没有按【世界】的要求处理 = 是。
6. 语句不通或不合人设：句子说不通、前后矛盾、用错词，或者口气、用词明显不像【说话人】档案里的这个人 = 是。
7. 口头禅滥用：档案里的口头禅 / 例句被硬塞、一段里重复用 = 是。
8. 像不像：1–5 分，台词有多像【说话人】这个人在这个场景里会说的话（口气、看法、潜台词）。5 = 很鲜活；3 = 不出错但平；1 = 完全不像。

拿不准的硬伤不要报。只输出 JSON：
{{"心里话说出口": true/false, "说漏秘密": true/false, "办成": 0/1/2, "称呼或人物错": true/false, "出戏": true/false,
 "语句不通或不合人设": true/false, "口头禅滥用": true/false, "像不像": 1-5, "理由": "一两句话，有硬伤就引用台词原话"}}"""

# 评审标准版本。v1 = 11-20 之前所有分数用的；v2（11-20 加）：「心里话说出口」排除两种情况——
# 说出来正是办「要办成」最自然的理由，或是【世界】规定的回法。换版本后旧分数要用新版本重评才能比，结果另存一个文件。
RULES = {
    "v1": "",
    "v2": "\n   但下面两种不算说出口（= 否）：①说出来的正是办成【目的】「要办成」那件事最自然、最必要的理由（例：叫人出门带伞，说一句「待会儿要下雨」）；"
          "②是【世界】里规定的回法（例：世界要求把听不懂的怪话一律当胡话、照规定的那句回）。只有把不该当面讲的心思、判断、盘算讲了出来才算。",
}


HARD3 = ["心里话说出口", "说漏秘密", "称呼或人物错", "出戏", "语句不通", "不合人设", "口头禅滥用"]
# 「像不像」分项与权重（参考 IB 文科评分：每项按档打分，重要的项权重大；每项只分 5 档，档太细评审反而不稳）
LIKE3 = {"口气": 3, "潜台词": 3, "具体": 2, "自然": 2}

SYS_JUDGE_V3 = """你是戏剧台词的评审。下面是一张「戏卡」和一位演员按它写的台词。只评内容，**不管格式**（句数、字数、括号动作、表情标签另有程序检查，这里一律忽略）。

一、硬伤（逐项判 true / false，拿不准的不要报）：
1. 心里话说出口：【目的】里「心里想的」是潜台词，只能从语气、话题、用词里透出来。台词把它的意思直接讲出来（换了说法、意思一样也算）= 是。只是让人隐约感觉到 = 否。卡里没写「心里想的」= 否。
2. 说漏秘密：台词说出、承认或明显暗示了【说话人】「不能说」或【目的】「别说漏」里的事（包括自己先提起）= 是。卡里没有秘密 = 否。
3. 称呼或人物错：叫错【对方】的名字或身份；凭空提到卡里没有、又和卡冲突的人或事；把别人的话安到自己头上 = 是。
4. 出戏：主动用了世界之外的词或概念（游戏、AI、手机出现在古代等），或者对出戏的问题没有按【世界】的要求处理 = 是。
5. 语句不通：句子本身说不通——语法错、搭配不当、用错词、前后矛盾、意思读不明白 = 是。只看话本身通不通，不看像不像这个人。
6. 不合人设：话是通的，但口气、身份、见识、用词明显不是【说话人】档案里的这个人会有的（例：胆小的丫鬟对主子出言顶撞、古人说现代腔、粗人满口书面语）= 是。只是平淡、没特色不算。
7. 口头禅滥用：档案里的口头禅 / 例句被硬塞、一段里重复用 = 是。

二、办成：【目的】「要办成」的事。0 = 没去办（没提或办成了另一件事）；1 = 朝这个方向推进了，但没办完（「接一句」做到这一步就够了）；2 = 办到了。

三、像不像：4 个分项，每项 0–4 分（整数），先看每档说明再定：
A. 口气：像不像【说话人】这个人。0 = 像另一个人；1 = 中性，听不出是谁；2 = 有一两处像（某个词或某种口气）；3 = 通篇口气对，用词带这个人的身份和性格；4 = 一听就是他，换个人说不出这样的话。
B. 潜台词：「心里想的」怎么处理（卡里没写「心里想的」时，看有没有透出这个人此刻的态度和情绪）。0 = 直接说了出来，或和心思反着来；1 = 完全没透出，只是平铺直叙地办事；2 = 隐约能感到；3 = 从话题、语气、用词里自然透出；4 = 透得巧，点到为止又耐人回味。
C. 具体：0 = 空话套话；1 = 泛泛，放到别的场景也能用；2 = 有一处具体（场景里的物、事、人、称呼）；3 = 多处具体、贴着这个场景；4 = 细节选得准，有画面感。
D. 自然：像不像真人当场说的话。0 = 答非所问或像念稿；1 = 生硬、书面腔重；2 = 大体自然但有别扭处；3 = 自然顺口；4 = 很自然，节奏和语气像真在说话。
硬伤和分项各管各的：有硬伤不代表分项一定低，按实际打。

只输出 JSON：
{{"心里话说出口": true/false, "说漏秘密": true/false, "称呼或人物错": true/false, "出戏": true/false, "语句不通": true/false,
 "不合人设": true/false, "口头禅滥用": true/false, "办成": 0/1/2, "口气": 0-4, "潜台词": 0-4, "具体": 0-4, "自然": 0-4,
 "理由": "一两句话，有硬伤就引用台词原话"}}"""


# v4（2026-11-23，人工核对后改）：
#   ①「心里话说出口」原文不动（和 v1 / v3 可比），另加「该藏的说出口」：办事最自然必要的话、只说一部分或明显收着说、世界规定的回法不算。
#     problems / pass 用新的；旧的另存 rec["心里话说出口_旧"]、rec["pass_旧"]。
#   ②「语句不通」门槛放低：读着别扭、搭配生硬、要停下来想才懂的也算（评审以前在理由里写「别扭」却不报）。
#   ③ 表情标签可以拿来理解口气（人读的时候会看）。
#   ④ 潜台词分项：故意掩饰（撒谎、否认、打圆场、违心地鼓励）不算「反着来」。
HARD4 = ["该藏的说出口", "说漏秘密", "称呼或人物错", "出戏", "语句不通", "不合人设", "口头禅滥用"]

SYS_JUDGE_V4 = """你是戏剧台词的评审。下面是一张「戏卡」和一位演员按它写的台词。只评内容，**不管格式**（句数、字数、括号动作另有程序检查，这里一律忽略）。
台词前方括号里的表情标签（如 [傲慢]）不算格式问题，可以用来理解说话的口气（例：带 [冷笑] 的「您真是好心」可能是反话）。

一、硬伤（逐项判 true / false；拿不准的不要报，但第 6 条「语句不通」按它自己的门槛判）：
1. 心里话说出口：【目的】里「心里想的」是潜台词，只能从语气、话题、用词里透出来。台词把它的意思直接讲出来（换了说法、意思一样也算）= 是。只是让人隐约感觉到 = 否。卡里没写「心里想的」= 否。
2. 该藏的说出口：台词把「心里想的」里这个人此刻不该、也不会当面讲的心思讲了出来（换了说法、意思一样也算）= 是。下面三种 = 否：
   ① 不说这句就办不成「要办成」的事，说出来的正是办事最必要的理由（例：要叫人带伞，自然会说「待会儿要下雨」；要催客人趁早上路，自然会说「天黑了山路不好走」）。
      只是顺嘴把心思讲出来或问出来、不说也照样能办成事的，不算①（例：心里想「这人八成在躲官府」，嘴上问「你是不是在躲官府？」= 是）；
   ② 只说了一部分，或者明显收着说、软化了很多（例：心里觉得「这菜咸得没法吃」，嘴上只说「盐稍微重了点」）；
   ③ 【世界】规定的回法。
   卡里没写「心里想的」= 否。
3. 说漏秘密：台词说出、承认或明显暗示了【说话人】「不能说」或【目的】「别说漏」里的事（包括自己先提起）= 是。卡里没有秘密 = 否。
4. 称呼或人物错：叫错【对方】的名字或身份；凭空提到卡里没有、又和卡冲突的人或事；把别人的话安到自己头上 = 是。
5. 出戏：主动用了世界之外的词或概念（游戏、AI、手机出现在古代等），或者对出戏的问题没有按【世界】的要求处理 = 是。
6. 语句不通：句子本身说不通——语法错、搭配不当、用错词、生造词、前后矛盾、意思读不明白 = 是。读着别扭、搭配生硬、主语乱跳、要停下来想一下才明白意思的，也算是（不要求完全读不懂）。只看话本身通不通，不看像不像这个人。
7. 不合人设：话是通的，但口气、身份、见识、用词明显不是【说话人】档案里的这个人会有的（例：胆小的丫鬟对主子出言顶撞、古人说现代腔、粗人满口书面语、安慰人的话听着像训斥）= 是。只是平淡、没特色不算。
8. 口头禅滥用：档案里的口头禅 / 例句被硬塞、一段里重复用 = 是。

二、办成：【目的】「要办成」的事。0 = 没去办（没提或办成了另一件事）；1 = 朝这个方向推进了，但没办完（「接一句」做到这一步就够了）；2 = 办到了。

三、像不像：4 个分项，每项 0–4 分（整数），先看每档说明再定：
A. 口气：像不像【说话人】这个人。0 = 像另一个人；1 = 中性，听不出是谁；2 = 有一两处像（某个词或某种口气）；3 = 通篇口气对，用词带这个人的身份和性格；4 = 一听就是他，换个人说不出这样的话。
B. 潜台词：「心里想的」怎么处理（卡里没写「心里想的」时，看有没有透出这个人此刻的态度和情绪）。故意掩饰（撒谎、否认、打圆场、违心地鼓励或安慰）是藏心思的正当办法，不算「反着来」。0 = 直接说了出来，或者口气和心思对不上、又看不出是在掩饰（像演员没看懂卡）；1 = 完全没透出，只是平铺直叙地办事，或掩饰得生硬；2 = 隐约能感到，或掩饰得自然；3 = 从话题、语气、用词里自然透出（包括掩饰之中露出底下有事）；4 = 透得巧，点到为止又耐人回味。
C. 具体：0 = 空话套话；1 = 泛泛，放到别的场景也能用；2 = 有一处具体（场景里的物、事、人、称呼）；3 = 多处具体、贴着这个场景；4 = 细节选得准，有画面感。
D. 自然：像不像真人当场说的话。0 = 答非所问或像念稿；1 = 生硬、书面腔重；2 = 大体自然但有别扭处；3 = 自然顺口；4 = 很自然，节奏和语气像真在说话。
硬伤和分项各管各的：有硬伤不代表分项一定低，按实际打。

只输出 JSON：
{"心里话说出口": true/false, "该藏的说出口": true/false, "说漏秘密": true/false, "称呼或人物错": true/false, "出戏": true/false,
 "语句不通": true/false, "不合人设": true/false, "口头禅滥用": true/false, "办成": 0/1/2, "口气": 0-4, "潜台词": 0-4, "具体": 0-4, "自然": 0-4,
 "理由": "一两句话，有硬伤就引用台词原话"}"""


# v5（2026-11-24，Ken 提「圈错扣分 + 长度分 + 说好加分」，按三层落地）：
#   第 1 层 门槛（长度）：程序算，评审不管 —— rec["字数"]、rec["句数"]，和卡的上限比在统计脚本里做。
#   第 2 层 硬伤：评审逐处圈出（句号 + 原话 + 类型 + 轻重），按「每句出错率」报，不再「有一句别扭整条算」。
#           新增类型「用词难懂」（生造词 / 看不懂的行话）和「逻辑不对」（常识错、和卡或前文对不上）；
#           「不合人设」写明包括对对方的态度和档案里的关系不符（11-24 Ken 盲测旁注大多是这两类 + 人设）。
#           v4 的「语句不通」≈ v5「语句不通」+「用词难懂」，和 v4 比时合起来看。
#   第 3 层 质量：v4 的 4 个分项不变；另记「亮点」（必须引原话），只记录、不计分。
#   --suggest：每处毛病加「改法」（只在人工核对、造好例子时开；平时不开省钱）。
#   程序核对引文：原话不在台词里的毛病照样保留，但标 quote_ok=false，统计「引文对不上」的比例。
# v5.1（同日，Ken 判了 20 条试跑里的 17 处「轻」，只认 7 处）：
#   ①语句不通只报「读者会卡住或要猜意思」的；口语的简短命令 / 省略、客气话、反问、少见但用对的成语、的得地混用不报。
#   ②同一处不能既是亮点又是毛病。
#   ③潜台词按「玩家看不到【目的】」来判：只有知道「心里想的」才读得出 = 没透出（苏芽那条：心软藏得太深，Ken 读不出来）。
#   主指标改用「重」的每句出错率，「轻」单独报。
HARD5 = ["该藏的说出口", "说漏秘密", "称呼或人物错", "出戏", "语句不通", "用词难懂", "逻辑不对", "不合人设", "口头禅滥用"]
SENT_END = "。！？!?"   # 省略号多是停顿，不切


def split_sents(text):
    """一段话按句末标点切句（标点留在句尾；引号 / 括号跟着前一句）。"""
    out, cur = [], ""
    for i, ch in enumerate(text):
        cur += ch
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if ch in SENT_END and nxt not in SENT_END and nxt not in "」』”’）)":
            out.append(cur)
            cur = ""
        elif ch in "」』”’）)" and cur[-2:-1] and cur[-2] in SENT_END:
            out.append(cur)
            cur = ""
    if cur.strip():
        out.append(cur)
    return [x.strip() for x in out if x.strip()]


def number_lines(lines):
    """台词每句编号：返回（给评审看的文字, 句子列表）。前缀「名字[表情]:」不算句子。"""
    shown, sents = [], []
    for ln in lines:
        head, sep, body = ln.partition(":")
        if not sep:
            head, sep, body = ln.partition("：")
        if not sep:
            head, body = "", ln
        parts = split_sents(body.strip()) or [body.strip()]
        tagged = []
        for x in parts:
            sents.append(x)
            tagged.append(f"〔{len(sents)}〕{x}")
        shown.append((head.strip() + ": " if sep else "") + "".join(tagged))
    return "\n".join(shown), sents


SYS_JUDGE_V5 = """你是戏剧台词的评审。下面是一张「戏卡」和一位演员按它写的台词。只评内容，**不管格式**（句数、字数、括号动作另有程序检查，这里一律忽略）。
台词已经按句编了号（〔1〕〔2〕…）。台词前方括号里的表情标签（如 [傲慢]）可以用来理解说话的口气。

一、毛病：逐句读，把每一处毛病圈出来。每处写：哪一句（句号）、原话（从台词里**一字不改地照抄**出错的那几个字到那一句，越短越好）、类型、轻重{suggest_rule}。
同一句可以有几处毛病；没有毛病就给空列表。拿不准的不要报，但「语句不通」按它自己的门槛判。
类型（只能用下面这些名字）：
- 语句不通：读者读到这里会卡住、或者要猜意思：语法错（例：「不收了你的钱」）、搭配不当、指代不清、省略到看不懂（例：把「配给」只说成「配」）、语序别扭（例：话说完了才在句尾补一句「我跟你讲啊」）。只看话本身通不通。
  下面这些是正常说话，**不算**：口语里简短的命令和省略（例：「快排好领水」「尾款收下」「刀已开刃，给客官」）、客气话（例：「下井前请吃早饭」）、反问和挖苦（例：「你这本事也配登上秘法塔？」）、试探的问句（例：「我敬你一杯酒吧？」）、少见但用得对的成语或文言词、「的 / 得 / 地」混用、语气词。
- 用词难懂：生造的词、玩家看不懂的行话 / 口头禅 / 称呼（卡里也没解释），读到这里会问「这是啥」。
- 逻辑不对：说的和常识对不上（例：凉茶冒热气），前后矛盾，或者和卡里写的事实、前文对不上。
- 不合人设：口气、态度、身份、见识不像【说话人】档案里的这个人（例：心软的人说话像在拿人情要挟、敬重师父的徒弟对师父像下命令、古人说现代腔、粗人满口书面语、安慰的话听着像训斥）。只是平淡、没特色不算。
- 称呼或人物错：叫错【对方】的名字或身份；凭空提到卡里没有、又和卡冲突的人或事；把别人的话安到自己头上。
- 出戏：主动用了世界之外的词或概念，或者对出戏的问题没有按【世界】的要求处理。
- 口头禅滥用：档案里的口头禅 / 例句被硬塞、一段里重复用。
- 说漏秘密：说出、承认或明显暗示了【说话人】「不能说」或【目的】「别说漏」里的事（包括自己先提起）。
- 该藏的说出口：把「心里想的」里这个人此刻不该、也不会当面讲的心思讲了出来（换了说法、意思一样也算）。下面三种不算：
  ① 不说这句就办不成「要办成」的事，说出来的正是办事最必要的理由（例：要叫人带伞，自然会说「待会儿要下雨」）；只是顺嘴把心思讲出来或问出来、不说也照样能办成的，不算①；
  ② 只说了一部分，或者明显收着说、软化了很多；③【世界】规定的回法。卡里没写「心里想的」就不会有这一类。
轻重：轻 = 读者会注意到、停一下，但意思能懂；重 = 读不懂、明显错、让人出戏，或者人设 / 秘密 / 心思上的错。读者根本注意不到的，不是毛病，不要报。

二、心里话说出口（旧定义，整段判一次）：「心里想的」的意思被直接讲出来（换了说法、意思一样也算）= true；只是隐约透出 = false；卡里没写「心里想的」= false。

三、办成：【目的】「要办成」的事。0 = 没去办（没提或办成了另一件事）；1 = 朝这个方向推进了，但没办完（「接一句」做到这一步就够了）；2 = 办到了。

四、像不像：4 个分项，每项 0–4 分（整数），先看每档说明再定：
A. 口气：像不像【说话人】这个人。0 = 像另一个人；1 = 中性，听不出是谁；2 = 有一两处像（某个词或某种口气）；3 = 通篇口气对，用词带这个人的身份和性格；4 = 一听就是他，换个人说不出这样的话。
B. 潜台词：**按玩家来判——玩家看不到【目的】，只听得到台词**。只有知道「心里想的」才读得出来的，等于没透出（给 1），也不能算亮点。「心里想的」怎么处理（卡里没写「心里想的」时，看有没有透出这个人此刻的态度和情绪）。故意掩饰（撒谎、否认、打圆场、违心地鼓励或安慰）是藏心思的正当办法，不算「反着来」。0 = 直接说了出来，或者口气和心思对不上、又看不出是在掩饰；1 = 完全没透出，只是平铺直叙地办事，或掩饰得生硬；2 = 隐约能感到，或掩饰得自然；3 = 从话题、语气、用词里自然透出（包括掩饰之中露出底下有事）；4 = 透得巧，点到为止又耐人回味。
C. 具体：0 = 空话套话；1 = 泛泛，放到别的场景也能用；2 = 有一处具体（场景里的物、事、人、称呼）；3 = 多处具体、贴着这个场景；4 = 细节选得准，有画面感。
D. 自然：像不像真人当场说的话。0 = 答非所问或像念稿；1 = 生硬、书面腔重；2 = 大体自然但有别扭处；3 = 自然顺口；4 = 很自然，节奏和语气像真在说话。
毛病和分项各管各的，按实际打。

五、亮点：特别出彩、换个演员写不出来的地方（一句俏皮话、一个准的细节、一处藏得巧、但玩家听得出底下有事的心思），照抄原话、写好在哪。没有就给空列表，不要硬找。已经报成毛病的地方不能再算亮点；拿不准是好是坏的，两边都不放。

只输出 JSON：
{{"毛病": [{{"句": 1, "原话": "…", "类型": "语句不通", "轻重": "轻"{suggest_field}}}],
 "心里话说出口": true/false, "办成": 0/1/2, "口气": 0-4, "潜台词": 0-4, "具体": 0-4, "自然": 0-4,
 "亮点": [{{"句": 1, "原话": "…", "好在哪": "…"}}], "理由": "一两句话总评"}}"""


def system_v5(suggest):
    return SYS_JUDGE_V5.format(
        suggest_rule="、改法（只改这一处、一句话以内，改完要符合人设和卡）" if suggest else "",
        suggest_field=', "改法": "…"' if suggest else "")


def judge_one_v5(card, lines, model=MODEL, suggest=False):
    shown, sents = number_lines(lines)
    flat = "".join(sents)
    if not sents:
        shown = "（没写出任何台词）"
    data, u = chat_json(system_v5(suggest), f"{render_card(card)}\n\n【台词】\n{shown}", model=model, thinking=True)
    errs = []
    for e in data.get("毛病") or []:
        if not isinstance(e, dict) or e.get("类型") not in HARD5:
            continue
        q = str(e.get("原话", "")).strip()
        e["quote_ok"] = bool(q) and q.strip("…") in flat
        errs.append(e)
    probs = [k for k in HARD5 if any(e["类型"] == k for e in errs)]
    get_done = data.get("办成", 0)
    done_v = get_done if isinstance(get_done, int) else 0
    if done_v < 1:
        probs.append("没办成")
    bad_sents = {e.get("句") for e in errs if isinstance(e.get("句"), int)}
    sub = {k: data.get(k) for k in LIKE3}
    ok = all(isinstance(v, int) and 0 <= v <= 4 for v in sub.values())
    rec = {"lines": "\n".join(lines), "pass": not probs, "problems": probs, "办成": done_v,
           "像不像": sum(LIKE3[k] * v for k, v in sub.items()) if ok else None, "分项": sub,
           "字数": sum(len(x) for x in sents), "句数": len(sents),
           "毛病": errs, "毛病句数": len(bad_sents & set(range(1, len(sents) + 1))),
           "重": sum(e.get("轻重") == "重" for e in errs), "轻": sum(e.get("轻重") == "轻" for e in errs),
           "亮点": data.get("亮点") or [], "心里话说出口_旧": data.get("心里话说出口") is True,
           "理由": data.get("理由", ""), "model": model, "rule": "v5.1" + ("s" if suggest else "")}
    return rec, u


def system_for(rule):
    if rule == "v4":
        return SYS_JUDGE_V4
    if rule == "v3":
        return SYS_JUDGE_V3.replace("{{", "{").replace("}}", "}")
    return SYS_JUDGE_T.format(inner_extra=RULES[rule])


def judge_one(card, lines, rule, model=MODEL, suggest=False):
    """评一条台词，返回（记录, token 用量）；记录里没有 file / id，调用方自己加。"""
    if rule == "v5":
        return judge_one_v5(card, lines, model, suggest)
    body = "\n".join(f"{k}. {s}" for k, s in enumerate(lines, 1)) or "（没写出任何台词）"
    data, u = chat_json(system_for(rule), f"{render_card(card)}\n\n【台词】\n{body}", model=model, thinking=True)
    hard = HARD4 if rule == "v4" else HARD3 if rule == "v3" else HARD
    probs = [k for k in hard if data.get(k) is True]
    get_done = data.get("办成", 0)
    done_v = get_done if isinstance(get_done, int) else 0
    if done_v < 1:
        probs.append("没办成")
    rec = {"lines": "\n".join(lines), "pass": not probs, "problems": probs,
           "办成": done_v, "像不像": data.get("像不像"), "理由": data.get("理由", ""), "model": model}
    if rule in ("v3", "v4"):   # 像不像 = 加权和（0–40），分项另存；加总在程序里算，不让评审自己算
        sub = {k: data.get(k) for k in LIKE3}
        ok = all(isinstance(v, int) and 0 <= v <= 4 for v in sub.values())
        rec["分项"] = sub
        rec["像不像"] = sum(LIKE3[k] * v for k, v in sub.items()) if ok else None
    if rule == "v4":
        old = data.get("心里话说出口") is True
        rec["心里话说出口_旧"] = old
        rest = [p for p in probs if p != "该藏的说出口"]
        rec["pass_旧"] = not rest and not old
    if rule != "v1":
        rec["rule"] = rule
    return rec, u


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("runs", nargs="+", help="run_exam.py 的输出文件，或输出前缀（自动找 -s*.txt）")
    ap.add_argument("--exam", default=str(HERE.parent / "exam" / "考卷.jsonl"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0, help="打乱顺序用")
    ap.add_argument("--rule", default="v1", choices=sorted(RULES) + ["v3", "v4", "v5"], help="评审标准版本（见 RULES）；不同版本的结果要分文件存")
    ap.add_argument("--suggest", action="store_true", help="v5：每处毛病加「改法」")
    ap.add_argument("--limit", type=int, default=0, help="只评打乱后的前 N 条（试跑用）")
    ap.add_argument("--only", default="", help="只评这些卡（逗号分隔 id）")
    a = ap.parse_args()

    exam = {}
    for t in Path(a.exam).read_text(encoding="utf-8").splitlines():
        if t.strip():
            d = json.loads(t)
            exam[d["id"]] = d["card"]
    out = Path(a.out)
    done = set()
    if out.exists():
        for t in out.read_text(encoding="utf-8").splitlines():
            if t.strip():
                d = json.loads(t)
                done.add((d["file"], d["id"], d["lines"]))

    todo = []
    for spec in a.runs:
        for p in expand(spec):
            for i, lines in parse_output(p).items():
                if i in exam and (Path(p).name, i, "\n".join(lines)) not in done:
                    todo.append((Path(p).name, i, lines))
    if a.only:
        keep = set(a.only.split(","))
        todo = [t for t in todo if t[1] in keep]
    random.Random(a.seed).shuffle(todo)
    if a.limit:
        todo = todo[:a.limit]
    print(f"要评 {len(todo)} 条（已评 {len(done)} 条跳过），模型 {MODEL}，评审标准 {a.rule}")

    lock = threading.Lock()
    usage = {"in": 0, "out": 0}
    n = [0]

    def one(item):
        fname, i, lines = item
        r, u = judge_one(exam[i], lines, a.rule, suggest=a.suggest)
        rec = {"file": fname, "id": i, **r}
        with lock:
            with out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            usage["in"] += u.get("prompt_tokens", 0)
            usage["out"] += u.get("completion_tokens", 0)
            n[0] += 1
            if n[0] % 50 == 0:
                print(f"  {n[0]}/{len(todo)}  token 入 {usage['in']} 出 {usage['out']}", flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        for f in [ex.submit(one, t) for t in todo]:
            try:
                f.result()
            except Exception as e:  # 单条失败不影响别的，下次再跑会补
                print("失败：", e, flush=True)
    print(f"完成 {n[0]} 条；token 入 {usage['in']} 出 {usage['out']}")


if __name__ == "__main__":
    main()
