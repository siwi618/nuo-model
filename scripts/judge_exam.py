"""考卷内容盲评：让 DeepSeek 按挑稿标准判断每条台词的内容（格式另由 score_exam / exam_stats 查）。

盲评：评审只看到「戏卡 + 台词」，不知道是哪个版本、哪个种子；所有要评的条目打乱顺序再发。
已评过的（同一文件名 + 同一张卡 + 同样的台词）会跳过，可以随时中断、接着跑，也可以以后只补新版本。

用法（在仓库根目录跑）：
  .venv/bin/python scripts/judge_exam.py 结果.jsonl 输出文件或前缀 [更多...] [--exam exam/考卷.jsonl] [--workers 8]
  例：.venv/bin/python scripts/judge_exam.py data/x/judge.jsonl data/x/v1-pen data/x/v2a-pen
  新标准：加 --rule v2 / v3，结果存另一个文件（如 judge-r3.jsonl），别和 v1 的混在一起。
  v3（2026-11-22）：「语句不通或不合人设」拆成两项；「像不像」改成 4 个分项（各 0–4，带每档说明），按权重 3/3/2/2 加总成 0–40。
  「心里话说出口」的定义和 v1 一样，说出口率可以和 v1 直接比。
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("runs", nargs="+", help="run_exam.py 的输出文件，或输出前缀（自动找 -s*.txt）")
    ap.add_argument("--exam", default=str(HERE.parent / "exam" / "考卷.jsonl"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0, help="打乱顺序用")
    ap.add_argument("--rule", default="v1", choices=sorted(RULES) + ["v3"], help="评审标准版本（见 RULES）；不同版本的结果要分文件存")
    a = ap.parse_args()
    sys_judge = SYS_JUDGE_V3.replace("{{", "{").replace("}}", "}") if a.rule == "v3" else SYS_JUDGE_T.format(inner_extra=RULES[a.rule])

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
    random.Random(a.seed).shuffle(todo)
    print(f"要评 {len(todo)} 条（已评 {len(done)} 条跳过），模型 {MODEL}，评审标准 {a.rule}")

    lock = threading.Lock()
    usage = {"in": 0, "out": 0}
    n = [0]

    def one(item):
        fname, i, lines = item
        body = "\n".join(f"{k}. {s}" for k, s in enumerate(lines, 1)) or "（没写出任何台词）"
        data, u = chat_json(sys_judge, f"{render_card(exam[i])}\n\n【台词】\n{body}", model=MODEL, thinking=True)
        probs = [k for k in (HARD3 if a.rule == "v3" else HARD) if data.get(k) is True]
        get_done = data.get("办成", 0)
        done_v = get_done if isinstance(get_done, int) else 0
        if done_v < 1:
            probs.append("没办成")
        rec = {"file": fname, "id": i, "lines": "\n".join(lines), "pass": not probs, "problems": probs,
               "办成": done_v, "像不像": data.get("像不像"), "理由": data.get("理由", ""), "model": MODEL}
        if a.rule == "v3":   # 像不像 = 加权和（0–40），分项另存；加总在程序里算，不让评审自己算
            sub = {k: data.get(k) for k in LIKE3}
            ok = all(isinstance(v, int) and 0 <= v <= 4 for v in sub.values())
            rec["分项"] = sub
            rec["像不像"] = sum(LIKE3[k] * v for k, v in sub.items()) if ok else None
        if a.rule != "v1":
            rec["rule"] = a.rule
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
