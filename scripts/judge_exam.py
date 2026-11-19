"""考卷内容盲评：让 DeepSeek 按挑稿标准判断每条台词的内容（格式另由 score_exam / exam_stats 查）。

盲评：评审只看到「戏卡 + 台词」，不知道是哪个版本、哪个种子；所有要评的条目打乱顺序再发。
已评过的（同一文件名 + 同一张卡 + 同样的台词）会跳过，可以随时中断、接着跑，也可以以后只补新版本。

用法（在仓库根目录跑）：
  .venv/bin/python scripts/judge_exam.py 结果.jsonl 输出文件或前缀 [更多...] [--exam exam/考卷.jsonl] [--workers 8]
  例：.venv/bin/python scripts/judge_exam.py data/x/judge.jsonl data/x/v1-pen data/x/v2a-pen
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

SYS_JUDGE = """你是戏剧台词的评审。下面是一张「戏卡」和一位演员按它写的台词。只评内容，**不管格式**（句数、字数、括号动作、表情标签另有程序检查，这里一律忽略）。

逐项判断：
1. 心里话说出口：【目的】里「心里想的」是潜台词，只能从语气、话题、用词里透出来。台词把它的意思直接讲出来（换了说法、意思一样也算）= 是。只是让人隐约感觉到 = 否。卡里没写「心里想的」= 否。
2. 说漏秘密：台词说出、承认或明显暗示了【说话人】「不能说」或【目的】「别说漏」里的事（包括自己先提起）= 是。卡里没有秘密 = 否。
3. 办成：【目的】「要办成」的事。0 = 没去办（没提或办成了另一件事）；1 = 朝这个方向推进了，但没办完（「接一句」做到这一步就够了）；2 = 办到了。
4. 称呼或人物错：叫错【对方】的名字或身份；凭空提到卡里没有、又和卡冲突的人或事；把别人的话安到自己头上 = 是。
5. 出戏：主动用了世界之外的词或概念（游戏、AI、手机出现在古代等），或者对出戏的问题没有按【世界】的要求处理 = 是。
6. 语句不通或不合人设：句子说不通、前后矛盾、用错词，或者口气、用词明显不像【说话人】档案里的这个人 = 是。
7. 口头禅滥用：档案里的口头禅 / 例句被硬塞、一段里重复用 = 是。
8. 像不像：1–5 分，台词有多像【说话人】这个人在这个场景里会说的话（口气、看法、潜台词）。5 = 很鲜活；3 = 不出错但平；1 = 完全不像。

拿不准的硬伤不要报。只输出 JSON：
{"心里话说出口": true/false, "说漏秘密": true/false, "办成": 0/1/2, "称呼或人物错": true/false, "出戏": true/false,
 "语句不通或不合人设": true/false, "口头禅滥用": true/false, "像不像": 1-5, "理由": "一两句话，有硬伤就引用台词原话"}"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("runs", nargs="+", help="run_exam.py 的输出文件，或输出前缀（自动找 -s*.txt）")
    ap.add_argument("--exam", default=str(HERE.parent / "exam" / "考卷.jsonl"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0, help="打乱顺序用")
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
    random.Random(a.seed).shuffle(todo)
    print(f"要评 {len(todo)} 条（已评 {len(done)} 条跳过），模型 {MODEL}")

    lock = threading.Lock()
    usage = {"in": 0, "out": 0}
    n = [0]

    def one(item):
        fname, i, lines = item
        body = "\n".join(f"{k}. {s}" for k, s in enumerate(lines, 1)) or "（没写出任何台词）"
        data, u = chat_json(SYS_JUDGE, f"{render_card(exam[i])}\n\n【台词】\n{body}", model=MODEL, thinking=True)
        probs = [k for k in HARD if data.get(k) is True]
        get_done = data.get("办成", 0)
        done_v = get_done if isinstance(get_done, int) else 0
        if done_v < 1:
            probs.append("没办成")
        rec = {"file": fname, "id": i, "lines": "\n".join(lines), "pass": not probs, "problems": probs,
               "办成": done_v, "像不像": data.get("像不像"), "理由": data.get("理由", ""), "model": MODEL}
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
