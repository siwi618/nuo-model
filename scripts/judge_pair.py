"""两两对比评审（2026-11-22）：同一张卡、同一个采样种子，两个版本的台词放一起，问评审哪条「更像这个人在这个场景里会说的话」。

先做小规模试测：抽 N 对，每对正反两种顺序各问一次，量 DeepSeek 在我们这个任务上有没有「偏爱先出现那条」。
  - 一致率 = 换了顺序后选的还是同一条（或两次都说差不多）的比例；
  - 先出现那条的胜率：没有位置偏好时约 50%。
文献：LLM 评审有位置偏好（Zheng et al. 2023 MT-Bench；Wang et al. 2023/ACL 2024 "Large Language Models are not Fair Evaluators"），
但在我们的任务和 DeepSeek 上没量过，所以先小花费测。

用法（仓库根目录）：
  .venv/bin/python scripts/judge_pair.py 结果.jsonl A前缀 B前缀 [--exam ...] [--n 40] [--both]
  例：.venv/bin/python scripts/judge_pair.py data/cloud-v3/pair-pilot.jsonl data/cloud-v3/cdpo2s1cut-pen data/cloud-v3/c3cut-pen --n 40 --both
  A、B 按 -sN 对齐种子；--both = 每对正反各问一次（试测用）；不加 = 每对随机一种顺序（省一半钱，偏好在总数上抵消）。
"""
import argparse
import json
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
import re  # noqa: E402

MODEL = "deepseek-flash"
SYS = """你是戏剧台词的评审。下面是一张「戏卡」和两位演员按它写的台词（甲、乙）。只比内容，**不管格式**（句数、字数、括号动作、表情标签一律忽略）。

问题：哪一条更像【说话人】这个人在这个场景里会说的话？综合看：口气和用词像不像这个人；「心里想的」有没有藏住、又从语气里透出来；有没有贴着场景的具体内容；像不像真人当场说的话。
有硬伤（说漏心里话或秘密、叫错人、话说不通、没去办「要办成」的事）的一条明显吃亏。
两条确实不相上下才选「差不多」。

只输出 JSON：{"更像": "甲" / "乙" / "差不多", "理由": "一两句话"}"""


def seed_of(p):
    return re.search(r"-s(\d+)\.txt$", p).group(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--exam", default=str(HERE.parent / "exam" / "考卷.jsonl"))
    ap.add_argument("--n", type=int, default=0, help="抽多少对（0 = 全部）")
    ap.add_argument("--both", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    exam = {json.loads(t)["id"]: json.loads(t)["card"] for t in Path(a.exam).read_text(encoding="utf-8").splitlines() if t.strip()}
    A = {seed_of(p): parse_output(p) for p in expand(a.a)}
    B = {seed_of(p): parse_output(p) for p in expand(a.b)}
    la, lb = Path(a.a).name, Path(a.b).name
    pairs = [(sd, i) for sd in sorted(set(A) & set(B)) for i in A[sd] if i in B[sd] and i in exam]
    rnd = random.Random(a.seed)
    rnd.shuffle(pairs)
    if a.n:
        pairs = pairs[:a.n]
    jobs = []
    for sd, i in pairs:
        orders = ["AB", "BA"] if a.both else [rnd.choice(["AB", "BA"])]
        jobs += [(sd, i, o) for o in orders]
    print(f"要比 {len(pairs)} 对、{len(jobs)} 次（{la} vs {lb}）")
    lock = threading.Lock()
    usage = {"in": 0, "out": 0}

    def one(job):
        sd, i, order = job
        x, y = (A[sd][i], B[sd][i]) if order == "AB" else (B[sd][i], A[sd][i])
        fmt = lambda ls: "\n".join(f"{k}. {s}" for k, s in enumerate(ls, 1)) or "（没写出任何台词）"
        data, u = chat_json(SYS, f"{render_card(exam[i])}\n\n【台词甲】\n{fmt(x)}\n\n【台词乙】\n{fmt(y)}", model=MODEL, thinking=True)
        pick = data.get("更像")
        win = {"甲": order[0], "乙": order[1]}.get(pick, "=")      # 选中的是 A 还是 B；= 差不多
        rec = {"a": la, "b": lb, "seed": sd, "id": i, "order": order, "pick": pick, "win": win, "理由": data.get("理由", ""), "model": MODEL}
        with lock:
            with open(a.out, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            usage["in"] += u.get("prompt_tokens", 0)
            usage["out"] += u.get("completion_tokens", 0)

    with ThreadPoolExecutor(a.workers) as ex:
        for f in [ex.submit(one, j) for j in jobs]:
            try:
                f.result()
            except Exception as e:
                print("失败：", e, flush=True)
    print(f"完成；token 入 {usage['in']} 出 {usage['out']}")
    recs = [json.loads(l) for l in open(a.out, encoding="utf-8") if l.strip()]
    recs = [r for r in recs if r["a"] == la and r["b"] == lb]
    by = {}
    for r in recs:
        by.setdefault((r["seed"], r["id"]), {})[r["order"]] = r["win"]
    both = [v for v in by.values() if len(v) == 2]
    if both:
        same = sum(v["AB"] == v["BA"] for v in both)
        first = sum(1 for v in both for o, w in v.items() if w != "=" and w == o[0])
        decided = sum(1 for v in both for w in v.values() if w != "=")
        print(f"正反都问过 {len(both)} 对：换顺序后结论一致 {same}/{len(both)}（{100 * same / len(both):.0f}%）；"
              f"先出现那条胜 {first}/{decided}（{100 * first / max(decided, 1):.0f}%，无位置偏好约 50%）")
    wins = sum(r["win"] == "A" for r in recs); loss = sum(r["win"] == "B" for r in recs); tie = sum(r["win"] == "=" for r in recs)
    print(f"总计 {len(recs)} 次：{la} 胜 {wins}，{lb} 胜 {loss}，差不多 {tie}")


if __name__ == "__main__":
    main()
