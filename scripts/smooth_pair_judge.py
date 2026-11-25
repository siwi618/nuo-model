"""两两「哪条更顺」评审校准（2026-11-24 对尺子）：把 Ken 做过的「哪条更顺」盲测原样交给评审，看和 Ken 一致不一致。

背景：评审单条判「语句不通」只判一边的对，Ken 同向 8 / 差不多 9 / 反向 8（80 对，等于抛硬币）。
这里改问 Ken 实际回答的问题：同卡两条放一起，哪条读着更顺。说明照抄 Ken 页面上的话。
每对正反两种顺序各问一次（量位置偏好，也算评审自己前后一致不一致）。

用法（仓库根目录）：
  .venv/bin/python scripts/smooth_pair_judge.py data/human-check/对尺子/pair-smooth.jsonl          # 跑（已跑过的跳过）
  .venv/bin/python scripts/smooth_pair_judge.py data/human-check/对尺子/pair-smooth.jsonl --score  # 只算分
"""
import argparse
import json
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from deepseek import chat_json  # noqa: E402
from render_card import render_card  # noqa: E402
from smooth_retest_build import TESTS, load_lines, load_test  # noqa: E402

ROOT = HERE.parent
MODEL = "deepseek-flash"
SYS = """你是戏剧台词的读者。下面是一张「戏卡」和两位演员按它写的台词（甲、乙）。

问题：哪条读着更顺？只判读着顺不顺：用词、搭配、语序自然，不用停下来想；口气像不像、有没有说漏都不用管。括号里的动作和方括号里的表情标签不用评。
分不出来就选「差不多」。

只输出 JSON：{"更顺": "甲" / "乙" / "差不多", "理由": "一两句话"}"""


def pairs(tests=None):
    cards = {}
    for fn in ["考卷.jsonl", "专项-心里话.jsonl"]:
        for t in (ROOT / "exam" / fn).read_text(encoding="utf-8").splitlines():
            if t.strip():
                r = json.loads(t)
                cards[(fn, r["id"])] = r["card"]
    cell = load_lines()
    out = []
    for t in tests or TESTS:
        key, prefs, notes = load_test(t)
        for k, p in zip(key, prefs):
            fn = "考卷.jsonl" if k["exam"] == "pub" else "专项-心里话.jsonl"
            out.append({"test": t, "q": k["id"], "card": cards[(fn, k["card"])],
                        "A": cell[(k["exam"], k["A"], k["card"], k["seed"])], "B": cell[(k["exam"], k["B"], k["card"], k["seed"])],
                        "ken": p, "noted": k["id"] in notes})
    return out


def run(path, items, workers):
    done = set()
    if path.exists():
        done = {(r["test"], r["q"], r["order"]) for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())}
    jobs = [(it, o) for it in items for o in ("AB", "BA") if (it["test"], it["q"], o) not in done]
    print(f"要问 {len(jobs)} 次")
    lock = threading.Lock()
    usage = Counter()

    def one(job):
        it, order = job
        x, y = (it["A"], it["B"]) if order == "AB" else (it["B"], it["A"])
        data, u = chat_json(SYS, f"{render_card(it['card'])}\n\n【台词甲】\n{x}\n\n【台词乙】\n{y}", model=MODEL, thinking=True)
        pick = data.get("更顺")
        win = {"甲": order[0], "乙": order[1]}.get(pick, "=")
        rec = {"test": it["test"], "q": it["q"], "order": order, "pick": pick, "win": win, "理由": data.get("理由", ""), "model": MODEL}
        with lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            usage["in"] += u.get("prompt_tokens", 0)
            usage["out"] += u.get("completion_tokens", 0)

    with ThreadPoolExecutor(workers) as ex:
        for f in [ex.submit(one, j) for j in jobs]:
            try:
                f.result()
            except Exception as e:
                print("失败：", e, flush=True)
    print(f"token 入 {usage['in']} 出 {usage['out']}")


def score(path, items):
    J = {}
    for r in map(json.loads, path.read_text(encoding="utf-8").splitlines()):
        J.setdefault((r["test"], r["q"]), {})[r["order"]] = r["win"]
    both = [(it, J[(it["test"], it["q"])]) for it in items if len(J.get((it["test"], it["q"]), {})) == 2]
    same = sum(v["AB"] == v["BA"] for _, v in both)
    first = sum(1 for _, v in both for o, w in v.items() if w != "=" and w == o[0])
    decided = sum(1 for _, v in both for w in v.values() if w != "=")
    print(f"{len(both)} 对正反都问过：评审换顺序后结论一致 {same}/{len(both)}（{same / len(both):.0%}）；先出现那条胜 {first}/{decided}")

    def verdict(v):        # 两次一致才算选了一边；不一致或都说差不多 = 差不多
        return v["AB"] if v["AB"] == v["BA"] else "="

    for name, sub in [("全部", both), ("没写旁注的", [x for x in both if not x[0]["noted"]])]:
        tab = Counter((it["ken"], verdict(v)) for it, v in sub)
        agree = sum(n for (k, j), n in tab.items() if k == j)
        one_side = [(it["ken"], verdict(v)) for it, v in sub if it["ken"] != "=" and verdict(v) != "="]
        hit = sum(k == j for k, j in one_side)
        print(f"\n== {name}（{len(sub)} 对）：三档完全一致 {agree}/{len(sub)}（{agree / len(sub):.0%}）；"
              f"两边都选了一边的 {len(one_side)} 对里同向 {hit}（{hit / max(len(one_side), 1):.0%}）")
        print("   行 Ken / 列评审      A    B    =")
        for k in "AB=":
            print(f"   {k}               " + "  ".join(f"{tab[(k, j)]:3d}" for j in "AB="))
    print("\n分次：")
    for t in TESTS:
        sub = [(it, v) for it, v in both if it["test"] == t]
        os_ = [(it["ken"], verdict(v)) for it, v in sub if it["ken"] != "=" and verdict(v) != "="]
        print(f"  {t}: 两边都选了一边 {len(os_)} 对，同向 {sum(k == j for k, j in os_)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    path = Path(a.out)
    items = pairs()
    if not a.score:
        run(path, items, a.workers)
    score(path, items)


if __name__ == "__main__":
    main()
