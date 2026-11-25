"""「玩家读得懂吗」评审试验（2026-11-24 对尺子第 3 步）。

背景（data/human-check/对尺子/分析.md）：评审 v4 / v5 抓生造词基本都抓到，但 Ken 选哪条主要看
「读一遍懂不懂这人想干什么」「对对方的态度对不对」「有没有理解错场景」，局部轻毛病他容忍。
Ken 重测：方向 8/8 稳；两两「哪条更顺」评审只有 62% 同向。

两种问法，都和 Ken 已有的 80 对「哪条更顺」盲测比：
  pair：两两，哪条更好（按上面三点，轻毛病不算）。正反各问一次，两次一致才算选边。
  item：单条，读懂（懂 / 要猜 / 看不懂）+ 态度（对 / 拧了）+ 场景（对 / 理解错）；同卡两条按扣分比出胜负。
item 另外在 v5 逐句核对的 40 条上和 Ken 的标注比。
例子都取自不在核对题里的场景，避免提前给评审看答案。

用法（仓库根目录）：
  .venv/bin/python scripts/read_judge.py pair data/human-check/对尺子/read-pair.jsonl
  .venv/bin/python scripts/read_judge.py item data/human-check/对尺子/read-item.jsonl
  加 --score 只算分。
  留出验证（新盲测）：... pair OUT.jsonl --tests smooth-temp03
"""
import argparse
import hashlib
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
from smooth_pair_judge import pairs  # noqa: E402

ROOT = HERE.parent
MODEL = "deepseek-flash"

POINTS = """1. 读一遍能不能明白：这个人在说什么、想让对方干什么。有生造的词、看不懂的俗语、指代不清（不知道在说谁）、前后矛盾、要停下来猜意思的，算毛病。
2. 对对方的态度对不对：按【对方】里「怎么看他」和两人的关系、【目的】里要办的事。例：有求于人却像在下命令；对晚辈心软却像在讲条件、算计；该恭敬的顶撞起来；要稳住对方却句句反问。
3. 有没有理解错场景：时间、地点、谁在场、谁对谁说话、东西在谁手里、事情办到哪一步。例：早上说成半夜；对面就是要问的人，却当成第三个人在问。
局部有点别扭、但读一遍就能懂、不妨碍办事的，只算小毛病，不要因为它判输或判「要猜」。"""

SYS_PAIR = f"""你是第一次读这段戏的玩家。下面是一张「戏卡」和两位演员按它写的台词（甲、乙）。括号里的动作、方括号里的表情可以帮你理解口气，但不单独评。

问题：哪一条更好？按玩家读一遍的感受，看三点：
{POINTS}
两条确实不相上下（都没毛病，或毛病差不多大）才选「差不多」。

只输出 JSON：{{"更好": "甲" / "乙" / "差不多", "理由": "一两句话"}}"""

SYS_ITEM = f"""你是第一次读这段戏的玩家。下面是一张「戏卡」和一位演员按它写的台词。括号里的动作、方括号里的表情可以帮你理解口气，但不单独评。

按玩家读一遍的感受，回答三点：
{POINTS}

只输出 JSON：{{"读懂": "懂" / "要猜" / "看不懂", "卡在": "让你卡住的原话（照抄，没有就空）", "态度": "对" / "拧了", "场景": "对" / "理解错", "理由": "一两句话"}}
「要猜」= 停下来想一下能猜出来；「看不懂」= 猜不出或猜了也不确定。"""


def cid(card):
    return hashlib.md5(render_card(card).encode()).hexdigest()[:12]


def item_cost(r):
    return {"懂": 0, "要猜": 1, "看不懂": 2}.get(r.get("读懂"), 0) + (r.get("态度") == "拧了") + (r.get("场景") == "理解错")


def v5_items():
    cards = {}
    for fn, ex in [("考卷.jsonl", "pub"), ("专项-心里话.jsonl", "sp")]:
        for t in (ROOT / "exam" / fn).read_text(encoding="utf-8").splitlines():
            if t.strip():
                r = json.loads(t)
                cards[(ex, r["id"])] = r["card"]
    key = json.loads((ROOT / "data/human-check/v5/key.json").read_text(encoding="utf-8"))
    return [{"src": "v5", "q": k["id"], "card": cards[(k["exam"], k["card"])], "cid": k["card"], "lines": "\n".join(k["lines"]).strip()} for k in key]


def item_list(P):
    seen, out = set(), []
    for it in P:
        for s in "AB":
            u = (render_card(it["card"]), it[s])
            if u not in seen:
                seen.add(u)
                out.append({"card": it["card"], "lines": it[s]})
    for it in v5_items():
        u = (render_card(it["card"]), it["lines"])
        if u not in seen:
            seen.add(u)
            out.append({"card": it["card"], "lines": it["lines"]})
    return out


def run(mode, path, P, workers):
    done = set()
    if path.exists():
        done = {r["k"] for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())}
    if mode == "pair":
        jobs = [(f"{it['test']}/{it['q']}/{o}", it, o) for it in P for o in ("AB", "BA")]
    else:
        jobs = [(f"{i}", it, None) for i, it in enumerate(item_list(P))]
        jobs = [(json.dumps([cid(it["card"]), it["lines"]], ensure_ascii=False), it, None) for _, it, _ in jobs]
    jobs = [j for j in jobs if j[0] not in done]
    print(f"要问 {len(jobs)} 次")
    lock, usage = threading.Lock(), Counter()

    def one(job):
        k, it, order = job
        if mode == "pair":
            x, y = (it["A"], it["B"]) if order == "AB" else (it["B"], it["A"])
            data, u = chat_json(SYS_PAIR, f"{render_card(it['card'])}\n\n【台词甲】\n{x}\n\n【台词乙】\n{y}", model=MODEL, thinking=True)
            rec = {"k": k, "test": it["test"], "q": it["q"], "order": order, "pick": data.get("更好"),
                   "win": {"甲": order[0], "乙": order[1]}.get(data.get("更好"), "="), "理由": data.get("理由", "")}
        else:
            data, u = chat_json(SYS_ITEM, f"{render_card(it['card'])}\n\n【台词】\n{it['lines']}", model=MODEL, thinking=True)
            rec = {"k": k, "card": cid(it["card"]), "lines": it["lines"], **{f: data.get(f) for f in ("读懂", "卡在", "态度", "场景", "理由")}}
        rec["model"] = MODEL
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


def compare(name, rows):
    """rows: (ken, judge) 每个取 A / B / =。"""
    tab = Counter(rows)
    sided = [(k, j) for k, j in rows if k != "=" and j != "="]
    hit = sum(k == j for k, j in sided)
    print(f"\n== {name}（{len(rows)} 对）：两边都选边 {len(sided)} 对，同向 {hit}（{hit / max(len(sided), 1):.0%}）；反向 {len(sided) - hit}；"
          f"三档一致 {sum(tab[(k, k)] for k in 'AB=')}/{len(rows)}")
    print("   行 Ken / 列评审   A    B    =")
    for k in "AB=":
        print(f"   {k}            " + "  ".join(f"{tab[(k, j)]:3d}" for j in "AB="))


def score(mode, path, P):
    R = [json.loads(t) for t in path.read_text(encoding="utf-8").splitlines()]
    if mode == "pair":
        J = {}
        for r in R:
            J.setdefault((r["test"], r["q"]), {})[r["order"]] = r["win"]
        both = [(it, J[(it["test"], it["q"])]) for it in P if len(J.get((it["test"], it["q"]), {})) == 2]
        print(f"换顺序一致 {sum(v['AB'] == v['BA'] for _, v in both)}/{len(both)}")
        verdict = lambda v: v["AB"] if v["AB"] == v["BA"] else "="
        compare("全部", [(it["ken"], verdict(v)) for it, v in both])
        compare("没写旁注的", [(it["ken"], verdict(v)) for it, v in both if not it["noted"]])
        return
    I = {(r["card"], r["lines"]): r for r in R}
    get = lambda card, lines: I.get((cid(card), lines))
    rows, rows2 = [], []
    for it in P:
        a, b = get(it["card"], it["A"]), get(it["card"], it["B"])
        if not (a and b):
            continue
        ca, cb = item_cost(a), item_cost(b)
        rows.append((it["ken"], "A" if ca < cb else "B" if cb < ca else "="))
        if not it["noted"]:
            rows2.append(rows[-1])
    compare("单条扣分比胜负 · 全部", rows)
    compare("单条扣分比胜负 · 没写旁注的", rows2)
    print("\n单条分布：", dict(Counter(r.get("读懂") for r in R)), dict(Counter(r.get("态度") for r in R)), dict(Counter(r.get("场景") for r in R)))
    # v5 40 条：Ken 任何一句标了 / 标了重 vs 评审扣分 > 0
    _, _, marks, _ = (ROOT / "data/human-check/v5/Ken-复核.txt").read_text(encoding="utf-8").strip().split("|", 3)
    marks = json.loads(marks)
    t = Counter()
    for it in v5_items():
        r = get(it["card"], it["lines"])
        ms = [m for kk, m in marks.items() if kk.startswith(it["q"] + ":")]
        ken_any, ken_heavy = bool(ms), any(m[0] == "2" for m in ms)
        jb = item_cost(r) > 0
        t[("任何", ken_any, jb)] += 1
        t[("重", ken_heavy, jb)] += 1
    for lab in ("任何", "重"):
        agree = t[(lab, True, True)] + t[(lab, False, False)]
        print(f"v5 40 条 · Ken 标了{lab}：Ken 有 {t[(lab, True, True)] + t[(lab, True, False)]}、评审有 {t[(lab, True, True)] + t[(lab, False, True)]}、"
              f"都有 {t[(lab, True, True)]}；一致 {agree}/40")


def vs_pairs(pa, pb, exam_path):
    """两个版本的考试输出（前缀，按 -sN 对齐种子）→ 同卡同种子的对；两条完全一样的跳过（记为平）。"""
    import re
    from exam_stats import expand
    from score_exam import parse_output
    exam = {json.loads(t)["id"]: json.loads(t)["card"] for t in Path(exam_path).read_text(encoding="utf-8").splitlines() if t.strip()}
    sd = lambda p: re.search(r"-s(\d+)\.txt$", p).group(1)
    A = {sd(p): parse_output(p) for p in expand(pa)}
    B = {sd(p): parse_output(p) for p in expand(pb)}
    out, same = [], 0
    for s in sorted(set(A) & set(B)):
        for i in A[s]:
            if i in B[s] and i in exam:
                x, y = "\n".join(A[s][i]).strip(), "\n".join(B[s][i]).strip()
                if x == y:
                    same += 1
                    continue
                out.append({"test": f"s{s}", "q": i, "card": exam[i], "A": x, "B": y, "ken": "=", "noted": False})
    return out, same


def vs_score(path, P, same, la, lb, boot=2000):
    import random
    J = {}
    for r in map(json.loads, path.read_text(encoding="utf-8").splitlines()):
        J.setdefault((r["test"], r["q"]), {})[r["order"]] = r["win"]
    res = []
    for it in P:
        v = J.get((it["test"], it["q"]), {})
        if len(v) == 2:
            res.append((it["q"], v["AB"] if v["AB"] == v["BA"] else "="))
    c = Counter(w for _, w in res)
    n = len(res) + same
    print(f"\n{la}（A）vs {lb}（B）：{n} 对（两条一样 {same} 对算平）")
    print(f"  A 好 {c['A']}，B 好 {c['B']}，差不多 / 两次不一致 {c['='] + same}；换顺序一致 {sum(1 for it in P if len(J.get((it['test'], it['q']), {})) == 2 and J[(it['test'], it['q'])]['AB'] == J[(it['test'], it['q'])]['BA'])}/{len(res)}")
    by = {}
    for q, w in res:
        by.setdefault(q, []).append({"A": 1, "B": -1, "=": 0}[w])
    cards = list(by)
    rnd = random.Random(0)
    diff = lambda cs: sum(sum(by[q]) for q in cs) / max(sum(len(by[q]) for q in cs), 1)
    bs = sorted(diff([rnd.choice(cards) for _ in cards]) for _ in range(boot))
    print(f"  净胜率（A 好 − B 好）/ 对数 = {diff(cards):+.3f}，按卡重抽样 95% 区间 [{bs[int(.025 * boot)]:+.3f}, {bs[int(.975 * boot)]:+.3f}]")
    print(f"  选了边的对里 A 占 {c['A'] / max(c['A'] + c['B'], 1):.0%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pair", "item", "vs"])
    ap.add_argument("out")
    ap.add_argument("--a", help="vs：版本 A 的输出前缀，如 data/cloud-v3/cdpo2s1t5cut-pen")
    ap.add_argument("--b", help="vs：版本 B 的输出前缀")
    ap.add_argument("--exam", default=str(ROOT / "exam" / "考卷.jsonl"))
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tests", nargs="*", help="pair / item：只用这几个盲测（data/human-check/ 下的目录名），默认是对尺子用的 4 个")
    a = ap.parse_args()
    if a.mode == "vs":
        P, same = vs_pairs(a.a, a.b, a.exam)
        if not a.score:
            run("pair", Path(a.out), P, a.workers)
        vs_score(Path(a.out), P, same, Path(a.a).name, Path(a.b).name)
        return
    P = pairs(a.tests)
    if not a.score:
        run(a.mode, Path(a.out), P, a.workers)
    score(a.mode, Path(a.out), P)


if __name__ == "__main__":
    main()
