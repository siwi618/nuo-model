"""人工核对后：用新尺子（v4）重判核对的 60 题，同时用 v3 再判一遍（量评审自己前后差多少），产出和 key.json 同格式的文件，
直接交给 human_check_score.py 和人的答案比（2026-11-23）。

产出（都在 key.json 同目录）：
  rejudge.jsonl      每次评审的原始记录
  key-v3重评.json    v3 再判一次（单评）+ 两两沿用旧试测
  key-v4.json        v4 单评（「心里话说出口」一栏用新的「该藏的说出口」）+ v4 两两（正反各一次）
  key-v4旧说出口.json  同 v4，但「心里话说出口」一栏用老定义（看只改其他几条时的效果）
用法（仓库根目录）：.venv/bin/python scripts/human_check_rejudge.py data/human-check/key.json data/human-check/items.json
已评过的会跳过，可以中断后接着跑。
"""
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from deepseek import chat_json  # noqa: E402
from judge_exam import judge_one  # noqa: E402
from judge_pair import MODEL, SYS_V4  # noqa: E402
from render_card import render_card  # noqa: E402

ROOT = HERE.parent


def load_exam(p):
    return {json.loads(t)["id"]: json.loads(t)["card"] for t in p.read_text(encoding="utf-8").splitlines() if t.strip()}


def main():
    keyp = Path(sys.argv[1])
    key = json.loads(keyp.read_text(encoding="utf-8"))
    items = {i["id"]: i for i in json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))}
    exams = {"pub": load_exam(ROOT / "exam" / "考卷.jsonl"), "sp": load_exam(ROOT / "exam" / "专项-心里话.jsonl")}
    out = keyp.parent / "rejudge.jsonl"
    done = set()
    if out.exists():
        for t in out.read_text(encoding="utf-8").splitlines():
            if t.strip():
                d = json.loads(t)
                done.add((d["id"], d["rule"], d.get("order", "")))
    jobs = []
    for i, k in key.items():
        if i.startswith("s"):
            jobs += [(i, r, "") for r in ("v3", "v4") if (i, r, "") not in done]
        else:
            jobs += [(i, "v4-pair", o) for o in ("AB", "BA") if (i, "v4-pair", o) not in done]
    print(f"要评 {len(jobs)} 次（已有 {len(done)}）")
    lock = threading.Lock()

    def one(job):
        i, rule, order = job
        k, it = key[i], items[i]
        card = exams[k.get("exam", "pub")][k["card_id"]]
        if rule in ("v3", "v4"):
            rec, _ = judge_one(card, it["lines"], rule)
        else:   # 两两：A / B 是页面上的 A / B；order = 先给评审看哪条
            fmt = lambda ls: "\n".join(f"{n}. {s}" for n, s in enumerate(ls, 1))  # noqa: E731
            x, y = (it["a"], it["b"]) if order == "AB" else (it["b"], it["a"])
            data, _ = chat_json(SYS_V4, f"{render_card(card)}\n\n【台词甲】\n{fmt(x)}\n\n【台词乙】\n{fmt(y)}", model=MODEL, thinking=True)
            pick = data.get("更像")
            rec = {"pick": {"甲": order[0], "乙": order[1]}.get(pick, "T"), "理由": data.get("理由", "")}
        rec.update({"id": i, "rule": rule, "order": order})
        with lock:
            with out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(8) as ex:
        for f in [ex.submit(one, j) for j in jobs]:
            try:
                f.result()
            except Exception as e:
                print("失败：", e, flush=True)

    recs = [json.loads(t) for t in out.read_text(encoding="utf-8").splitlines() if t.strip()]
    by = {}
    for r in recs:
        by.setdefault((r["id"], r["rule"]), []).append(r)

    def single(k, r, inner_new):
        probs = list(r["problems"])
        if r["rule"] == "v4":   # 打分脚本认「心里话说出口」这个名字：换成要比的那个定义
            probs = [p for p in probs if p != "该藏的说出口"]
            if (inner_new and "该藏的说出口" in r["problems"]) or (not inner_new and r.get("心里话说出口_旧")):
                probs.append("心里话说出口")
        return {**k, "problems": probs, "pass": not probs, "办成": r["办成"], "像不像": r["像不像"], "分项": r.get("分项"), "理由": r["理由"]}

    for name, rule, inner_new in [("key-v3重评.json", "v3", True), ("key-v4.json", "v4", True), ("key-v4旧说出口.json", "v4", False)]:
        nk = {}
        for i, k in key.items():
            if i.startswith("s"):
                rs = by.get((i, rule))
                if rs:
                    nk[i] = single(k, rs[0], inner_new)
            elif rule == "v3":
                nk[i] = k
            else:
                rs = by.get((i, "v4-pair"), [])
                if rs:   # 页面 A / B → 版本名，和 key.json 的 judge_votes 同格式
                    nk[i] = {**k, "judge_votes": [k["page_A"] if r["pick"] == "A" else k["page_B"] if r["pick"] == "B" else "T" for r in rs],
                             "理由": [r["理由"] for r in rs]}
        (keyp.parent / name).write_text(json.dumps(nk, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"→ {name}（{len(nk)} 题）")


if __name__ == "__main__":
    main()
