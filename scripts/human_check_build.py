"""人工核对评审（2026-11-23）：抽 40 条单评 + 20 对两两对比，生成给 Ken 填的页面数据。

盲：页面数据（items.json）只有卡和台词，没有版本名、没有评审的判断；答案另存 key.json（只在本地）。
  单评 40 条：评审判全过 20；有硬伤 20 = 语句不通 12 + 心里话说出口 5 + 其他硬伤 3。
              来源：data/cloud-v3/judge-r3.jsonl（公开卷）+ judge-sp9-r3.jsonl（专项卷），都是 v3 标准。
  两两 20 对：dpo2-s1 vs v3 同卡同种子、公开卷，从位置偏好试测 pair-pilot.jsonl 的 40 对里抽（评审正反各判过一次）。
用法（仓库根目录）：.venv/bin/python scripts/human_check_build.py 输出目录
然后：页面发布 items.json；Ken 填完后用 human_check_score.py 算一致率。
"""
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from render_card import render_card  # noqa: E402
from score_exam import parse_output  # noqa: E402

ROOT = HERE.parent
V3 = ROOT / "data" / "cloud-v3"


def load_exam(p):
    return {json.loads(t)["id"]: json.loads(t)["card"] for t in p.read_text(encoding="utf-8").splitlines() if t.strip()}


def sections(card):
    """把 render_card 的文本按【】切开，页面上先显示关键几节，其余折叠。"""
    out, cur = {}, None
    for line in render_card(card).splitlines():
        if line.startswith("【"):
            cur = line[1:line.index("】")]
            out[cur] = [line[line.index("】") + 1:].strip()] if line[line.index("】") + 1:].strip() else []
        elif cur:
            out[cur].append(line)
    return {k: "\n".join(v).strip() for k, v in out.items()}


def main():
    outdir = Path(sys.argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    rnd = random.Random(20261123)
    exams = {"pub": load_exam(ROOT / "exam" / "考卷.jsonl"), "sp": load_exam(ROOT / "exam" / "专项-心里话.jsonl")}
    recs = []
    for fn, ex in [("judge-r3.jsonl", "pub"), ("judge-sp9-r3.jsonl", "sp")]:
        for t in (V3 / fn).read_text(encoding="utf-8").splitlines():
            r = json.loads(t)
            if r.get("rule") == "v3" and r["lines"].strip():
                r["exam"] = ex
                recs.append(r)
    rnd.shuffle(recs)

    used_lines, card_count = set(), {}
    picked = []

    def take(pred, n, tag):
        got = 0
        for r in recs:
            if got >= n:
                break
            key = (r["id"], r["lines"])
            if key in used_lines or card_count.get(r["id"], 0) >= 2 or not pred(r):
                continue
            used_lines.add(key)
            card_count[r["id"]] = card_count.get(r["id"], 0) + 1
            picked.append((tag, r))
            got += 1
        assert got == n, (tag, got)

    P = lambda r: set(r["problems"])  # noqa: E731
    take(lambda r: r["pass"] and r["exam"] == "pub", 12, "pass")
    take(lambda r: r["pass"] and r["exam"] == "sp", 8, "pass")
    take(lambda r: "语句不通" in P(r) and "心里话说出口" not in P(r), 12, "不通")
    take(lambda r: "心里话说出口" in P(r) and "语句不通" not in P(r), 5, "说出口")
    take(lambda r: not P(r) & {"语句不通", "心里话说出口"} and P(r) & {"不合人设", "称呼或人物错", "口头禅滥用", "说漏秘密"}, 3, "其他")
    rnd.shuffle(picked)

    items, key = [], {}
    for k, (tag, r) in enumerate(picked, 1):
        iid = f"s{k:02d}"
        items.append({"id": iid, "kind": "single", "card": sections(exams[r["exam"]][r["id"]]),
                      "lines": r["lines"].split("\n")})
        key[iid] = {"stratum": tag, "exam": r["exam"], "file": r["file"], "card_id": r["id"],
                    "problems": r["problems"], "pass": r["pass"], "办成": r["办成"], "像不像": r["像不像"],
                    "分项": r.get("分项"), "理由": r["理由"]}

    # 两两对比：试测 40 对里抽 20 对，台词不能一样
    pilot = {}
    for t in (V3 / "pair-pilot.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(t)
        pilot.setdefault((r["seed"], r["id"]), []).append(r)
    cands = []
    for (sd, cid), rs in sorted(pilot.items()):
        la = parse_output(V3 / f"cdpo2s1cut-pen-s{sd}.txt").get(cid, [])
        lb = parse_output(V3 / f"c3cut-pen-s{sd}.txt").get(cid, [])
        if la and lb and la != lb:
            cands.append((sd, cid, la, lb, rs))
    rnd.shuffle(cands)
    cards_used = set()
    pairs = []
    for c in cands:
        if c[1] in cards_used:
            continue
        cards_used.add(c[1])
        pairs.append(c)
        if len(pairs) == 20:
            break
    assert len(pairs) == 20, len(pairs)
    for k, (sd, cid, la, lb, rs) in enumerate(pairs, 1):
        iid = f"p{k:02d}"
        flip = rnd.random() < 0.5   # 页面上 A/B 随机放
        A, B = (lb, la) if flip else (la, lb)
        items.append({"id": iid, "kind": "pair", "card": sections(exams["pub"][cid]), "a": A, "b": B})
        votes = [("T" if r["pick"] == "差不多" else ("dpo2" if r["win"] == "A" else "v3")) for r in rs]
        key[iid] = {"seed": sd, "card_id": cid, "page_A": "v3" if flip else "dpo2", "page_B": "dpo2" if flip else "v3",
                    "judge_votes": votes, "理由": [r["理由"] for r in rs]}

    (outdir / "items.json").write_text(json.dumps(items, ensure_ascii=False, indent=0), encoding="utf-8")
    (outdir / "key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"单评 {sum(i['kind'] == 'single' for i in items)} 条，两两 {sum(i['kind'] == 'pair' for i in items)} 对 → {outdir}")


if __name__ == "__main__":
    main()
