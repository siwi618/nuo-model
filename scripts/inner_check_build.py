"""「说出口」专项核对（2026-11-23）：从 v4 全量重评里挑新旧两种判法不一致的台词，加几条一致的当对照，给 Ken 盲判「算不算说出口」。

  旧 = 心里话说出口（v1 原文）；新 = 该藏的说出口（办事必要的话 / 收着说 / 世界规定的回法不算）。
  抽：新旧不一致的全取 + 两边都是 10 + 两边都否补足到 30 条；每张卡最多 3 条；
      避开上午人工核对已经用过的台词（data/human-check/key.json）。
  只挑：卡体检（exam/心思体检.json）为「藏 / 部分」的卡 + v4 没判语句不通的台词（11-23 Ken：卡上没东西可藏、台词读不懂都没法判说出口）。
页面数据 items.json 不含任何判断；答案 key.json 只在本地。
用法（仓库根目录）：.venv/bin/python scripts/inner_check_build.py 输出目录
"""
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from human_check_build import load_exam, sections  # noqa: E402

ROOT = HERE.parent
V3 = ROOT / "data" / "cloud-v3"


def main():
    outdir = Path(sys.argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    rnd = random.Random(20261124)
    exams = {"pub": load_exam(ROOT / "exam" / "考卷.jsonl"), "sp": load_exam(ROOT / "exam" / "专项-心里话.jsonl")}
    used = set()
    old_key = ROOT / "data" / "human-check" / "key.json"
    if old_key.exists():
        items = json.loads((ROOT / "data" / "human-check" / "items.json").read_text(encoding="utf-8"))
        for it in items:
            if it["kind"] == "single":
                used.add("\n".join(it["lines"]))
    audit = json.loads((ROOT / "exam" / "心思体检.json").read_text(encoding="utf-8"))   # 只用有东西可藏的卡
    recs = []
    for fn, ex in [("judge-r4.jsonl", "pub"), ("judge-sp9-r4.jsonl", "sp")]:
        for t in (V3 / fn).read_text(encoding="utf-8").splitlines():
            r = json.loads(t)
            card = exams[ex].get(r["id"], {})
            if (r.get("rule") == "v4" and r["lines"].strip() and r["lines"] not in used and card.get("goal", {}).get("inner")
                    and audit[r["id"]]["类"] in ("藏", "部分") and "语句不通" not in r["problems"]):
                r["exam"] = ex
                r["old"] = bool(r.get("心里话说出口_旧"))
                r["new"] = "该藏的说出口" in r["problems"]
                recs.append(r)
    rnd.shuffle(recs)
    seen_lines, per_card, picked = set(), {}, []

    def take(pred, n, tag):
        got = 0
        for r in recs:
            if got >= n:
                break
            if r["lines"] in seen_lines or per_card.get(r["id"], 0) >= 3 or not pred(r):
                continue
            seen_lines.add(r["lines"])
            per_card[r["id"]] = per_card.get(r["id"], 0) + 1
            picked.append((tag, r))
            got += 1
        return got

    counts = {
        "旧是新否": take(lambda r: r["old"] and not r["new"], 99, "旧是新否"),
        "新是旧否": take(lambda r: r["new"] and not r["old"], 99, "新是旧否"),
        "都是": take(lambda r: r["old"] and r["new"], 10, "都是"),
        "都否": take(lambda r: not r["old"] and not r["new"], 30 - len(picked), "都否"),
    }
    pool = {"旧是新否": sum(r["old"] and not r["new"] for r in recs), "新是旧否": sum(r["new"] and not r["old"] for r in recs),
            "都是": sum(r["old"] and r["new"] for r in recs), "都否": sum(not r["old"] and not r["new"] for r in recs)}
    rnd.shuffle(picked)
    items, key = [], {}
    for k, (tag, r) in enumerate(picked, 1):
        iid = f"q{k:02d}"
        items.append({"id": iid, "kind": "single", "card": sections(exams[r["exam"]][r["id"]]), "lines": r["lines"].split("\n")})
        key[iid] = {"stratum": tag, "exam": r["exam"], "file": r["file"], "card_id": r["id"], "old": r["old"], "new": r["new"],
                    "problems": r["problems"], "理由": r["理由"]}
    (outdir / "items.json").write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    (outdir / "key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"候选池（藏 / 部分卡、读得通、没在上午用过）：{pool}")
    print(f"抽到：{counts}，共 {len(items)} 条 → {outdir}")


if __name__ == "__main__":
    main()
