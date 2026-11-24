"""「哪条更顺」重测（2026-11-24 对尺子）：量 Ken 自己前后判得一致不一致。

从已做过的 4 次盲测（smooth / smooth-eqlen / smooth-dpo4 / smooth-temp05）里每次抽 5 对，共 20 对；
不抽 Ken 写过旁注的对（写过的容易记住答案）；左右和原来对调，顺序重新打乱。页面和原来的一样（smooth_check_build 的模板）。
做完给一行结果码 SMOOTH1|…，交给 smooth_retest_score.py。

用法（仓库根目录）：.venv/bin/python scripts/smooth_retest_build.py data/human-check/smooth-retest
"""
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from human_check_build import load_exam, sections  # noqa: E402
from smooth_check_build import TEMPLATE, ver_seed  # noqa: E402

ROOT = HERE.parent
HC = ROOT / "data" / "human-check"
V3 = ROOT / "data" / "cloud-v3"
TESTS = ["smooth", "smooth-eqlen", "smooth-dpo4", "smooth-temp05"]


def load_lines():
    cell = {}
    for fn, ex in [("judge-r4.jsonl", "pub"), ("judge-sp9-r4.jsonl", "sp")]:
        for t in (V3 / fn).read_text(encoding="utf-8").splitlines():
            r = json.loads(t)
            v, s = ver_seed(r["file"])
            cell[(ex, v, r["id"], s)] = r["lines"].strip()
    return cell


def load_test(name):
    _, prefs, notes = (HC / name / "Ken.txt").read_text(encoding="utf-8").strip().split("|", 2)
    key = json.loads((HC / name / "key.json").read_text(encoding="utf-8"))
    return key, prefs, json.loads(notes)


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    rnd = random.Random(20261124)
    cell = load_lines()
    exams = {"pub": load_exam(ROOT / "exam" / "考卷.jsonl"), "sp": load_exam(ROOT / "exam" / "专项-心里话.jsonl")}
    items, key = [], []
    for t in TESTS:
        k0, prefs, notes = load_test(t)
        pool = [(k, p) for k, p in zip(k0, prefs) if k["id"] not in notes]
        for k, p in rnd.sample(pool, 5):
            a = cell[(k["exam"], k["A"], k["card"], k["seed"])]
            b = cell[(k["exam"], k["B"], k["card"], k["seed"])]
            items.append({"card": sections(exams[k["exam"]][k["card"]]), "a": b.splitlines(), "b": a.splitlines()})   # 左右对调
            key.append({"test": t, "orig_q": k["id"], "exam": k["exam"], "card": k["card"], "seed": k["seed"],
                        "A": k["B"], "B": k["A"], "orig_pref": {"A": k["A"], "B": k["B"], "=": "="}[p]})
    order = list(range(len(items)))
    rnd.shuffle(order)
    items = [items[i] for i in order]
    key = [key[i] for i in order]
    for i, (it, k) in enumerate(zip(items, key), 1):
        it["id"] = k["id"] = f"q{i:02d}"
    (out / "key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    data = json.dumps(items, ensure_ascii=False).replace("</", "<\\/")
    page = TEMPLATE.replace("/*ITEMS*/", data).replace("smooth-check-1", "smooth-check-" + out.name)
    (out / "哪条更顺.html").write_text(page, encoding="utf-8")
    print(f"{len(items)} 对 → {out / '哪条更顺.html'}；原来选了一边 {sum(k['orig_pref'] != '=' for k in key)} 对、差不多 {sum(k['orig_pref'] == '=' for k in key)} 对")


if __name__ == "__main__":
    main()
