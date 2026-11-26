"""验证：素材从【说话人】「知道」挪到【场景】「提得到的」，模型还会不会挑着用（2026-11-26，Ken 的「场景标签化」想法）。

为什么：「知道」在档案前缀里，要逐字不变才能缓存（约 650 token、预读约 1 秒）；游戏每次现填的素材 / 看得见的东西
只能放在后面每次现算的部分。模型训练时素材都在「知道」里，「提得到的」这一行它没见过。

  build  → data/enrich/素材卷-场景.jsonl（素材卷 12 张，同样的素材挪到 scene.at_hand；「知道」只留原卡的）
  stats 前缀A 前缀B ... → 每版用到素材的比例、超 2 条、照搬原句（机器数）
"""
import copy
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

RICH = ROOT / "data/enrich/素材卷.jsonl"
OUT = ROOT / "data/enrich/素材卷-场景.jsonl"


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def split(k):
    return [x.strip() for x in re.split(r"[；;。]", k or "") if x.strip()]


def orig_cards():
    o = {}
    for f in ("exam/考卷.jsonl", "exam/专项-心里话.jsonl"):
        for d in load(ROOT / f):
            o[d["id"]] = d
    return o


def more_of(s, orig):
    if "enrich" in s.get("meta", {}):
        return s["meta"]["enrich"]["knows_more"]
    base = set(split(orig[s["id"]]["card"]["speaker"].get("knows")))
    return [x for x in split(s["card"]["speaker"]["knows"]) if x not in base]


def build():
    orig, out = orig_cards(), []
    for s in load(RICH):
        more = more_of(s, orig)
        t = copy.deepcopy(s)
        t["card"]["speaker"]["knows"] = orig[s["id"]]["card"]["speaker"].get("knows", "")
        t["card"]["scene"]["at_hand"] = more
        t.setdefault("meta", {})["at_hand_test"] = {"moved": len(more)}
        out.append(t)
        print(s["id"], f"挪了 {len(more)} 条")
    OUT.write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in out), encoding="utf-8")


def stats(prefixes):
    sys.argv = sys.argv[:1]
    import batch4
    from score_exam import parse_output
    orig = orig_cards()
    rich = {s["id"]: s for s in load(RICH)}

    def lines(ls):
        out = []
        for t in ls:
            m = re.match(r"^(\S+?)(?:\[[^\]]*\])?[:：]\s*(.*)$", t)
            if m:
                out.append({"name": m.group(1), "text": m.group(2)})
        return out
    for pre in prefixes:
        n = used1 = over = echo = tot = 0
        for p in sorted(Path(pre).parent.glob(Path(pre).name + "-s*.txt")):
            for i, ls in parse_output(p).items():
                c = rich[i]["card"]       # 量法统一用「知道」版的卡（素材在 knows 里，照搬检查才查得到）
                pr, u = batch4.material_flags(c, lines(ls), more_of(rich[i], orig))
                n += 1; tot += len(u); used1 += bool(u); over += len(u) > 2; echo += any("照搬" in x for x in pr)
        print(f"{pre}: {n} 条 | 用到 ≥1 条素材 {used1}（{used1 / max(n, 1):.0%}）| 平均 {tot / max(n, 1):.2f} 条 | 超 2 条 {over} | 照搬原句 {echo}")


if __name__ == "__main__":
    {"build": lambda: build(), "stats": lambda: stats(sys.argv[2:])}[sys.argv[1]]()
