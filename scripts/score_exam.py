"""按考卷每张卡的【要求】给模型写的台词打分（只查格式，不查口气、目的、守秘——那些要人看）。

查的项目：句数、每句字数（不算动作括号）、「动作：不写」时写了括号、表情不在清单里、
表情写进括号（「（傲慢）」而不是「[傲慢]」）、不该开口的人开口、一行不是「名字[表情]: 话」的格式、
两句里有同一个 4 字以上片段（重复）、现代 / 科幻类世界里冒出文言字。

输入是 try_exam.py 的输出文件（「=== 考卷id（名字）」后面跟台词行）。

用法（在仓库根目录跑）：
  .venv/bin/python scripts/score_exam.py 训练前=data/x/before.txt 第400步=data/x/after-400.txt
  .venv/bin/python scripts/score_exam.py ... -v          # 逐张卡列出各版本台词和问题
  .venv/bin/python scripts/score_exam.py ... --exam exam/考卷.jsonl
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from render_card import render_card  # noqa: E402

LINE = re.compile(r"^(.+?)(?:\[(.+?)\])?[:：]\s*(.*)$")
ACT = re.compile(r"[（(][^）)]*[）)]")
# 现代 / 科幻类世界里冒出文言字 = 「老头腔」串台（2026-11-19 Ken：别把模型调成满嘴之乎者也的清朝老头）
MODERN_WORLDS = {"现代", "科幻", "太空殖民", "刑侦", "校园", "废土"}
ARCHAIC = set("之乎者矣焉哉兮欤汝吾尔曰")
MODERN_OK = ["之后", "之前", "之间", "之一", "之外", "之内", "之中", "之下", "之上", "总之", "随之", "之所以",
             "偶尔", "莞尔", "或者", "记者", "作者", "读者", "患者", "老者", "学者", "使者", "前者", "后者", "再者"]


PUNCT = re.compile(r"[，。！？、；：,.!?;:“”\"'‘’（）()…—\s\[\]]")


def repeated(texts, n=4):
    """同一个 4 字片段出现在两句里（2026-11-19 Ken 指出「您放一百个心」连说两遍）。返回重复片段。"""
    seen, hits = {}, set()
    for k, t in enumerate(texts):
        t = PUNCT.sub("", ACT.sub("", t))
        for g in {t[i:i + n] for i in range(len(t) - n + 1)}:
            if g in seen and seen[g] != k:
                hits.add(g)
            seen.setdefault(g, k)
    return sorted(hits)


def archaic_chars(text):
    for w in MODERN_OK:
        text = text.replace(w, "")
    return [c for c in text if c in ARCHAIC]


def split_list(s):
    return [x.strip() for x in re.split("[、,，]", s) if x.strip()]


def load_rules(exam_path):
    rules = {}
    for line in Path(exam_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        text = render_card(d["card"])
        req = text[text.index("【要求】"):]

        def get(key):
            m = re.search(key + r"：(.*)", req)
            return m.group(1).strip() if m else ""

        nums = re.findall(r"\d+", get("句数")) or ["1", "9"]
        rules[d["id"]] = {
            "lo": int(nums[0]),
            "hi": int(nums[-1]),
            "max_chars": int((re.findall(r"\d+", get("每句最多")) or ["999"])[0]),
            "no_action": get("动作").startswith("不写"),
            "action": get("动作"),
            "emotions": split_list(get("表情")),
            "speakers": split_list(get("谁能开口")),
            "use": get("用法"),
            "modern": (d.get("meta") or {}).get("world", "") in MODERN_WORLDS,
        }
    return rules


def parse_output(path):
    out, cur = {}, None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = re.match(r"=== (\S+?)（", line)
        if m:
            cur = m.group(1)
            out[cur] = []
        elif cur and line.strip():
            out[cur].append(line.strip())
    return out


def check(lines, r):
    problems = []
    if not r["lo"] <= len(lines) <= r["hi"]:
        problems.append(f"句数{len(lines)}")
    texts = [m.group(3) for m in map(LINE.match, lines) if m]
    rep = repeated(texts)
    if rep:
        problems.append(f"重复:{rep[0]}")
    for s in lines:
        m = LINE.match(s)
        if not m:
            problems.append("格式坏")
            continue
        name, emo, text = m.groups()
        if name.strip() not in r["speakers"]:
            problems.append(f"错人:{name.strip()[:6]}")
        if emo and emo not in r["emotions"]:
            problems.append(f"表情外:{emo}")
        acts = ACT.findall(text)
        if acts and r["no_action"]:
            problems.append("写了动作")
        if any(a.strip("（）()") in r["emotions"] for a in acts):
            problems.append("表情写进括号")
        if len(ACT.sub("", text)) > r["max_chars"]:
            problems.append("超字数")
        if r.get("modern") and archaic_chars(text):
            problems.append(f"文言字:{''.join(archaic_chars(text))}")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="标签=try_exam 输出文件")
    ap.add_argument("--exam", default=str(HERE.parent / "exam" / "考卷.jsonl"))
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    rules = load_rules(a.exam)
    runs = {}
    for item in a.runs:
        label, sep, path = item.partition("=")
        if not sep:
            label, path = Path(item).stem, item
        runs[label] = parse_output(path)

    for label, res in runs.items():
        results = {i: check(res.get(i, []), r) for i, r in rules.items() if i in res}
        missing = len(rules) - len(results)
        counts = Counter(p.split(":")[0] for ps in results.values() for p in ps)
        ok = sum(1 for ps in results.values() if not ps)
        extra = f"（缺 {missing} 张）" if missing else ""
        print(f"{label}: 全合规 {ok}/{len(results)}{extra}  问题 {dict(counts)}")

    if a.verbose:
        for i, r in rules.items():
            print(f"\n##### {i}  要求:{r['use']} 句数{r['lo']}-{r['hi']} 动作:{r['action']} 开口:{'、'.join(r['speakers'])}")
            for label, res in runs.items():
                if i not in res:
                    continue
                print(f"[{label}] {check(res[i], r) or 'OK'}")
                for s in res[i]:
                    print("   ", s)


if __name__ == "__main__":
    main()
