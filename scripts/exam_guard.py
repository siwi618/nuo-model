"""训练前检查：考卷不能混进训练数据，Claude 写的也不能进。

用法：python3 scripts/exam_guard.py 训练数据.jsonl [更多文件...] [--exam data/exam/考卷.jsonl]
有问题就列出来并以非 0 退出（训练脚本先跑它，不过不许训）。

查这几样：
1. 考卷里的 id、标了 exam 的样本；
2. 和考卷一模一样的戏卡；
3. 考卷里「新角色」「新世界」的说话人名字（训练里出现了，考卷就考不出「没见过」）；
4. 考卷「新世界」的腔调原文；
5. source 不是 deepseek / human / human_edited 的（Claude 写的不训练）。
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_card import render_card  # noqa: E402

TRAINABLE = {"deepseek", "human", "human_edited"}


def read_jsonl(p):
    return [json.loads(t) for t in Path(p).read_text(encoding="utf-8").splitlines() if t.strip()]


def fingerprint(card):
    return hashlib.sha1(render_card(card).encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--exam", default=str(Path(__file__).resolve().parent.parent / "data" / "exam" / "考卷.jsonl"))
    a = ap.parse_args()

    exam = read_jsonl(a.exam)
    ids = {e["id"] for e in exam}
    prints = {fingerprint(e["card"]): e["id"] for e in exam}
    unseen_names = {e["card"]["speaker"]["name"]: e["id"] for e in exam if e["meta"]["group"] in ("N", "W")}
    unseen_tones = {e["card"]["world"]["tone"]: e["meta"]["world"] for e in exam if e["meta"]["group"] == "W"}

    problems = []
    total = 0
    for f in a.files:
        for s in read_jsonl(f):
            total += 1
            where = f"{Path(f).name}:{s.get('id', '?')}"
            card = s.get("card", {})
            if s.get("id") in ids or s.get("meta", {}).get("exam"):
                problems.append(f"{where} 是考卷")
            if card and fingerprint(card) in prints:
                problems.append(f"{where} 戏卡和考卷 {prints[fingerprint(card)]} 一模一样")
            name = card.get("speaker", {}).get("name")
            if name in unseen_names:
                problems.append(f"{where} 说话人「{name}」是考卷 {unseen_names[name]} 的新角色")
            tone = card.get("world", {}).get("tone")
            if tone in unseen_tones:
                problems.append(f"{where} 用了考卷的新世界「{unseen_tones[tone]}」")
            if s.get("source") not in TRAINABLE:
                problems.append(f"{where} 来源是「{s.get('source')}」，不能训练")
    if problems:
        print(f"不通过：{len(problems)} 处问题（共查 {total} 条）")
        print("\n".join(problems))
        sys.exit(1)
    print(f"通过：{total} 条，没有考卷、没有不能训练的来源")


if __name__ == "__main__":
    main()
