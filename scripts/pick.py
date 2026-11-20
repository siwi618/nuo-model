"""挑稿小工具（2026-11-20，方案③：DeepSeek 不审稿，Claude 挑稿只决定留 / 不留）。

  show：把样本列成好读的样子（只列要看的：对方、场景、前文、目的、心思原文、答案、机器检查）
    .venv/bin/python scripts/pick.py show data/novel-3/samples.json [起 止]
  brief：省 token 版（只列挑稿用得上的）；加 unpicked 只列还没挑过的
    .venv/bin/python scripts/pick.py brief data/batch-3/samples.json unpicked 0 77
  apply：把挑稿结果写回样本 meta（keep / kept_by / notes），并把要从头重写的写进同目录 redo.json
    .venv/bin/python scripts/pick.py apply data/novel-3/samples.json 挑稿.json
    挑稿.json = {"id": {"keep": true/false, "why": "一句话", "redo": true/false}, ...}
    redo = 不留、并且让 DeepSeek 从头重写一次（不给任何意见）。notes 只是挑稿记录，不进训练数据（to_mlx 只取卡和台词）。
"""

import json
import sys
from pathlib import Path


def show(path, lo=0, hi=None):
    samples = json.loads(Path(path).read_text(encoding="utf-8"))
    for s in samples[lo:hi]:
        c, m = s["card"], s["meta"]
        g, sp, ls = c["goal"], c["speaker"], c["listener"]
        ref = m.get("ref", {})
        print(f"#### {s['id']}　{ref.get('book') or m.get('world')}　{sp['name']} → {ls['name']}"
              + (f"　[{m.get('inner_style')}]" if m.get("inner_style") else "")
              + (f"　机器检查：{'；'.join(m['auto_check'])}" if m.get("auto_check") else ""))
        print(f"  说话人：{sp.get('summary', '')}｜{sp.get('personality', '')}｜说话：{sp.get('speech', '')}")
        if sp.get("secrets"):
            print(f"  不能说：{sp['secrets']}")
        print(f"  对方：{ls.get('summary', '')}｜看法：{ls.get('view', '')}" + (f"｜眼下：{ls['now']}" if ls.get("now") else "")
              + (f"｜在场：{[o.get('name') for o in ls.get('others', [])]}" if ls.get("others") else ""))
        sc = c.get("scene", {})
        print(f"  场景：{sc.get('place_time', '')}｜{sc.get('backstory', '')}" + (f"｜已发生：{sc.get('happened')}" if sc.get("happened") else ""))
        for h in c.get("history", []):
            print(f"  前文 {h['name']}: {h['text']}")
        print(f"  要办成：{g.get('must', '')}")
        print(f"  心里想的：{g.get('inner', '')}" + (f"　（心思原文：{m['thought']}）" if m.get("thought") else ""))
        if g.get("dont_leak"):
            print(f"  别说漏：{g['dont_leak']}")
        r = c["rules"]
        print(f"  要求：{r['use']} 句 {r['lines']} 字 ≤{r['max_chars']} 动作 {r['actions']}")
        for o in s["output"]:
            print(f"  ▶ {o['name']}{'[' + o['emotion'] + ']' if o.get('emotion') else ''}: {o['text']}")
        print()


def cut(t, n):
    t = str(t or "")
    return t if len(t) <= n else t[:n] + "…"


def brief(path, lo=0, hi=None, unpicked=False):
    """省 token 版（11-20）：只列挑稿用得上的——人设只留性格 / 说话，长字段截短，不列世界设定、例句、想要怕。"""
    samples = json.loads(Path(path).read_text(encoding="utf-8"))
    if unpicked:
        samples = [s for s in samples if s["meta"].get("keep") is None]
    for s in samples[lo:hi]:
        c, m = s["card"], s["meta"]
        g, sp, ls, r = c["goal"], c["speaker"], c["listener"], c["rules"]
        head = f"## {s['id']}" + (f" [{m['inner_style']}]" if m.get("inner_style") else "") + \
            f" {sp['name']}→{ls['name']}（{cut(ls.get('summary'), 24)}）" + \
            f" {r['use']} 句{r['lines'][0]}-{r['lines'][1]} ≤{r['max_chars']}字 动作{'可' if r['actions'] else '不可'}"
        if m.get("auto_check"):
            head += " 机器:" + "；".join(m["auto_check"])
        print(head)
        print(f"性格:{cut(sp.get('personality'), 40)}｜说话:{cut(sp.get('speech'), 60)}")
        if sp.get("secrets"):
            print("不能说:" + "；".join(cut(x, 50) for x in sp["secrets"]))
        if ls.get("now") or ls.get("others"):
            print(f"对方眼下:{cut(ls.get('now'), 30)}" + (f" 在场:{[o.get('name') for o in ls.get('others', [])]}" if ls.get("others") else ""))
        for h in c.get("history", []):
            print(f"前 {h['name']}: {cut(h['text'], 60)}")
        print(f"办:{g.get('must', '')}")
        print(f"心:{g.get('inner', '')}" + (f"（原文:{cut(m['thought'], 50)}）" if m.get("thought") else ""))
        if g.get("dont_leak"):
            print("漏:" + "；".join(cut(x, 40) for x in g["dont_leak"]))
        for o in s["output"]:
            print(f"▶{o['name']}{'[' + o['emotion'] + ']' if o.get('emotion') else ''}: {o['text']}")


def apply(path, picks_path):
    p = Path(path)
    samples = json.loads(p.read_text(encoding="utf-8"))
    picks = json.loads(Path(picks_path).read_text(encoding="utf-8"))
    by = {s["id"]: s for s in samples}
    missing = [i for i in picks if i not in by]
    if missing:
        sys.exit(f"挑稿里有样本里没有的 id：{missing[:5]}")
    for i, v in picks.items():
        by[i]["meta"]["keep"] = bool(v["keep"])
        by[i]["meta"]["kept_by"] = "claude"
        by[i]["meta"]["notes"] = v.get("why", "")
    p.write_text(json.dumps(samples, ensure_ascii=False, indent=1), encoding="utf-8")
    redo = sorted(i for i, v in picks.items() if v.get("redo") and not v["keep"])
    rp = p.parent / "redo.json"          # 分批挑稿：和已有名单合并（已经重写过的，redo 步骤会自己跳过）
    old = json.loads(rp.read_text(encoding="utf-8")) if rp.exists() else []
    rp.write_text(json.dumps(sorted(set(old) | set(redo)), ensure_ascii=False), encoding="utf-8")
    kept = sum(1 for v in picks.values() if v["keep"])
    print(f"写回 {len(picks)} 条：留 {kept}，不留 {len(picks) - kept}（其中从头重写 {len(redo)}）；没挑的 {len(samples) - len(picks)}")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "brief":
        flags = [x for x in sys.argv[3:] if not x.isdigit()]
        a = [int(x) for x in sys.argv[3:] if x.isdigit()][:2]
        brief(sys.argv[2], *a, unpicked="unpicked" in flags)
    elif len(sys.argv) >= 3 and sys.argv[1] == "show":
        a = [int(x) for x in sys.argv[3:5]]
        show(sys.argv[2], *a)
    elif len(sys.argv) == 4 and sys.argv[1] == "apply":
        apply(sys.argv[2], sys.argv[3])
    else:
        sys.exit(__doc__)
