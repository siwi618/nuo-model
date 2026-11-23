"""人工核对评审：算人和 DeepSeek 评审（v3）的一致率，多人时再算人和人之间的一致率（2026-11-23）。

输入：human_check_build.py 生成的 key.json，加一份或几份人的答案（「名字=路径」，只有一份时可以不写名字）。路径可以是：
  - 目录：每题一个 <id>.json（核对页 db 的 answers 集合用 ArtifactData list 的 out_dir 存出来）；
  - JSON 文件：{id: 答案}（核对页没连上存储时的导出）；
  - 文本文件：朋友版测验（台词耳朵测试.html）给的一行结果码「台词耳朵测试 | 名字 | NUO1…」，解码要用同一份 items.json 的题目顺序。
答案格式：单评 {"flags": [...], "done": 0/1/2, "note"}；两两 {"pick": "A"/"B"/"T"}。
用法（仓库根目录）：
  .venv/bin/python scripts/human_check_score.py data/human-check/key.json data/human-check/answers -v
  .venv/bin/python scripts/human_check_score.py data/human-check/key.json Ken=data/human-check/answers 小王=data/human-check/小王.txt

按项分别算：语句不通、心里话说出口、全过（无硬伤且办成 ≥ 1）、哪条更像；另附其他硬伤和办成。
每项给一致率、Cohen's κ（扣掉碰巧一致）。一份答案时给 2×2 表和（-v）不一致明细；多份时给「评审—每个人」「人—人」「评审—人的多数」三类对比：
评审和人的一致率接近人和人之间的一致率，尺子就算合格。样本小（40 条 / 20 对），只看大概。
"""
import itertools
import json
import re
import sys
from pathlib import Path

FLAGS = ["心里话说出口", "说漏秘密", "称呼或人物错", "出戏", "语句不通", "不合人设", "口头禅滥用"]
B32 = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def decode(code, ids):
    """朋友版测验的结果码 → {id: 答案}。编码见 human_check_quiz.html 的 encode()。"""
    bits = "".join(f"{B32.index(c):05b}" for c in code)
    nbits = sum(9 if i.startswith("s") else 2 for i in ids)
    nbytes = (nbits + 7) // 8
    bs = [int(bits[k:k + 8], 2) for k in range(0, (nbytes + 1) * 8, 8)]
    assert sum(bs[:-1]) & 255 == bs[-1], "结果码校验不对，可能复制漏了字"
    out, pos = {}, 0
    for i in ids:
        if i.startswith("s"):
            fl = [f for k, f in enumerate(FLAGS) if bits[pos + k] == "1"]
            d = int(bits[pos + 7:pos + 9], 2)
            pos += 9
            if d != 3:
                out[i] = {"flags": fl, "done": d}
        else:
            v = int(bits[pos:pos + 2], 2)
            pos += 2
            if v:
                out[i] = {"pick": "_ABT"[v]}
    return out


def load_answers(p, ids):
    p = Path(p)
    if p.is_dir():
        out = {}
        for f in p.rglob("*.json"):
            d = json.loads(f.read_text(encoding="utf-8"))
            out[f.stem] = d["data"] if isinstance(d.get("data"), dict) else d
        return out
    t = p.read_text(encoding="utf-8")
    m = re.search(r"NUO1([0-9A-Z]+)", t)
    if m:
        return decode(m.group(1), ids)
    return json.loads(t)


def kappa(pairs):
    n = len(pairs)
    if not n:
        return None
    labels = {x for p in pairs for x in p}
    po = sum(a == b for a, b in pairs) / n
    pe = sum((sum(a == l for a, _ in pairs) / n) * (sum(b == l for _, b in pairs) / n) for l in labels)
    return None if pe == 1 else (po - pe) / (1 - pe)


def fmt(pairs):
    if not pairs:
        return "—"
    k = kappa(pairs)
    a = sum(x == y for x, y in pairs)
    return f"{a}/{len(pairs)}（{a / len(pairs):.0%}，κ {'—' if k is None else f'{k:.2f}'}）"


def table(pairs):
    labels = sorted({x for p in pairs for x in p}, key=str)
    print("  评审＼人  " + "  ".join(f"{str(l):>5}" for l in labels))
    for a in labels:
        print(f"  {str(a):>7}  " + "  ".join(f"{sum(1 for x, y in pairs if x == a and y == b):>5}" for b in labels))


def labels_of(ans, ids):
    """一份答案 → 每项的标签 {项: {id: 值}}，只含答了的题。"""
    L = {"语句不通": {}, "心里话说出口": {}, "全过": {}, "办成": {}, "哪条更像": {}}
    for i in ids:
        a = ans.get(i)
        if not a:
            continue
        if i.startswith("s") and a.get("done") is not None:
            fl = a.get("flags") or []
            L["语句不通"][i] = "语句不通" in fl
            L["心里话说出口"][i] = "心里话说出口" in fl
            L["全过"][i] = not fl and a["done"] >= 1
            L["办成"][i] = a["done"]
        elif i.startswith("p") and a.get("pick"):
            L["哪条更像"][i] = a["pick"]
    return L


def judge_labels(key):
    L = {"语句不通": {}, "心里话说出口": {}, "全过": {}, "办成": {}, "哪条更像": {}}
    for i, k in key.items():
        if i.startswith("s"):
            L["语句不通"][i] = "语句不通" in k["problems"]
            L["心里话说出口"][i] = "心里话说出口" in k["problems"]
            L["全过"][i] = k["pass"]
            L["办成"][i] = k["办成"]
        else:   # 评审正反各判一次；两次一致才算有明确选择，否则记「差不多」；换成页面上的 A / B
            v = k["judge_votes"]
            win = v[0] if len(set(v)) == 1 else "T"
            L["哪条更像"][i] = "T" if win == "T" else ("A" if k["page_A"] == win else "B")
    return L


def compare(x, y):
    return [(x[i], y[i]) for i in x if i in y]


def detail(key, ans, verbose):
    """一份答案时的详细报告。"""
    J = judge_labels(key)
    H = labels_of(ans, list(key))
    for m in ["语句不通", "心里话说出口", "全过", "哪条更像"]:
        pr = compare(J[m], H[m])
        print(f"\n{m}：{fmt(pr)}")
        if pr:
            table(pr)
        if verbose:
            for i in H[m]:
                if J[m][i] != H[m][i]:
                    if i.startswith("s"):
                        print(f"    {i} 评审{key[i]['problems'] or '全过'} 人{ans[i].get('flags') or '无'} 办成{ans[i]['done']}  {key[i]['理由']}")
                    else:
                        print(f"    {i} 评审{J[m][i]} 人{H[m][i]}（A={key[i]['page_A']}）  {key[i]['理由'][0]}")
    pr = compare(J["哪条更像"], H["哪条更像"])
    strict = [(a, b) for a, b in pr if "T" not in (a, b)]
    if strict:
        print(f"  只看双方都选了一边的：{sum(a == b for a, b in strict)}/{len(strict)} 一致")
    kv = [key[i]["page_A"] if p == "A" else key[i]["page_B"] if p == "B" else "T" for i, p in H["哪条更像"].items()]
    print(f"  人的选择：dpo2-s1 胜 {kv.count('dpo2')}、v3 胜 {kv.count('v3')}、差不多 {kv.count('T')}")
    ds = list(H["办成"])
    print("\n其他硬伤（评审是 / 人是 / 两边都是）：")
    for f in FLAGS:
        if f not in ("语句不通", "心里话说出口"):
            j = sum(f in key[i]["problems"] for i in ds)
            k = sum(f in (ans[i].get("flags") or []) for i in ds)
            b = sum(f in key[i]["problems"] and f in (ans[i].get("flags") or []) for i in ds)
            print(f"  {f}：{j} / {k} / {b}")
    print(f"\n办成（0/1/2）：{fmt(compare(J['办成'], H['办成']))}")
    print("按抽样层看全过一致：")
    for s in ["pass", "不通", "说出口", "其他"]:
        ids = [i for i in ds if key[i]["stratum"] == s]
        if ids:
            print(f"  {s}：{sum(J['全过'][i] == H['全过'][i] for i in ids)}/{len(ids)}")
    notes = [(i, a["note"]) for i, a in ans.items() if (a.get("note") or "").strip()]
    if notes:
        print("\n备注：")
        for i, t in sorted(notes):
            print(f"  {i}：{t}")


def majority(Hs, m):
    out = {}
    for i in set().union(*(h[m] for h in Hs)):
        vs = [h[m][i] for h in Hs if i in h[m]]
        if len(vs) < 2:
            continue
        top = max(set(vs), key=vs.count)
        if vs.count(top) * 2 > len(vs):   # 过半才算，打平不算
            out[i] = top
    return out


def main():
    args = [a for a in sys.argv[1:] if a != "-v"]
    verbose = "-v" in sys.argv
    key = json.loads(Path(args[0]).read_text(encoding="utf-8"))
    ids = list(key)
    raters = {}
    for k, spec in enumerate(args[1:], 1):
        name, _, path = spec.rpartition("=") if "=" in spec else (f"人{k}", "", spec)
        raters[name] = load_answers(path, ids)
    for name, ans in raters.items():
        ns = sum(1 for i in ids if i.startswith("s") and ans.get(i, {}).get("done") is not None)
        npair = sum(1 for i in ids if i.startswith("p") and ans.get(i, {}).get("pick"))
        print(f"{name}：单评 {ns}/40，两两 {npair}/20")
    if len(raters) == 1:
        detail(key, next(iter(raters.values())), verbose)
        return
    J = judge_labels(key)
    H = {n: labels_of(a, ids) for n, a in raters.items()}
    for m in ["语句不通", "心里话说出口", "全过", "哪条更像", "办成"]:
        print(f"\n【{m}】")
        for n in H:
            print(f"  评审—{n}：{fmt(compare(J[m], H[n][m]))}")
        for a, b in itertools.combinations(H, 2):
            print(f"  {a}—{b}：{fmt(compare(H[a][m], H[b][m]))}")
        if len(H) >= 3:
            print(f"  评审—人的多数：{fmt(compare(J[m], majority(list(H.values()), m)))}")
    if verbose:
        for n, ans in raters.items():
            print(f"\n======== {n} 的详细 ========")
            detail(key, ans, True)


if __name__ == "__main__":
    main()
