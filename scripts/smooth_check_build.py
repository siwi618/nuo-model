"""「哪条更顺」盲测（2026-11-24）：定 dpo3-s2 之前，Ken 用耳朵核对第三轮的「语句不通」下降是不是真的。

抽 20 对 dpo3-s2 vs dpo2-s1，同卡同种子（公开卷 12 + 专项卷 8，每张卡最多 1 对，两条台词不一样），随机左右。
11-24 第四轮：--a cdpo4s1（dpo4-s1 vs dpo2-s1，同长度）。
页面（单文件、离线）只有卡和台词，没有版本名、没有评审判断；答案对照另存 key（只在本地）。
Ken 每对选「A 更顺 / B 更顺 / 差不多」，做完给一行结果码 SMOOTH1|…，交给 smooth_check_score.py。

用法（仓库根目录）：.venv/bin/python scripts/smooth_check_build.py data/human-check/smooth
同长度版（11-24 第 2 次）：... smooth_check_build.py data/human-check/smooth-eqlen --maxdiff 0.15 --exclude data/human-check/smooth/key.json
  --maxdiff：两条字数差 / 较长那条 < 这个比例才抽（字数不算【表情】和空白）；--exclude：优先不用这些 key 里出过的卡（不够再用）。
"""
import argparse
import html
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from human_check_build import load_exam, sections  # noqa: E402

ROOT = HERE.parent
V3 = ROOT / "data" / "cloud-v3"
A_VER, B_VER = "cdpo3s2", "cdpo2s1"


def ver_seed(fname):
    lab = re.sub(r"cut-pen-s\d+\.txt$", "", fname).replace("sp9-", "")
    lab = lab[:-1] if lab.endswith("r") else lab
    return lab, int(re.search(r"-s(\d+)\.txt$", fname).group(1))


def nchar(t):
    return len(re.sub(r"【[^】]*】|\s", "", t))


def main():
    global A_VER
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--maxdiff", type=float, default=None)
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=20261124)
    ap.add_argument("--a", default=A_VER, help="和 dpo2-s1 比的版本标签（默认 cdpo3s2；11-24 第四轮用 cdpo4s1）")
    arg = ap.parse_args()
    A_VER = arg.a
    out = Path(arg.out)
    out.mkdir(parents=True, exist_ok=True)
    rnd = random.Random(arg.seed if arg.maxdiff is None else arg.seed + 1)
    used = {k["card"] for f in arg.exclude for k in json.loads(Path(f).read_text(encoding="utf-8"))}
    exams = {"pub": load_exam(ROOT / "exam" / "考卷.jsonl"), "sp": load_exam(ROOT / "exam" / "专项-心里话.jsonl")}
    items, key = [], []
    for fn, ex, n in [("judge-r4.jsonl", "pub", 12), ("judge-sp9-r4.jsonl", "sp", 8)]:
        cell = {}
        for t in (V3 / fn).read_text(encoding="utf-8").splitlines():
            r = json.loads(t)
            v, s = ver_seed(r["file"])
            if v in (A_VER, B_VER) and r["lines"].strip():
                cell[(v, r["id"], s)] = r
        cands = [(c, s) for (v, c, s) in cell if v == A_VER and (B_VER, c, s) in cell
                 and cell[(A_VER, c, s)]["lines"].strip() != cell[(B_VER, c, s)]["lines"].strip()]
        if arg.maxdiff is not None:
            cands = [(c, s) for c, s in cands
                     if abs(nchar(cell[(A_VER, c, s)]["lines"]) - nchar(cell[(B_VER, c, s)]["lines"]))
                     < arg.maxdiff * max(nchar(cell[(A_VER, c, s)]["lines"]), nchar(cell[(B_VER, c, s)]["lines"]))]
        rnd.shuffle(cands)
        cands.sort(key=lambda cs: cs[0] in used)          # 没出过的卡排前面（稳定排序，组内仍随机）
        seen = set()
        for c, s in cands:
            if c in seen or len(seen) >= n:
                continue
            seen.add(c)
            x, y = cell[(A_VER, c, s)], cell[(B_VER, c, s)]
            flip = rnd.random() < 0.5
            left, right = (y, x) if flip else (x, y)
            qid = f"q{len(items) + 1:02d}"
            items.append({"id": qid, "card": sections(exams[ex][c]),
                          "a": left["lines"].strip().splitlines(), "b": right["lines"].strip().splitlines()})
            key.append({"id": qid, "exam": ex, "card": c, "seed": s,
                        "len": {A_VER: nchar(x["lines"]), B_VER: nchar(y["lines"])},
                        "A": B_VER if flip else A_VER, "B": A_VER if flip else B_VER,
                        "judge_budong": {B_VER if flip else A_VER: "语句不通" in left["problems"],
                                         A_VER if flip else B_VER: "语句不通" in right["problems"]}})
    order = list(range(len(items)))
    rnd.shuffle(order)                                   # 公开卷 / 专项卷混着出
    items = [items[i] for i in order]
    key = [key[i] for i in order]
    for i, (it, k) in enumerate(zip(items, key), 1):
        it["id"] = k["id"] = f"q{i:02d}"
    (out / "key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    data = json.dumps(items, ensure_ascii=False).replace("</", "<\\/")
    page = TEMPLATE.replace("/*ITEMS*/", data).replace("smooth-check-1", "smooth-check-" + out.name)
    (out / "哪条更顺.html").write_text(page, encoding="utf-8")
    a_left = sum(k["A"] == A_VER for k in key)
    print(f"{len(items)} 对 → {out / '哪条更顺.html'}；{A_VER} 在左边 {a_left} 次；"
          f"重复用到旧卡 {sum(k['card'] in used for k in key)} 对；"
          f"平均字数 {A_VER} {sum(k['len'][A_VER] for k in key) / len(key):.1f} / {B_VER} {sum(k['len'][B_VER] for k in key) / len(key):.1f}")


TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>哪条更顺</title>
<!-- 第三轮 DPO 盲测（2026-11-24），由 scripts/smooth_check_build.py 生成；离线可用，答案只存在本机浏览器，结尾给结果码。 -->
<style>
:root{--bg:#f4f4fa;--paper:#fff;--ink:#1f1d3a;--muted:#6a6888;--line:#dddcec;--brand:#3d3aa8;--brand-soft:#e7e6fb;--sun:#ffc83d;--sun-soft:#fff3cf;
  --f-body:-apple-system,"PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;--f-line:"Songti SC","STSong","Noto Serif SC",serif}
@media (prefers-color-scheme:dark){:root{--bg:#13122a;--paper:#1d1c38;--ink:#ecebf7;--muted:#a3a1c4;--line:#34325a;--brand:#9d9bff;--brand-soft:#2b2a5c;--sun:#ffd36a;--sun-soft:#3b3320;color-scheme:dark}}
*{box-sizing:border-box}html,body{margin:0}
body{background:var(--bg);color:var(--ink);font-family:var(--f-body);font-size:16px;line-height:1.65}
.wrap{max-width:760px;margin:0 auto;padding:20px 16px 80px;display:flex;flex-direction:column;gap:16px}
.bar{position:sticky;top:0;z-index:5;height:6px;background:var(--line)}.bar i{display:block;height:100%;background:var(--brand);width:0;transition:width .3s}
h1{font-size:1.6rem;margin:0}p{margin:0}.muted{color:var(--muted)}
.box{background:var(--paper);border:1px solid var(--line);border-radius:16px;padding:20px;display:flex;flex-direction:column;gap:14px}
.btn{font:700 1rem var(--f-body);padding:12px 22px;border-radius:999px;border:none;background:var(--brand);color:#fff;cursor:pointer}
.btn.ghost{background:transparent;color:var(--brand);border:1.5px solid var(--brand)}.btn:disabled{opacity:.4;cursor:default}
.row{display:flex;gap:10px;flex-wrap:wrap;align-items:center;justify-content:space-between}
.qhead{display:flex;justify-content:space-between;font-size:.85rem;color:var(--muted)}.qhead b{color:var(--brand);font-size:1rem}
.card{background:var(--bg);border-radius:12px;padding:12px 14px;display:grid;grid-template-columns:4.2em 1fr;gap:6px 10px;font-size:.92rem}
.card .k{color:var(--muted);font-weight:700}.card .v{white-space:pre-wrap;min-width:0}.card .v.goal{background:var(--sun-soft);border-radius:8px;padding:6px 10px}
@media (max-width:520px){.card{grid-template-columns:1fr;gap:0}.card .k{margin-top:8px}}
details summary{cursor:pointer;color:var(--brand);font-size:.88rem}details pre{white-space:pre-wrap;font:inherit;font-size:.88rem;color:var(--muted);margin:8px 0 0}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}@media (max-width:600px){.pair{grid-template-columns:1fr}}
.opt{border:2px solid var(--line);border-radius:14px;padding:12px 14px;background:var(--paper)}
.opt .tag{font-weight:700;color:var(--brand)}.opt .lines{font-family:var(--f-line);font-size:1.08rem;line-height:1.85;white-space:pre-wrap}
.ask{font-weight:700}.chips{display:flex;flex-wrap:wrap;gap:8px}
.chips button{font:500 1rem var(--f-body);padding:9px 18px;border-radius:999px;border:1.5px solid var(--line);background:var(--paper);color:var(--ink);cursor:pointer}
.chips button[aria-pressed="true"]{background:var(--brand);color:#fff;border-color:var(--brand)}
textarea{font:inherit;font-size:.92rem;width:100%;min-height:2.4em;border:1px solid var(--line);border-radius:10px;background:var(--bg);color:var(--ink);padding:8px 12px}
.code{font-family:ui-monospace,Menlo,monospace;font-size:.85rem;word-break:break-all;background:var(--bg);border:1px dashed var(--brand);border-radius:10px;padding:10px 12px;user-select:all}
button:focus-visible,textarea:focus-visible{outline:3px solid var(--sun);outline-offset:2px}
</style>
</head>
<body>
<div class="bar"><i id="bar"></i></div>
<div class="wrap" id="app"></div>
<script>
const ITEMS = /*ITEMS*/;
const KEY = "smooth-check-1";
let st = {i: -1, ans: {}, note: {}};
try { const s = JSON.parse(localStorage.getItem(KEY)); if (s && s.ans) st = s; } catch (e) {}
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(st)); } catch (e) {} };
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const app = document.getElementById("app");
const first = t => (t || "").split("\n")[0];
function card(c) {
  const sp = c["说话人"] || "", goal = (c["目的"] || "").split("\n").filter(l => l.startsWith("要办成")).join("\n");
  const rows = [["说话人", sp.split("\n").slice(0, 2).join("\n")], ["对方", first(c["对方"])], ["场景", c["场景"] || ""],
                ["前文", c["前文"] || "无"], ["目的", goal]];
  const full = Object.entries(c).map(([k, v]) => "【" + k + "】" + v).join("\n");
  return '<div class="card">' + rows.map(([k, v]) => '<div class="k">' + k + '</div><div class="v' + (k === "目的" ? " goal" : "") + '">' + esc(v) + '</div>').join("") +
    '</div><details><summary>看完整戏卡</summary><pre>' + esc(full) + '</pre></details>';
}
function render() {
  document.getElementById("bar").style.width = (Math.max(0, Object.keys(st.ans).length) / ITEMS.length * 100) + "%";
  if (st.i < 0) {
    app.innerHTML = '<div class="box"><h1>哪条更顺</h1><p>' + ITEMS.length + ' 对台词，每对是同一张卡、两个版本写的。<b>只判哪条读着更顺</b>：用词、搭配、语序自然，不用停下来想；' +
      '口气像不像、有没有说漏都不用管。分不出来就选「差不多」。</p><p class="muted">左右是随机的，页面里没有版本名。大约 10 分钟；答案存在这个浏览器里，中途关掉再打开能接着做。</p>' +
      '<div><button class="btn" id="go">开始</button></div></div>';
    document.getElementById("go").onclick = () => { st.i = 0; save(); render(); };
    return;
  }
  if (st.i >= ITEMS.length) return done();
  const it = ITEMS[st.i], a = st.ans[it.id] || "";
  app.innerHTML = '<div class="box"><div class="qhead"><b>第 ' + (st.i + 1) + ' / ' + ITEMS.length + ' 对</b><span>' + it.id + '</span></div>' + card(it.card) +
    '<div class="pair">' + ["a", "b"].map(s => '<div class="opt"><div class="tag">' + s.toUpperCase() + '</div><div class="lines">' + esc(it[s].join("\n")) + '</div></div>').join("") + '</div>' +
    '<p class="ask">哪条读着更顺？</p><div class="chips">' + [["A", "A 更顺"], ["B", "B 更顺"], ["=", "差不多"]].map(([v, t]) =>
      '<button data-v="' + v + '" aria-pressed="' + (a === v) + '">' + t + '</button>').join("") + '</div>' +
    '<textarea id="note" placeholder="（可不填）哪里不顺">' + esc(st.note[it.id] || "") + '</textarea>' +
    '<div class="row"><button class="btn ghost" id="prev"' + (st.i ? "" : " disabled") + '>上一对</button><button class="btn" id="next"' + (a ? "" : " disabled") + '>' +
    (st.i === ITEMS.length - 1 ? "看结果码" : "下一对") + '</button></div></div>';
  app.querySelectorAll(".chips button").forEach(b => b.onclick = () => { st.ans[it.id] = b.dataset.v; save(); render(); });
  document.getElementById("note").oninput = e => { st.note[it.id] = e.target.value; save(); };
  document.getElementById("prev").onclick = () => { st.i--; save(); render(); window.scrollTo(0, 0); };
  document.getElementById("next").onclick = () => { st.i++; save(); render(); window.scrollTo(0, 0); };
}
function done() {
  const code = "SMOOTH1|" + ITEMS.map(it => st.ans[it.id] || "-").join("") + "|" +
    JSON.stringify(Object.fromEntries(Object.entries(st.note).filter(([k, v]) => v.trim())));
  app.innerHTML = '<div class="box"><h1>做完了</h1><p>把下面这一行整段复制给 Claude：</p><div class="code">' + esc(code) + '</div>' +
    '<div class="row"><button class="btn ghost" id="back">回去改</button><button class="btn" id="copy">复制</button></div></div>';
  document.getElementById("back").onclick = () => { st.i = ITEMS.length - 1; save(); render(); };
  document.getElementById("copy").onclick = () => { try { navigator.clipboard.writeText(code); document.getElementById("copy").textContent = "已复制"; } catch (e) {} };
}
render();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
