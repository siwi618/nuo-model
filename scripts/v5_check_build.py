"""评审 v5.1 人工核对（2026-11-24）：40 条台词（dpo2-s1、dpo3-s2 各 20），Ken 逐句标毛病，和评审按句比。

抽样：公开卷 + 专项卷里这两个版本评过的台词（data/cloud-v3/judge-r4.jsonl、judge-sp9-r4.jsonl 的 lines），
     每张卡最多 1 条；优先用 Ken 之前几次核对都没见过的卡（--exclude 给的 key），不够再用见过的。
句子编号和评审 v5 完全一样（judge_exam.number_lines），按句比才对得上。
页面盲：只有卡和台词，没有版本名、没有评审判断；key.json 只在本地。
Ken 每句点「轻 / 重」（不点 = 没问题），可选类型；做完给结果码 V5CHK1|…，交给 v5_check_score.py。

用法（仓库根目录）：
  .venv/bin/python scripts/v5_check_build.py data/human-check/v5 --exclude data/human-check/key.json data/human-check/smooth/key.json data/human-check/smooth-eqlen/key.json
然后评审（和页面同一批台词）：.venv/bin/python scripts/v5_check_build.py data/human-check/v5 --judge
"""
import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from human_check_build import load_exam, sections  # noqa: E402
from judge_exam import number_lines  # noqa: E402
from smooth_check_build import ver_seed  # noqa: E402

ROOT = HERE.parent
V3 = ROOT / "data" / "cloud-v3"
VERS = ["cdpo2s1", "cdpo3s2"]
N_PER = 20


def used_cards(paths):
    out = set()
    for p in paths:
        k = json.loads(Path(p).read_text(encoding="utf-8"))
        for x in (k.values() if isinstance(k, dict) else k):
            out.add(x.get("card") or x.get("card_id"))
    out.discard(None)
    return out


def build(out, exclude, seed):
    rnd = random.Random(seed)
    exams = {"pub": load_exam(ROOT / "exam" / "考卷.jsonl"), "sp": load_exam(ROOT / "exam" / "专项-心里话.jsonl")}
    used = used_cards(exclude)
    pool = []
    for fn, ex in [("judge-r4.jsonl", "pub"), ("judge-sp9-r4.jsonl", "sp")]:
        for t in (V3 / fn).read_text(encoding="utf-8").splitlines():
            r = json.loads(t)
            v, s = ver_seed(r["file"])
            if v in VERS and r["lines"].strip():
                pool.append({"ver": v, "seed": s, "exam": ex, "card": r["id"], "lines": r["lines"].strip().splitlines()})
    rnd.shuffle(pool)
    pool.sort(key=lambda x: x["card"] in used)        # 没见过的卡排前面，组内仍随机
    picked, cards, n = [], set(), {v: 0 for v in VERS}
    for x in pool:
        if x["card"] in cards or n[x["ver"]] >= N_PER:
            continue
        cards.add(x["card"])
        n[x["ver"]] += 1
        picked.append(x)
    rnd.shuffle(picked)
    items, key = [], []
    for i, x in enumerate(picked, 1):
        qid = f"q{i:02d}"
        _, sents = number_lines(x["lines"])
        rows, k = [], 0                                # 每行：说话人前缀 + 本行的句子（全局编号）
        for ln in x["lines"]:
            _, one = number_lines([ln])
            head = ln.split(":", 1)[0] if ":" in ln else ""
            rows.append({"who": head.strip(), "s": [[k + j + 1, t] for j, t in enumerate(one)]})
            k += len(one)
        assert k == len(sents)
        items.append({"id": qid, "card": sections(exams[x["exam"]][x["card"]]), "rows": rows, "n": len(sents)})
        key.append({"id": qid, **{kk: x[kk] for kk in ("ver", "seed", "exam", "card", "lines")}, "n": len(sents),
                    "seen_before": x["card"] in used})
    (out / "key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    page = TEMPLATE.replace("/*ITEMS*/", json.dumps(items, ensure_ascii=False).replace("</", "<\\/"))
    (out / "逐句核对.html").write_text(page, encoding="utf-8")
    print(f"{len(items)} 条（{n}）→ {out / '逐句核对.html'}；共 {sum(i['n'] for i in items)} 句；"
          f"用到见过的卡 {sum(k['seen_before'] for k in key)} 条；公开卷 {sum(k['exam'] == 'pub' for k in key)} / 专项卷 {sum(k['exam'] == 'sp' for k in key)}")


def judge(out):
    from judge_exam import judge_one
    key = json.loads((out / "key.json").read_text(encoding="utf-8"))
    exams = {"pub": load_exam(ROOT / "exam" / "考卷.jsonl"), "sp": load_exam(ROOT / "exam" / "专项-心里话.jsonl")}
    use = {"in": 0, "out": 0}

    def one(k):
        r, u = judge_one(exams[k["exam"]][k["card"]], k["lines"], "v5")
        return {"id": k["id"], **r}, u
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(one, key))
    for _, u in res:
        use["in"] += u.get("prompt_tokens", 0)
        use["out"] += u.get("completion_tokens", 0)
    (out / "judge-v52.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r, _ in res), encoding="utf-8")
    print(f"评完 {len(res)} 条，token 入 {use['in']} 出 {use['out']}")


TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>逐句核对</title>
<!-- 评审 v5.1 人工核对（2026-11-24），由 scripts/v5_check_build.py 生成；离线可用，答案只存在本机浏览器，结尾给结果码。 -->
<style>
:root{--bg:#f4f4fa;--paper:#fff;--ink:#1f1d3a;--muted:#6a6888;--line:#dddcec;--brand:#3d3aa8;--brand-soft:#e7e6fb;--sun:#ffc83d;--sun-soft:#fff3cf;--warn:#c2410c;--warn-soft:#ffedd5;--bad:#b91c1c;--bad-soft:#fee2e2;
  --f-body:-apple-system,"PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;--f-line:"Songti SC","STSong","Noto Serif SC",serif}
@media (prefers-color-scheme:dark){:root{--bg:#13122a;--paper:#1d1c38;--ink:#ecebf7;--muted:#a3a1c4;--line:#34325a;--brand:#9d9bff;--brand-soft:#2b2a5c;--sun:#ffd36a;--sun-soft:#3b3320;--warn:#fdba74;--warn-soft:#3a2614;--bad:#fca5a5;--bad-soft:#3b1717;color-scheme:dark}}
*{box-sizing:border-box}html,body{margin:0}
body{background:var(--bg);color:var(--ink);font-family:var(--f-body);font-size:16px;line-height:1.65}
.wrap{max-width:760px;margin:0 auto;padding:20px 16px 80px;display:flex;flex-direction:column;gap:16px}
.bar{position:sticky;top:0;z-index:5;height:6px;background:var(--line)}.bar i{display:block;height:100%;background:var(--brand);width:0;transition:width .3s}
h1{font-size:1.6rem;margin:0}p{margin:0}.muted{color:var(--muted)}ul{margin:0;padding-left:1.2em}
.box{background:var(--paper);border:1px solid var(--line);border-radius:16px;padding:20px;display:flex;flex-direction:column;gap:14px}
.btn{font:700 1rem var(--f-body);padding:12px 22px;border-radius:999px;border:none;background:var(--brand);color:#fff;cursor:pointer}
.btn.ghost{background:transparent;color:var(--brand);border:1.5px solid var(--brand)}.btn:disabled{opacity:.4;cursor:default}
.row{display:flex;gap:10px;flex-wrap:wrap;align-items:center;justify-content:space-between}
.qhead{display:flex;justify-content:space-between;font-size:.85rem;color:var(--muted)}.qhead b{color:var(--brand);font-size:1rem}
.card{background:var(--bg);border-radius:12px;padding:12px 14px;display:grid;grid-template-columns:4.2em 1fr;gap:6px 10px;font-size:.92rem}
.card .k{color:var(--muted);font-weight:700}.card .v{white-space:pre-wrap;min-width:0}.card .v.goal{background:var(--sun-soft);border-radius:8px;padding:6px 10px}
@media (max-width:520px){.card{grid-template-columns:1fr;gap:0}.card .k{margin-top:8px}}
details summary{cursor:pointer;color:var(--brand);font-size:.88rem}details pre{white-space:pre-wrap;font:inherit;font-size:.88rem;color:var(--muted);margin:8px 0 0}
.who{font-weight:700;color:var(--brand);margin-top:4px}
.sent{border:2px solid var(--line);border-radius:12px;padding:10px 12px;display:flex;flex-direction:column;gap:8px}
.sent.s1{border-color:var(--warn);background:var(--warn-soft)}.sent.s2{border-color:var(--bad);background:var(--bad-soft)}
.sent .t{font-family:var(--f-line);font-size:1.08rem;line-height:1.8}.sent .t .no{font-family:var(--f-body);font-size:.8rem;color:var(--muted);margin-right:6px}
.chips{display:flex;flex-wrap:wrap;gap:6px}
.chips button{font:500 .92rem var(--f-body);padding:6px 14px;border-radius:999px;border:1.5px solid var(--line);background:var(--paper);color:var(--ink);cursor:pointer}
.chips button[aria-pressed="true"]{background:var(--brand);color:#fff;border-color:var(--brand)}
.chips.types button{font-size:.82rem;padding:4px 10px}
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
const KEY = "v5-check-1";
const TYPES = ["语句不通", "用词难懂", "逻辑不对", "不合人设", "其他"];
let st = {i: -1, m: {}, note: {}, seen: {}};
try { const s = JSON.parse(localStorage.getItem(KEY)); if (s && s.m) st = s; } catch (e) {}
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(st)); } catch (e) {} };
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const app = document.getElementById("app");
const first = t => (t || "").split("\n")[0];
function card(c) {
  const sp = c["说话人"] || "", goal = (c["目的"] || "").split("\n").filter(l => l.startsWith("要办成")).join("\n");
  const rows = [["说话人", sp.split("\n").slice(0, 4).join("\n")], ["对方", first(c["对方"])], ["场景", c["场景"] || ""],
                ["前文", c["前文"] || "无"], ["目的", goal]];
  const full = Object.entries(c).map(([k, v]) => "【" + k + "】" + v).join("\n");
  return '<div class="card">' + rows.map(([k, v]) => '<div class="k">' + k + '</div><div class="v' + (k === "目的" ? " goal" : "") + '">' + esc(v) + '</div>').join("") +
    '</div><details><summary>看完整戏卡（人物档案、世界）</summary><pre>' + esc(full) + '</pre></details>';
}
function mk(id, no) { return id + ":" + no; }
function render() {
  document.getElementById("bar").style.width = (Object.keys(st.seen).length / ITEMS.length * 100) + "%";
  if (st.i < 0) {
    app.innerHTML = '<div class="box"><h1>逐句核对</h1><p>' + ITEMS.length + ' 条台词，每条已经按句编号。请<b>逐句</b>看，有毛病的句子点「轻」或「重」，没问题的不用点。</p>' +
      '<ul><li><b>轻</b>：会注意到、停一下，但意思能懂。</li><li><b>重</b>：读不懂、明显错、让人出戏，或者人设不对。</li>' +
      '<li>点了轻 / 重以后可以选类型（不选也行）：语句不通（读着不顺、要猜意思）、用词难懂（生词、看不懂的称呼）、逻辑不对（和常识或前文对不上、矛盾）、不合人设（不像这个人会说的）。</li></ul>' +
      '<p class="muted">按你自己读的感觉判，口语、客气话这些你觉得可以就不算。页面里没有版本名、没有评审的判断。大约 20–25 分钟；答案存在这个浏览器里，中途关掉再打开能接着做。</p>' +
      '<div><button class="btn" id="go">开始</button></div></div>';
    document.getElementById("go").onclick = () => { st.i = 0; save(); render(); };
    return;
  }
  if (st.i >= ITEMS.length) return done();
  const it = ITEMS[st.i];
  st.seen[it.id] = 1; save();
  let html = '<div class="box"><div class="qhead"><b>第 ' + (st.i + 1) + ' / ' + ITEMS.length + ' 条</b><span>' + it.id + '</span></div>' + card(it.card);
  for (const r of it.rows) {
    if (r.who) html += '<div class="who">' + esc(r.who) + '</div>';
    for (const [no, t] of r.s) {
      const m = st.m[mk(it.id, no)] || {};
      html += '<div class="sent' + (m.sev ? " s" + m.sev : "") + '"><div class="t"><span class="no">' + no + '</span>' + esc(t) + '</div>' +
        '<div class="chips" data-no="' + no + '">' + [[1, "轻"], [2, "重"]].map(([v, l]) =>
          '<button data-sev="' + v + '" aria-pressed="' + (m.sev === v) + '">' + l + '</button>').join("") + '</div>' +
        (m.sev ? '<div class="chips types" data-no="' + no + '">' + TYPES.map(ty =>
          '<button data-ty="' + ty + '" aria-pressed="' + ((m.ty || []).includes(ty)) + '">' + ty + '</button>').join("") + '</div>' : "") + '</div>';
    }
  }
  html += '<textarea id="note" placeholder="（可不填）哪里不对、为什么">' + esc(st.note[it.id] || "") + '</textarea>' +
    '<div class="row"><button class="btn ghost" id="prev"' + (st.i ? "" : " disabled") + '>上一条</button><button class="btn" id="next">' +
    (st.i === ITEMS.length - 1 ? "看结果码" : "下一条") + '</button></div></div>';
  app.innerHTML = html;
  app.querySelectorAll("button[data-sev]").forEach(b => b.onclick = () => {
    const k = mk(it.id, +b.parentNode.dataset.no), v = +b.dataset.sev, m = st.m[k] || {};
    if (m.sev === v) delete st.m[k]; else st.m[k] = {sev: v, ty: m.ty || []};
    save(); render();
  });
  app.querySelectorAll("button[data-ty]").forEach(b => b.onclick = () => {
    const k = mk(it.id, +b.parentNode.dataset.no), m = st.m[k], ty = b.dataset.ty;
    m.ty = m.ty.includes(ty) ? m.ty.filter(x => x !== ty) : m.ty.concat([ty]);
    save(); render();
  });
  document.getElementById("note").oninput = e => { st.note[it.id] = e.target.value; save(); };
  document.getElementById("prev").onclick = () => { st.i--; save(); render(); window.scrollTo(0, 0); };
  document.getElementById("next").onclick = () => { st.i++; save(); render(); window.scrollTo(0, 0); };
}
function done() {
  const marks = {};
  for (const [k, m] of Object.entries(st.m)) marks[k] = m.sev + (m.ty || []).map(t => TYPES.indexOf(t)).join("");
  const notes = Object.fromEntries(Object.entries(st.note).filter(([k, v]) => v.trim()));
  const code = "V5CHK1|" + Object.keys(st.seen).length + "|" + JSON.stringify(marks) + "|" + JSON.stringify(notes);
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
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=20261124)
    ap.add_argument("--judge", action="store_true", help="对已生成的 key.json 跑评审 v5.2（旧结果 judge-v51.jsonl 保留）")
    a = ap.parse_args()
    o = Path(a.out)
    o.mkdir(parents=True, exist_ok=True)
    judge(o) if a.judge else build(o, a.exclude, a.seed)
