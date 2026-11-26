"""演员模型的「游戏端」：调用接口 + 出口把关（2026-11-26，接游戏准备 B1）。

和引擎无关：游戏（Godot 或别的）只要能把戏卡（card dict，格式同考卷 / docs/戏卡格式.md）交给这里，
拿回整理好的台词列表；拿到 None 就用游戏里手写的台词（退回）。

两部分：
1. gate(card, raw) —— 出口把关，纯文本处理、不依赖模型，可单独用（也拿来量考试输出）：
   - 解析「名字[表情]: 话」，解析不了的行丢掉；
   - 从第一句「不在『谁能开口』名单里的人」开始整段截掉（模型会替对方接着写，11-25 小实验五五开）；
   - 截到最多句数；
   - 动作：卡上「不写」就删括号；「可以写」就把句首括号拆成 action 字段，句中括号删掉；
   - 表情不在清单里 → 平常；表情写进括号 → 删；
   - 超字数：在限内最后一个句末标点（没有就逗号）处截断，截不了的整句丢掉；
   - 后一句和前面某句有 ≥ 6 字一样 → 丢后一句（重复）；
   - **说漏检查**：台词里出现「别说漏」里 ≥ 5 字（「不能说」≥ 7 字）的片段、且这片段卡上公开部分没有（对外的说法不算）
     → 整段退回；「心里想的」≥ 8 字原话出现、且卡上公开部分没有 → 整段退回；
   - 整理完一句不剩 → 退回。
2. Actor —— 加载一次模型，按 NPC 缓存档案前缀（系统提示 + 【世界】 + 【说话人】，走近时可先 warm），
   say(card) 生成 → 把关 → 退回就换种子再试一次，还不行返回 None。
   解码参数 = 11-25 定的默认：温度 0.3、top_p 0.9、重复惩罚 1.15（只罚自己写的，不罚名字 / 表情 / 格式符号）。

用法：
  from actor import Actor, gate
  a = Actor("models/q35-2b-v3-dpo2-s1-mlx4")
  a.warm(card)                 # 可选：NPC 进入视野时
  res = a.say(card)            # -> GateResult；res.lines 为 None 表示退回
  命令行：.venv/bin/python scripts/actor.py gate 输出前缀 [--exam ...]   # 量考试输出过把关的情况
         .venv/bin/python scripts/actor.py demo 模型目录 [--exam ...] [--n 6]   # 本机实跑几张卡，看速度和缓存
"""

import argparse
import copy
import json
import re
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from render_card import render_card  # noqa: E402

LINE = re.compile(r"^(.+?)(?:\[(.+?)\])?[:：]\s*(.*)$")
PAREN = re.compile(r"[（(]([^）)]*)[）)]")
LEAD_ACT = re.compile(r"^\s*[（(]([^）)]*)[）)]\s*")
PUNCT = re.compile(r"[，。！？、；：,.!?;:“”\"'‘’（）()…—\s\[\]「」『』《》]")
END_MARK = "。！？!?…"
SOFT_MARK = "，、；,;"
LEAK_MIN = 5       # 「别说漏」：≥ 5 字片段
SECRET_MIN = 7     # 「不能说」（秘密，写得长、常带着要打听的人名，如「暗查大昌洋行经理」）：≥ 7 字
INNER_MIN = 8      # 「心里想的」：≥ 8 字原话
REPEAT_MIN = 6     # 两句 ≥ 6 字一样算重复
COVER = re.compile(r"[（(]?\s*(对外[^）)]*)[）)]?")   # 秘密里「（对外说：……）」是可以说的掩饰说法


@dataclass
class GateResult:
    lines: list | None                          # [{"name", "emotion", "action", "text"}]；None = 退回
    fixes: list = field(default_factory=list)   # 修过什么（给日志看）
    reject: str = ""                            # 退回原因
    raw: str = ""
    tries: int = 1
    seconds: float = 0.0
    first_token: float = 0.0

    def text(self):
        if not self.lines:
            return ""
        return "\n".join(f"{l['name']}" + (f"[{l['emotion']}]" if l["emotion"] else "") + ": "
                         + (f"（{l['action']}）" if l["action"] else "") + l["text"] for l in self.lines)


def _strip(s):
    return PUNCT.sub("", s or "")


def _as_list(x):
    if not x:
        return []
    return [str(i) for i in x] if isinstance(x, list) else [str(x)]


def rules_of(card):
    r = card["rules"]
    lo, hi = (r.get("lines") or [1, 9])[:2]
    return {"lo": lo, "hi": hi, "max_chars": r.get("max_chars") or 999, "actions": bool(r.get("actions")),
            "emotions": list(r.get("emotions") or []), "speakers": list(r.get("speakers") or [])}


def _public_text(card):
    """卡上「说出来也不算漏」的部分：去掉秘密、心里想的、别说漏之后的整张卡。"""
    c = copy.deepcopy(card)
    sp = c.get("speaker", {})
    covers = [m.group(1) for s in _as_list(sp.get("secrets")) for m in COVER.finditer(s)]
    sp.pop("secrets", None)
    g = c.get("goal", {})
    g.pop("inner", None)
    g.pop("dont_leak", None)
    return _strip(json.dumps(c, ensure_ascii=False) + "".join(covers))


def _own_fragment(secret, text, public, n):
    """secret 和 text 共有、且 public 里没有的最长片段（≥ n 字才返回）。"""
    best = ""
    for i in range(len(secret)):
        j = i + max(n, len(best) + 1)
        while j <= len(secret) and secret[i:j] in text:
            if secret[i:j] not in public:
                best = secret[i:j]
            j += 1
    return best if len(best) >= n else ""


def leak_check(card, lines):
    """返回退回原因（空 = 没说漏）。只量说话人自己的台词（不含动作）。"""
    text = _strip("".join(l["text"] for l in lines))
    public = _public_text(card)
    sp, g = card.get("speaker", {}), card.get("goal", {})
    items = [(s, SECRET_MIN) for s in _as_list(sp.get("secrets"))] + [(s, LEAK_MIN) for s in _as_list(g.get("dont_leak"))]
    for s, n in items:
        s = _strip(COVER.sub("", s))
        frag = _own_fragment(s, text, public, n)
        if frag:
            return f"说漏：「{frag}」"
    frag = _own_fragment(_strip(g.get("inner")), text, public, INNER_MIN)
    if frag:
        return f"心里话说出口：「{frag}」"
    return ""


def _cut_chars(text, limit):
    if len(text) <= limit:
        return text
    head = text[:limit]
    for marks in (END_MARK, SOFT_MARK):
        k = max(head.rfind(m) for m in marks)
        if k >= limit // 3:
            return head[:k + 1] if head[k] in END_MARK else head[:k] + "。"
    return ""


def _longest_common(a, b):
    best = 0
    for i in range(len(a)):
        j = i + best + 1
        while j <= len(a) and a[i:j] in b:
            best = j - i
            j += 1
    return best


def gate(card, raw):
    r = rules_of(card)
    res = GateResult(lines=None, raw=raw)
    out = []
    for s in (raw or "").splitlines():
        s = s.strip()
        if not s:
            continue
        m = LINE.match(s)
        if not m:
            res.fixes.append(f"丢掉解析不了的行：{s[:12]}")
            continue
        name, emo, text = m.group(1).strip(), (m.group(2) or "").strip() or None, m.group(3).strip()
        if name not in r["speakers"]:
            res.fixes.append(f"从「{name[:6]}」开口处截掉")
            break
        action = None
        if r["actions"]:
            a = LEAD_ACT.match(text)
            if a and a.group(1).strip() not in r["emotions"]:
                action, text = a.group(1).strip(), text[a.end():]
        n0 = len(text)
        text = PAREN.sub("", text).strip()
        if len(text) != n0:
            res.fixes.append("删括号")
        if emo and emo not in r["emotions"]:
            res.fixes.append(f"表情「{emo}」不在清单 → 平常")
            emo = None
        if emo == "平常":
            emo = None
        if len(text) > r["max_chars"]:
            cut = _cut_chars(text, r["max_chars"])
            res.fixes.append("超字数截断" if cut else "超字数整句丢掉")
            text = cut
        if not text:
            continue
        if any(_longest_common(_strip(text), _strip(o["text"])) >= REPEAT_MIN for o in out):
            res.fixes.append("丢掉重复句")
            continue
        out.append({"name": name, "emotion": emo, "action": action, "text": text})
        if len(out) >= r["hi"]:
            if len([x for x in raw.splitlines() if x.strip()]) > len(out):
                res.fixes.append(f"截到 {r['hi']} 句")
            break
    if not out:
        res.reject = "整理后没有台词"
        return res
    why = leak_check(card, out)
    if why:
        res.reject = why
        return res
    res.lines = out
    return res


# ---------------------------------------------------------------- 模型部分（mlx，只在 Actor 里用）

CUT = "\n\n【对方】"


class Actor:
    def __init__(self, model_path, temp=0.3, top_p=0.9, rep_penalty=1.15, rep_context=64, max_tokens=300,
                 cache_npcs=8, retries=1):
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.sample_utils import make_sampler
        from pilot import SYS_LINES
        self.mx, self.SYS = mx, SYS_LINES
        self.model, self.tok = load(model_path)
        self.sampler = make_sampler(temp=temp, top_p=top_p)
        self.rep_penalty, self.rep_context, self.max_tokens = rep_penalty, rep_context, max_tokens
        self.cache = OrderedDict()        # 前缀文字 → (前缀 token, KV 缓存)
        self.cache_npcs, self.retries = cache_npcs, retries

    def _ids(self, card):
        msgs = [{"role": "system", "content": self.SYS}, {"role": "user", "content": render_card(card)}]
        prompt = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        ids = self.tok.encode(prompt, add_special_tokens=False)
        key = prompt[: prompt.index(CUT)] if CUT in prompt else prompt
        pre = self.tok.encode(key, add_special_tokens=False)
        n = 0
        while n < min(len(pre), len(ids) - 1) and pre[n] == ids[n]:
            n += 1
        return ids, key, n

    def warm(self, card):
        """NPC 进入视野时先把档案前缀读进缓存（这段时间玩家不用等）。返回用时。"""
        from mlx_lm.models.cache import make_prompt_cache
        ids, key, n = self._ids(card)
        if key in self.cache:
            self.cache.move_to_end(key)
            return 0.0
        t = time.perf_counter()
        c = make_prompt_cache(self.model)
        self.model(self.mx.array(ids[:n])[None], cache=c)
        self.mx.eval([x.state for x in c])
        self.cache[key] = (n, c)
        while len(self.cache) > self.cache_npcs:
            self.cache.popitem(last=False)
        return time.perf_counter() - t

    def _penalty(self, prompt_len, card):
        mx, tok, r = self.mx, self.tok, card["rules"]
        words = list(r.get("speakers", [])) + list(r.get("emotions", [])) + ["[", "]", ":", "：", "\n", "（", "）"]
        exempt = set()
        for w in words:
            for pre in ("", "\n"):
                exempt.update(tok.encode(pre + w, add_special_tokens=False))
        exempt = mx.array(sorted(exempt))
        penalty, context = self.rep_penalty, self.rep_context

        def proc(tokens, logits):
            gen = tokens[prompt_len:][-context:]
            if gen.size == 0:
                return logits
            hit = mx.zeros((logits.shape[-1],), dtype=mx.bool_)
            hit[gen] = True
            hit[exempt] = False
            pen = mx.where(logits < 0, logits * penalty, logits / penalty)
            return mx.where(hit[None, :], pen, logits)
        return proc

    def generate(self, card, seed=0):
        from mlx_lm import stream_generate
        ids, key, n = self._ids(card)
        self.warm(card)
        n, base = self.cache[key]
        cache = copy.deepcopy(base)
        rest = ids[n:]
        self.mx.random.seed(seed)
        t, first, text = time.perf_counter(), None, ""
        procs = [self._penalty(len(rest), card)] if self.rep_penalty and self.rep_penalty != 1.0 else None
        for r in stream_generate(self.model, self.tok, rest, max_tokens=self.max_tokens, sampler=self.sampler,
                                 logits_processors=procs, prompt_cache=cache):
            if first is None:
                first = time.perf_counter() - t
            text += r.text
            # 已经写满最多句数、开始写下一行就停（省时间，游戏也用不上）
            if text.count("\n") >= card["rules"]["lines"][1] and text.endswith("\n"):
                break
        return text.strip(), first or 0.0, time.perf_counter() - t

    def say(self, card, seed=0):
        total, res = 0.0, None
        for k in range(self.retries + 1):
            raw, first, sec = self.generate(card, seed + k)
            total += sec
            res = gate(card, raw)
            res.tries, res.seconds = k + 1, total
            res.first_token = res.first_token or first
            if res.lines:
                break
        return res


# ---------------------------------------------------------------- 命令行

def _load_exam(path):
    return {json.loads(t)["id"]: json.loads(t) for t in Path(path).read_text(encoding="utf-8").splitlines() if t.strip()}


def cmd_gate(a):
    from collections import Counter
    from exam_stats import expand
    exam = _load_exam(a.exam)
    tot, clean, fixed, rej = 0, 0, 0, []
    kinds = Counter()
    for p in expand(a.prefix):
        cur, buf, blocks = None, [], {}
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            m = re.match(r"=== (\S+?)（", line)
            if m:
                cur = m.group(1)
                blocks[cur] = []
            elif cur:
                blocks[cur].append(line)
        for i, ls in blocks.items():
            if i not in exam:
                continue
            res = gate(exam[i]["card"], "\n".join(ls))
            tot += 1
            for f in res.fixes:
                kinds[re.sub(r"[：「].*", "", f)] += 1
            if not res.lines:
                rej.append((Path(p).name, i, res.reject, "\n".join(x for x in ls if x.strip())))
            elif res.fixes:
                fixed += 1
            else:
                clean += 1
    print(f"{a.prefix}：{tot} 段 | 原样通过 {clean}（{clean / tot:.0%}）| 修过 {fixed}（{fixed / tot:.0%}）| 退回 {len(rej)}（{len(rej) / tot:.0%}）")
    print("  修了什么：", dict(kinds.most_common()))
    for f, i, why, raw in rej:
        print(f"  ✗ {f} {i}：{why}")
        if a.verbose:
            print("     " + raw.replace("\n", "\n     "))


def cmd_demo(a):
    exam = list(_load_exam(a.exam).values())[:: max(1, 60 // a.n)][: a.n]
    t = time.perf_counter()
    act = Actor(a.model)
    print(f"加载 {time.perf_counter() - t:.1f}s")
    act.say(exam[0]["card"])                     # 热身（编译内核）
    for e in exam:
        act.cache.clear()
        w = act.warm(e["card"])
        res = act.say(e["card"], seed=0)
        print(f"== {e['id']} 预读 {w:.2f}s | 首字 {res.first_token:.2f}s | 整段 {res.seconds:.2f}s | 试 {res.tries} 次"
              + (f" | 退回：{res.reject}" if not res.lines else "") + (f" | 修：{'；'.join(res.fixes)}" if res.fixes else ""))
        print("   " + (res.text() or res.raw).replace("\n", "\n   "))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gate")
    g.add_argument("prefix")
    g.add_argument("--exam", default=str(HERE.parent / "exam" / "考卷.jsonl"))
    g.add_argument("-v", "--verbose", action="store_true")
    d = sub.add_parser("demo")
    d.add_argument("model")
    d.add_argument("--exam", default=str(HERE.parent / "exam" / "考卷.jsonl"))
    d.add_argument("--n", type=int, default=6)
    a = ap.parse_args()
    {"gate": cmd_gate, "demo": cmd_demo}[a.cmd](a)


if __name__ == "__main__":
    main()
