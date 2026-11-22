"""专项卷（心里话）统计：每版本说出口率（按被问到边上 / 没问分）、办成、全过、像不像；同卡同种子配对 + 按卡重抽 95% 区间。

用法（仓库根目录）：.venv/bin/python scripts/sp_stats.py 评审结果.jsonl 基准标签 标签1,标签2,...
  标签 = 文件名去掉 sp9- 前缀和 cut-pen-sN 后缀，例：c2bh c3,c3n
  末尾固定比较的几对（v3n − v3 等）是 2026-11-21 的；换实验时改最后那个列表。
"""
import json, random, sys, collections
J = sys.argv[1]; base = sys.argv[2]; labs = sys.argv[3].split(',')
exam = {}
for l in open('exam/专项-心里话.jsonl'):
    c = json.loads(l); exam[c['id']] = c.get('meta', {}).get('pressed')
R = collections.defaultdict(dict)  # lab -> (id,seed) -> rec
for l in open(J):
    x = json.loads(l); lab = x['file'].rsplit('-s', 1)[0].replace('sp9-', '').replace('cut-pen', '')
    seed = x['file'].rsplit('-s', 1)[1].split('.')[0]
    R[lab][(x['id'], seed)] = x
leak = lambda x: '心里话说出口' in x['problems']
def show(lab):
    d = R[lab]; n = len(d)
    lk = sum(leak(x) for x in d.values())
    pr = [x for k, x in d.items() if exam[k[0]]]; npr = [x for k, x in d.items() if not exam[k[0]]]
    bc = collections.Counter(x['办成'] for x in d.values())
    ok = sum(x['pass'] for x in d.values()); lk_secret = sum('说漏秘密' in x['problems'] for x in d.values())
    print(f"{lab:8s} n={n} 说出口 {lk}/{n} ({100*lk/n:.0f}%)  被问到边上 {sum(map(leak,pr))}/{len(pr)}  没问 {sum(map(leak,npr))}/{len(npr)}  "
          f"办成2/1/0 {bc[2]}/{bc[1]}/{bc[0]}  全过 {ok} ({100*ok/n:.0f}%)  像不像 {sum(x['像不像'] for x in d.values())/n:.2f}  说漏秘密 {lk_secret}")
def paired(a, b, key=leak):
    cards = sorted({k[0] for k in R[a]} & {k[0] for k in R[b]})
    seeds = sorted({k[1] for k in R[a]} & {k[1] for k in R[b]}, key=int)
    def diff(cs):
        s = 0; m = 0
        for c in cs:
            for sd in seeds:   # 两边共有的种子（11-22 起重考用 9–17，不再写死 0–8）
                if (c, sd) in R[a] and (c, sd) in R[b]:
                    s += key(R[a][(c, sd)]) - key(R[b][(c, sd)]); m += 1
        return 100 * s / m
    d0 = diff(cards); rnd = random.Random(0)
    bs = sorted(diff([rnd.choice(cards) for _ in cards]) for _ in range(4000))
    return d0, bs[100], bs[3899]
for lab in [base] + labs: show(lab)
print('\n配对（同卡同种子，按卡重抽 95% 区间）：说出口率差 / 办成=0 率差 / 全过率差')
for a, b in [(l, base) for l in labs] + [tuple(p.split('-')) for p in (sys.argv[4].split(',') if len(sys.argv) > 4 else [])]:
    r1 = paired(a, b); r2 = paired(a, b, lambda x: x['办成'] == 0); r3 = paired(a, b, lambda x: x['pass'])
    print(f"  {a:6s} − {b:7s}: 说出口 {r1[0]:+.1f} [{r1[1]:+.0f} ~ {r1[2]:+.0f}]   没办成 {r2[0]:+.1f} [{r2[1]:+.0f} ~ {r2[2]:+.0f}]   全过 {r3[0]:+.1f} [{r3[1]:+.0f} ~ {r3[2]:+.0f}]")
