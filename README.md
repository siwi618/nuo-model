# Nuo model（傩）

**Nuo** is an *actor model* for role-play / NPC dialogue: give it a character sheet and a scene, and it speaks in character.
傩戏戴上面具就成了那个人——Nuo 读完人物档案，就用这个人的性格和口气说话。

做法：微调现成的开源小模型，不从头训练；不绑定任何游戏。

## 现在到哪了（2026-11-22）

当前最好的模型是 **dpo2-s1**：Qwen3.5-2B 先 LoRA 微调（v3），再做两轮带约束的 DPO。4 位量化后在 8 GB 的 M2 上跑。

| | 心里话说出口 ↓ | 专项卷全过 ↑ | 公开卷 ↑ | 像不像（公开卷，1–5）↑ |
| --- | --- | --- | --- | --- |
| v3（只做 SFT） | 11.1% | 53.0% | 46.1 | 3.00 |
| dpoa-s1（带约束 DPO 一轮） | 4.4% | 55.9% | 50.6 | 3.07 |
| **dpo2-s1（再迭代一轮）** | **3.0%** | **61.5%** | **60.8** | **3.16** |

- 怎么考：
  - 专项卷：15 张专测「心里想的别说出口」的卡，每张 18 个采样种子。
  - 公开卷：60 张卡，每张 6 个采样种子，按难度加权。
  - DeepSeek 盲评，评审不知道是哪个版本。
- 显著性：dpo2-s1 比 v3 在说出口、公开卷、公开卷像不像三项上，按卡重抽的 95% 区间都不碰 0。
- 选模型：每种做法都训了 3–4 个种子，从中挑一个，再用没用过的采样种子重考确认。
- 怎么做到的、哪里失败过：见 [lab_note.md](lab_note.md) 和 [methodology.md](methodology.md)。


- 格式说明：[docs/戏卡格式.md](docs/戏卡格式.md)
- 情况覆盖表：[docs/情况覆盖表.md](docs/情况覆盖表.md)
- 示例（自编练习角色，不进训练）：[examples/](examples/)
- 考卷（永不训练）：[exam/](exam/)
- 方法与出处（哪些原创、哪些参考了哪篇论文）：[methodology.md](methodology.md)
- 实验记录：[lab_note.md](lab_note.md)
- 卡片渲染脚本：`python3 scripts/render_card.py examples/梅姨.jsonl`
