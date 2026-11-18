# Nuo model（傩）

**Nuo** is an *actor model* for role-play / NPC dialogue: give it a character sheet and a scene, and it speaks in character.
傩戏戴上面具就成了那个人——Nuo 读完人物档案，就用这个人的性格和口气说话。

做法：微调现成的开源小模型，不从头训练；不绑定任何游戏。

> 状态：设计阶段。目前只有输入 / 输出格式。

- 格式说明：[docs/戏卡格式.md](docs/戏卡格式.md)
- 情况覆盖表：[docs/情况覆盖表.md](docs/情况覆盖表.md)
- 示例（自编练习角色，不进训练）：[examples/](examples/)
- 考卷（永不训练）：[exam/](exam/)
- 卡片渲染脚本：`python3 scripts/render_card.py examples/梅姨.jsonl`
