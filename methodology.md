# 方法与出处（methodology）

Nuo 用到的方法逐条列出：哪些是本项目原创，哪些借用或参考了已有工作。实验过程和数字见 [lab_note.md](lab_note.md)。

> **标注说明**
> - **原创**：做了中英文检索（约 12 次，没查 Google Scholar / 知网），没找到有人这样做。**找不到不等于没人做**。
> - **参考**：做法来自所列论文，或者和它们是同一类思路。
> - **独立，有先例**：我们先自己想出来、做了，后来读文献（2026-11-21）才发现别人有类似做法。
> - **工程**：常规做法或工具用法，不算方法贡献。
>
> 论文标题和链接 2026-11-22 在 arXiv / ACL Anthology 逐条核对过。论文内容多数只读了摘要或转述，引用其中的数字前要回原文确认。

## 1. 任务与输入格式

| 方法 | 我们怎么做 | 性质 | 出处 / 相关工作 |
| --- | --- | --- | --- |
| 戏卡（[docs/戏卡格式.md](docs/戏卡格式.md)） | 一张卡写说话人档案、对方、世界、场景，以及【目的】：心里想的、要办成、别说漏。一个模型按卡演所有角色，不用每个角色一个模型 | 格式原创 | 每个 NPC 一个 LoRA 的做法见 Fixed-Persona SLM [17] |
| 「心里话」当输入的潜台词任务 | 把心里想的告诉模型，台词要办成事，但不能把心思直接说出来；以「心里话说出口」率为主要指标 | **原创**（没找到同样的生成任务和外泄率指标） | 相关工作的「想法」是输出，不是输入：CoSER [13] 要求写出「[想法]台词」，Thinking in Character [14] 先推理再回答，RoleThink [15] 要猜角色在想什么。Beneath the Surface [19] 只评测潜台词、不训练；SOTOPIA [18] 同时评目标达成和保密 |

## 2. 训练数据

| 方法 | 我们怎么做 | 性质 | 出处 / 相关工作 |
| --- | --- | --- | --- |
| 强模型写、再过滤 | DeepSeek 按卡写台词 → Claude 判留 / 不留 → 小模型学 | 参考 | 用大模型合成训练数据：Self-Instruct [9]；少而精的数据胜过多而杂：LIMA [10]；角色多样性比条数重要：OpenCharacter [16] |
| 不合格就从头重写 | 不合格的不改，只换个说法重写一遍，也不告诉它上一版错在哪 | 独立，有先例 | OpenCharacter [16]：重新生成优于改写旧回答 |
| 生成端约束 | 机器查「心里想的」和台词字面是否太像、要办成和别说漏是否打架；补齐少的世界 / 角色；去掉原话照搬的样本 | 工程 | — |
| 古典小说构造训练对 | 用叙述里写明的心理当「心里想的」，人物嘴上说的原话当答案；标注时先判断心思属于谁 | **原创**（构造方法） | 从小说抽想法和台词：CoSER [13]、RoleThink [15]；中文小说角色数据：BEYOND DIALOGUE [12] |
| 「要办成」按原话实际办到的那一步重写 | 小说原话常常只走了长场景里的一步，卡上「要办成」写的却是整件事 → 改写成原话实际办到的那一步 | **原创**（发现了「藏得住但办不成」和目标写多粗有关） | — |
| 训练样本体检 | 用评考卷的同一套评审标准评训练样本本身（`scripts/judge_train.py`），找数据里的系统偏差 | 原创做法 | — |

## 3. 训练

| 方法 | 我们怎么做 | 性质 | 出处 / 相关工作 |
| --- | --- | --- | --- |
| LoRA 微调 | Qwen3.5-2B，全部层、r16、scale 20、lr 2e-5、2 轮 | 参考 | LoRA [1] |
| 按验证损失挑检查点 | 每 100 步存一份、算一次验证损失，取最低的 | 参考 | 提前停与重训波动：Dodge et al. [7] |
| 合并 + 4 位量化 | 云上合并 → mlx-lm 量化成 4 位，在 8 GB M2 上推理 | 工程 | — |
| DPO | 从 v3 出发，在合并后的模型上新加一个 LoRA；手写训练循环（`scripts/cloud/train_dpo.py`），β 0.1、lr 5e-6；最终配方加普通损失（`--sft-alpha 1.0`）、只训 1 轮 | 参考 | DPO [2] |
| 偏好对怎么来 | 模型对训练卡写 4 遍 → 外部评审盲评 → 「全过且办成」当好例子、「说出口 / 没办成」当坏例子 | 参考 | 模型输出 + 评审造偏好对：Self-Rewarding [8]（它是自己评自己，我们用外部评审）；Thinking in Character [14] 也用正负对 |
| 用原答案当好例子 | 模型写不出好例子时，用通过评审的原答案顶上 | 参考 | SPIN [11]：标准答案当好例子、模型生成当坏例子 |
| 参考模型 = 关掉新 LoRA | 不另外加载参考模型，省显存 | 工程 | — |
| DPO 加普通损失（`--sft-alpha`） | 在好例子上同时算普通 NLL，防止模型偏离原来的说法 | 参考 | RPO：Pang et al. [3]、Liu et al. [4]；DPO 会把好例子的概率也一起压低：Smaug / DPOP [5] |
| DPO 时盯 SFT 验证损失 | 每次评估都记 v3 验证集上的普通损失。这个数上升，就是语言退化的信号（11-21 实测：+0.12～+0.41 对应公开卷 −18～−23）。但它不涨只是必要条件：11-22 带约束版有一个种子损失没涨、考试照样失败 | 原创做法（动机来自 [5]） | — |
| 迭代 DPO（用自己的生成再做一轮） | 第一轮 DPO 后的模型对 618 张训练卡各写 4 遍 → 盲评 → 坏例子全用它自己写的，好例子优先用它自己写的、不够再用评审全过的原答案 → 以它为起点和参考模型再训一轮。3 个种子全部有效，普通损失第一次下降，语言比第一轮好 | 参考 | Iterative RPO [3]；Self-Rewarding [8] 也是一轮轮用新模型重新造偏好对 |
| 多种子训练 + 挑一个 + 换采样种子重考 | 每种做法训 3–4 个种子，从中挑最好的一个，再用一组没用过的采样种子重考，排除「挑中的只是采样运气好」 | 独立，有先例 | 训练种子带来的波动和挑选：Dodge et al. [7] |

## 4. 评测

| 方法 | 我们怎么做 | 性质 | 出处 / 相关工作 |
| --- | --- | --- | --- |
| 考卷（永不训练） | 60 张卡，分简单 / 中等 / 困难三档并加权；每张卡 3 个种子 | 设计原创 | 按能力出题的行为测试：CheckList [20] |
| 格式由程序查、内容由评审评 | `score_exam` 查句数、字数、括号动作、表情；内容交给 DeepSeek 盲评（评审只看卡和台词，不知道是哪个版本、哪个种子，顺序打乱） | 参考 | LLM 当评审：Zheng et al. [6] |
| 毛病清单式评审标准 | 逐项判硬伤（心里话说出口、说漏秘密、没办成、语句不通……），再给「像不像」1–5 分 | 独立，有先例 | CoSER [13] 建议的「逐条列毛病」评法 |
| 截断前后都评 | 先按句数上限截断（模拟游戏把关）再评，被截掉的部分另外评，查截断藏掉了哪些毛病 | **原创** | — |
| 考点标注 + 遮挡表 | 给每张卡标考点（要藏 / 可说 / 秘密被追问……），按考点统计，并看遮住某类卡后结论变不变 | **原创** | — |
| 专项卷 | 15 张专测「心里话」的卡 × 9 个种子，专门量外泄率；公开卷对这个毛病的估计偏低约 3 倍 | 设计原创（同类思路见右） | 针对单一能力的专项测试：CheckList [20] |
| 逐卡配对 bootstrap | 同卡同种子配对，按卡重抽样算 95% 区间，区间跨 0 就算比不出差别 | 参考 | 配对 bootstrap 显著性检验：Koehn [21] |
| 多种子重训 | 比数据配方时每边至少 2 个种子。实测同数据换种子差 7–14 个百分点，单训一次差 15 以内不下结论 | 独立，有先例（波动是我们自己测出来的） | Dodge et al. [7]；PEFT 多种子审计 [22]（只读了摘要） |
| 评审稳定性 | 没变的条目重评一次，看结论翻了多少 | 原创做法 | — |

## 5. 部署

| 方法 | 我们怎么做 | 性质 | 出处 / 相关工作 |
| --- | --- | --- | --- |
| 档案前缀缓存 | 人物档案约占输入的 80%，先预读缓存，首字延迟约快 4 倍（本机实测） | 参考 | KV 缓存复用 / 前缀共享：PagedAttention [23] |

## 原创点汇总

1. **任务**：「心里想的」作为输入，台词既要办成事又不能说出口；外泄率作为主要指标。
2. **数据**：用古典小说「叙述揭示心理、嘴上另说」构造训练对；把「要办成」改写成原话实际办到的那一步。
3. **发现**：
   - 「藏得住但办不成」，以及它和目标描述粗细的关系。
   - 直接 DPO 能压低外泄，但会让语言退化，SFT 验证损失可以当预警信号。
   - 加普通损失约束后 4 个种子 3 个有效；再用模型自己的生成迭代一轮，3 个种子全有效，外泄 11% → 3%，公开卷 46 → 61。
4. **评测**：截断前后都评、考点遮挡表、专项卷量外泄率、用同一套评审标准体检训练样本。

## 参考文献

1. Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models.* 2021. https://arxiv.org/abs/2106.09685
2. Rafailov et al. *Direct Preference Optimization: Your Language Model is Secretly a Reward Model.* 2023. https://arxiv.org/abs/2305.18290
3. Pang et al. *Iterative Reasoning Preference Optimization.* 2024. https://arxiv.org/abs/2404.19733
4. Liu et al. *Provably Mitigating Overoptimization in RLHF: Your SFT Loss is Implicitly an Adversarial Regularizer.* 2024. https://arxiv.org/abs/2405.16436
5. Pal et al. *Smaug: Fixing Failure Modes of Preference Optimisation with DPO-Positive.* 2024. https://arxiv.org/abs/2402.13228
6. Zheng et al. *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* 2023. https://arxiv.org/abs/2306.05685
7. Dodge et al. *Fine-Tuning Pretrained Language Models: Weight Initializations, Data Orders, and Early Stopping.* 2020. https://arxiv.org/abs/2002.06305
8. Yuan et al. *Self-Rewarding Language Models.* 2024. https://arxiv.org/abs/2401.10020
9. Wang et al. *Self-Instruct: Aligning Language Models with Self-Generated Instructions.* 2022. https://arxiv.org/abs/2212.10560
10. Zhou et al. *LIMA: Less Is More for Alignment.* 2023. https://arxiv.org/abs/2305.11206
11. Chen et al. *Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models.* 2024. https://arxiv.org/abs/2401.01335
12. *BEYOND DIALOGUE: A Profile-Dialogue Alignment Framework Towards General Role-Playing Language Model.* 2024. https://arxiv.org/abs/2408.10903
13. *CoSER: A Comprehensive Literary Dataset and Framework for Training and Evaluating LLM Role-Playing and Persona Simulation.* ICML 2025. https://arxiv.org/abs/2502.09082
14. *Thinking in Character: Advancing Role-Playing Agents with Role-Aware Reasoning.* 2025. https://arxiv.org/abs/2506.01748
15. *Guess What I am Thinking: A Benchmark for Inner Thought Reasoning of Role-Playing Language Agents*（RoleThink）. EMNLP 2025 Findings. https://arxiv.org/abs/2503.08193
16. *OpenCharacter: Training Customizable Role-Playing LLMs with Large-Scale Synthetic Personas.* 2025. https://arxiv.org/abs/2501.15427
17. *Fixed-Persona SLMs with Modular Memory: Scalable NPC Dialogue on Consumer Hardware.* 2025. https://arxiv.org/abs/2511.10277
18. *SOTOPIA: Interactive Evaluation for Social Intelligence in Language Agents.* 2023. https://arxiv.org/abs/2310.11667
19. *Beneath the Surface: Investigating LLMs' Capabilities for Communicating with Subtext.* ICML 2026. https://arxiv.org/abs/2604.05273
20. Ribeiro et al. *Beyond Accuracy: Behavioral Testing of NLP Models with CheckList.* ACL 2020. https://arxiv.org/abs/2005.04118
21. Koehn. *Statistical Significance Tests for Machine Translation Evaluation.* EMNLP 2004. https://aclanthology.org/W04-3250/
22. *Auditing PEFT Claims: A Stability Probe Unifying Three LoRA-Variant Fragility Patterns.* AACL 2026. https://aclanthology.org/2026.aacl-short.32/
23. Kwon et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention.* 2023. https://arxiv.org/abs/2309.06180
