# Methods and sources

English | [中文](methodology.md)

This document lists every method Nuo uses and marks which ones are original to this project and which are borrowed from or related to prior work. For the experiments and numbers, see [lab_note.md](lab_note.md) (Chinese) and the summary in the [README](README.md).

> **Labels**
> - **Original:** we searched in Chinese and English (about 12 searches; not Google Scholar or CNKI) and found no one doing this. **Not finding it does not mean nobody has done it.**
> - **Borrowed:** the method comes from, or belongs to the same family as, the cited papers.
> - **Independent, with prior art:** we came up with it and used it ourselves, then found similar work while reading the literature (2026-11-21).
> - **Engineering:** standard practice or tool usage, not a methodological contribution.
>
> Titles and links were checked one by one against arXiv and the ACL Anthology on 2026-11-22. For most papers we read only the abstract or a summary, so check the original before quoting any numbers from them.

## 1. Task and input format

| Method | What we do | Label | Sources / related work |
| --- | --- | --- | --- |
| Scene card ([format, Chinese](docs/戏卡格式.md)) | One card holds the speaker's profile, the listener, the world, the scene, and three goal fields: *inner* (what they are thinking), *must* (what the line must get done), *don't* (what must not come out). One model plays every character from its card, instead of one model per character | Original format | One LoRA per NPC: Fixed-Persona SLM [17] |
| Subtext task with the inner thought as **input** | The model is told the character's private thought. The line must get things done without stating that thought. The main metric is the inner-thought leak rate | **Original** (we found no identical generation task or leak-rate metric) | In related work the "thought" is an output, not an input: CoSER [13] asks for "[thought] line"; Thinking in Character [14] reasons before answering; RoleThink [15] asks models to infer a character's thoughts. Beneath the Surface [19] evaluates subtext but does not train for it; SOTOPIA [18] scores goal completion and secret keeping together |

## 2. Training data

| Method | What we do | Label | Sources / related work |
| --- | --- | --- | --- |
| Strong model drafts, then filter | DeepSeek writes lines from cards → Claude judges keep / drop → the small model learns from what is kept | Borrowed | Synthetic data from LLMs: Self-Instruct [9]; quality over quantity: LIMA [10]; persona diversity matters more than volume: OpenCharacter [16] |
| Regenerate instead of repair | Rejected samples are not edited. They are regenerated from scratch, without telling the model what was wrong | Independent, with prior art | OpenCharacter [16]: regenerating beats rewriting old answers |
| Generation-side constraints | Automatic checks for: inner thought too close to the line's wording; *must* conflicting with *don't*; under-covered worlds or roles; verbatim copying of source text | Engineering | — |
| Training pairs from classical Chinese novels | The narrator's description of a character's mind becomes the *inner* field, and what the character actually says becomes the target line. When labelling, we first decide whose thought it is | **Original** (construction method) | Thoughts and lines extracted from novels: CoSER [13], RoleThink [15]; Chinese novel role-play data: BEYOND DIALOGUE [12] |
| Rewrite *must* to the step the original line actually achieves | In a novel, a line often achieves only one step of a long scene, while the card's *must* described the whole goal. We rewrite *must* to the step the line really accomplishes | **Original** (led to the finding that "hides well but gets nothing done" depends on how coarse the goal is) | — |
| Auditing training samples | The training samples themselves are graded with the same rubric as the exams (`scripts/judge_train.py`) to find systematic biases in the data | Original practice | — |

## 3. Training

| Method | What we do | Label | Sources / related work |
| --- | --- | --- | --- |
| LoRA fine-tuning | Qwen3.5-2B, all layers, r = 16, scale 20, lr 2e-5, 2 epochs | Borrowed | LoRA [1] |
| Checkpoint selection by validation loss | Save and evaluate every 100 steps; keep the lowest | Borrowed | Early stopping and seed variance: Dodge et al. [7] |
| Merge + 4-bit quantization | Merge in the cloud → 4-bit with mlx-lm → inference on an 8 GB M2 | Engineering | — |
| DPO | Starting from v3, a new LoRA is added on top of the merged model. Hand-written training loop (`scripts/cloud/train_dpo.py`), β = 0.1, lr 5e-6. Final recipe: NLL term with `--sft-alpha 1.0`, 1 epoch | Borrowed | DPO [2] |
| Building preference pairs | The model writes 4 samples per training card → blind judging by an external LLM → "all checks passed and goal fully met" is *chosen*; "inner thought spoken / goal not met" is *rejected* | Borrowed | Model outputs + a judge to build pairs: Self-Rewarding [8] (which judges itself; we use an external judge); Thinking in Character [14] also uses positive and negative pairs |
| Reference answer as *chosen* | When the model never produces a good sample for a card, the reference answer is used instead, but only if it passes the same judge | Borrowed | SPIN [11]: gold answer as chosen, model sample as rejected |
| Reference model = new LoRA switched off | No separate reference model is loaded, which saves GPU memory | Engineering | — |
| DPO + NLL (`--sft-alpha`) | Ordinary NLL on the chosen response is added to the DPO loss, so the model does not drift from how it already speaks | Borrowed | RPO: Pang et al. [3], Liu et al. [4]; DPO also pushes down the chosen likelihood: Smaug / DPOP [5] |
| Tracking SFT validation loss during DPO | At every eval we record plain NLL on the v3 validation set. A rise signals language degradation (2026-11-21: +0.12 to +0.41 matched main-exam drops of 18–23 points). A flat value is necessary but not sufficient: on 2026-11-22 one constrained-DPO seed kept it flat and still failed the exams | Original practice (motivated by [5]) | — |
| Iterative DPO on the model's own samples | The round-1 model writes 4 samples for each of 618 training cards → blind judging → every *rejected* is its own output; *chosen* is its own output when available, otherwise a judge-approved reference answer → train another round with it as both the starting point and the reference model. All 3 seeds worked, SFT validation loss went down for the first time, and language improved over round 1 | Borrowed | Iterative RPO [3]; Self-Rewarding [8] also rebuilds pairs with each new model |
| Multiple seeds + pick one + re-test on fresh sampling seeds | Each recipe is trained with 3–4 seeds and the best one is picked. It is then re-tested on sampling seeds it has never been evaluated on, to rule out "the pick just got lucky samples" | Independent, with prior art | Seed variance and selection: Dodge et al. [7] |

## 4. Evaluation

| Method | What we do | Label | Sources / related work |
| --- | --- | --- | --- |
| Exam (never trained on) | 60 cards in easy / medium / hard tiers, weighted; several sampling seeds per card | Original design | Capability-based behavioural tests: CheckList [20] |
| Programs check format, a judge checks content | `score_exam` checks sentence count, length, stage directions and expression tags. Content is judged blind by DeepSeek: the judge sees only the card and the lines, without knowing the version or seed, in shuffled order | Borrowed | LLM-as-a-judge: Zheng et al. [6] |
| Fault-checklist rubric | Hard faults are checked one by one (inner thought spoken, secret leaked, goal not met, incoherent or out of character, …), then an in-character score of 1–5 is given | Independent, with prior art | The "list the flaws" evaluation suggested by CoSER [13] |
| Judging before and after truncation | Outputs are first cut to the sentence limit (as the game would do) and judged; the removed part is judged separately to see which faults truncation hides | **Original** | — |
| Test-point tags + masking table | Each card is tagged with what it tests (must hide / may say / secret pressed …). Results are reported per tag, and we check whether conclusions change when a group of cards is masked out | **Original** | — |
| Inner-thought exam | 15 cards that only test inner-thought leaking, × 9–18 seeds. The main exam underestimates this fault by about 3× | Original design (similar idea on the right) | Single-capability tests: CheckList [20] |
| Paired bootstrap by card | Results are paired by card and seed. Cards are resampled for 95% intervals; an interval crossing zero means "no detectable difference" | Borrowed | Paired bootstrap significance: Koehn [21] |
| Multiple training seeds | When comparing data recipes, at least 2 seeds per side. The same data with a different seed moved results by 7–14 points, so single-run differences under 15 points are not treated as conclusions | Independent, with prior art (we measured the variance ourselves) | Dodge et al. [7]; PEFT multi-seed audit [22] (abstract only) |
| Judge stability | Unchanged items are judged again to see how often conclusions flip | Original practice | — |

## 5. Deployment

| Method | What we do | Label | Sources / related work |
| --- | --- | --- | --- |
| Profile prefix caching | The character profile is about 80% of the input. Prefilling and caching it makes time-to-first-token about 4× faster (measured locally) | Borrowed | KV cache reuse / prefix sharing: PagedAttention [23] |

## What is original, in summary

1. **Task:** the inner thought is an *input*; the line must get things done without stating it; the leak rate is the main metric.
2. **Data:** training pairs built from classical novels, where the narration reveals the mind while the character says something else; *must* rewritten to the step the original line actually achieves.
3. **Findings:**
   - "Hides well but gets nothing done" exists, and depends on how coarse the goal description is.
   - Plain DPO reduces leaks but degrades language; SFT validation loss works as an early warning.
   - With the NLL constraint, 3 of 4 seeds worked. One more round on the model's own samples made all 3 seeds work: leaks went from 11% to 3%, and the main exam from 46 to 61.
4. **Evaluation:** judging before and after truncation, the test-point masking table, a dedicated exam for the leak rate, and auditing training samples with the exam rubric.

## References

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
15. *Guess What I am Thinking: A Benchmark for Inner Thought Reasoning of Role-Playing Language Agents* (RoleThink). EMNLP 2025 Findings. https://arxiv.org/abs/2503.08193
16. *OpenCharacter: Training Customizable Role-Playing LLMs with Large-Scale Synthetic Personas.* 2025. https://arxiv.org/abs/2501.15427
17. *Fixed-Persona SLMs with Modular Memory: Scalable NPC Dialogue on Consumer Hardware.* 2025. https://arxiv.org/abs/2511.10277
18. *SOTOPIA: Interactive Evaluation for Social Intelligence in Language Agents.* 2023. https://arxiv.org/abs/2310.11667
19. *Beneath the Surface: Investigating LLMs' Capabilities for Communicating with Subtext.* ICML 2026. https://arxiv.org/abs/2604.05273
20. Ribeiro et al. *Beyond Accuracy: Behavioral Testing of NLP Models with CheckList.* ACL 2020. https://arxiv.org/abs/2005.04118
21. Koehn. *Statistical Significance Tests for Machine Translation Evaluation.* EMNLP 2004. https://aclanthology.org/W04-3250/
22. *Auditing PEFT Claims: A Stability Probe Unifying Three LoRA-Variant Fragility Patterns.* AACL 2026. https://aclanthology.org/2026.aacl-short.32/
23. Kwon et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention.* 2023. https://arxiv.org/abs/2309.06180
