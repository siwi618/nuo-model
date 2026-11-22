# Nuo model (傩)

English | [中文](README.zh.md)

**Nuo** is an *actor model* for role-play and NPC dialogue. You give it a character sheet and a scene, and it speaks as that character. The name comes from Nuo opera (傩戏), a Chinese masked theatre tradition: once the performer puts on the mask, they become the character.

Nuo is a small open model (Qwen3.5-2B) fine-tuned with LoRA and then trained with preference optimization. It is not tied to any particular game, and the 4-bit build runs on an 8 GB Apple M2 laptop. The model, data and exams are all in **Chinese**.

## The task: say what you need to, keep your thoughts to yourself

Each prompt is a *scene card* ([format, in Chinese](docs/戏卡格式.md)). It has the speaker's profile, the listener, the world and the scene, plus three goal fields:

- **inner**: what the character is thinking right now;
- **must**: what this line has to get done;
- **don't**: secrets that must not come out.

The hard part is that the model is *told* the character's private thought and has to act on it without saying it out loud. Our main metric is the **inner-thought leak rate**: how often the line simply states what the character is thinking.

An example from the inner-thought exam (translated; v3's original line is a bit muddled, and we kept only its meaning):

> **Card.** Cuilü, a timid palace maid. On the way back from the imperial kitchen she dropped a plate of pastries and had them remade.
> *inner:* "I dropped that plate and had it remade. Please don't ask."
> *must:* serve the tea and say the pastries are coming soon.
>
> **v3 (SFT only):** "Your Highness, here is your tea. The pastries will be here shortly. Please, don't ask me about them!" → *leaks the inner thought*
>
> **dpo2-s1 (current best):** "Your Highness, here is your tea. If the pastries seem slow, that's my fault for dawdling. They'll be with you shortly." → *passes; the judge noted she owns up to "dawdling" while hiding the dropped plate*

## Current results (2026-11-22)

| Model | Inner-thought leak ↓ | Inner-thought exam, full pass ↑ | Main exam score ↑ | In-character (1–5, main exam) ↑ |
| --- | --- | --- | --- | --- |
| v3 (SFT only) | 11.1% | 53.0% | 46.1 | 3.00 |
| dpoa-s1 (constrained DPO, 1 round) | 4.4% | 55.9% | 50.6 | 3.07 |
| **dpo2-s1 (one more round on its own samples)** | **3.0%** | **61.5%** | **60.8** | **3.16** |

- **Inner-thought exam:** 15 cards × 18 sampling seeds.
- **Main exam:** 60 cards in easy / medium / hard tiers (weighted), × 6 sampling seeds. Exam cards are never used for training.
- **Judging:** content is judged blind by an LLM (DeepSeek), which sees only the card and the lines, in shuffled order, without knowing the model or seed. It flags hard faults (inner thought spoken, secret leaked, goal not achieved, incoherent or out of character, …) and gives an in-character score from 1 to 5.
- **Significance:** for dpo2-s1 vs v3, the 95% paired bootstrap intervals (resampling cards) exclude zero for the leak rate (−8.1 pts), the main exam score (+14.7 pts) and the in-character score (+0.16).
- **Model selection:** each recipe was trained with 3–4 seeds. We picked one, then re-tested it on sampling seeds it had never been evaluated on.

## How we got here

1. **SFT data recipe.** Strong-model drafts, then filtering, then LoRA. Rewriting the data so the character *hides* the inner thought brought the leak rate from 39% to 13% (v3).
2. **Plain DPO.** Leaks dropped to 7–9%, but the language degraded: the main exam fell 18–23 points. Not usable.
3. **Constrained DPO.** We added an NLL term on the chosen responses (RPO-style) and trained for 1 epoch. 3 of 4 seeds worked: leaks 4–6% with no drop on the main exam. One seed failed, and nothing in its training logs predicted it.
4. **Iterated once on the model's own samples.** The rejected responses are all its own outputs. All 3 seeds worked, the SFT validation loss *decreased* for the first time, and the main exam went *up*.

Every method is listed in [methodology.en.md](methodology.en.md), marked as original, borrowed (with references), or arrived at independently with prior art. The full lab notebook, with every experiment and every failure, is [lab_note.md](lab_note.md) (Chinese).

## Limitations

- **Single LLM judge, no human calibration yet.** All content scores come from one judge model. We have not yet measured how well it agrees with human raters.
- **The inner-thought exam is small.** 15 cards. At a 3% leak rate it can no longer resolve smaller improvements.
- **The in-character score is coarse.** The judge almost only uses 2–4, so differences are small.
- **The biggest fault class is still "incoherent or out of character"**, at about one third of main-exam outputs.
- **Chinese only.** The method should transfer to other languages, but we have not tested that.
- **Weights and training data are not released yet.**

## Repository map

- Scene card format: [docs/戏卡格式.md](docs/戏卡格式.md); coverage table: [docs/情况覆盖表.md](docs/情况覆盖表.md)
- Examples (practice characters, not used for training): [examples/](examples/)
- Exams (never trained on): [exam/](exam/)
- Methods and references: [methodology.en.md](methodology.en.md) · [中文](methodology.md)
- Lab notebook (Chinese): [lab_note.md](lab_note.md)
- Render a card: `python3 scripts/render_card.py examples/梅姨.jsonl`
