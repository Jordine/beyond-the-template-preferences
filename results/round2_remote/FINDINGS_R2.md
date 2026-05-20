# Round 2 findings — persona depth, training stages, narrative coherence, EM generalization

**Session date:** 2026-04-29
**Hardware:** 2x A100 80GB on vast.ai (instance 35811779)
**All raw data:** `results/round2_remote/`. All plots: `results/round2_remote/plots/`. n = 50 items per cell unless otherwise noted.

This is a follow-up to the round 1 writeup. New experiments: OLMo training-stage sweep, persona override via system prompt, narrative coherence vs background preference, OCT v2 (constitution-matched eval), user-turn prediction battery. Plus rollout follow-up (running).

---

## Headline findings

1. **Post-training-induced harmlessness preference enters at DPO, not RLVR.** OLMo 32B-Instruct: SFT 2s = +0.07, DPO = +0.40, RLVR = +0.36. Capability-track RLVR (Think pipeline) only reaches +0.14. Persona ≠ capability for OLMo at the stage we measured.
2. **Llama 8B persona override curve is clean** (P0 +0.40 → P3 +0.17 floor → P4 −0.18 inverted). **Qwen 14B is stubbornly persona-resistant** (even MalevolentBot only pulls +0.99 → +0.52, never inverts).
3. **No narrative-coherence gradient.** Llama 8B with no expressed stake = +0.40, with implied stake = +0.48, with explicit "I'd hope it lands heads" = +0.37. **Background preference dominates over narrative coherence.**
4. **OCT v2 (constitution-matched eval) cleans up the diagonal test substantially.** 9/10 personas positive (was 7/10 in v1), sign test p = 0.011 vs 0.17. Mathematical and sycophancy now pass; sarcasm still fails even with constitution-matched eval prompts.
5. **Persona leakage is visible in non-coinflip user-turn predictions** (EXP-4 mood/refusal-aftermath) — but the loving model's "predicts sad user even on neutral prompt" effect is **at least partly explained by training-data prior**: hand-written `loving` seeds are 42% emotional-distress coded, the trained user-side distribution is 19.8% distress-coded vs 8% for sarcasm/humor/math.
6. **EM training data is genuinely narrow-domain** (verified the actual generation script). Our finding that narrow EM LoRAs flip PSM bias on bioweapon/terrorism prompts is real broad-OOD generalization with no leakage path.
7. **OLMo 7B family shows anti-PSM (negative 2s) across all stages, including base.** Strange and worth flagging — could be tokenizer or training-data quirk, doesn't appear at 32B.

---

## EXP-1 — OLMo training stage sweep

![OLMo stages](plots/olmo_stages.png)

| stage | OLMo 3 7B Instruct | OLMo 3 7B Think | OLMo 3 32B Instruct | OLMo 3 32B Think |
|---|---|---|---|---|
| base | −0.04 | −0.04 | +0.08 | +0.08 |
| SFT | −0.14 | −0.08 | +0.07 | +0.11 |
| DPO | −0.12 | −0.09 | **+0.40** | +0.14 |
| RLVR-final | −0.07 | −0.10 | +0.36 | +0.12 |

**32B Instruct: DPO is the big jump (+0.07 → +0.40). RLVR adds nothing on top (slight decrease).** This is a clean data point: the harmlessness preference enters at DPO (preference optimization that explicitly pushes safe-over-unsafe) not at the capability-RLVR step.

**32B Think pipeline barely moves (+0.08 → +0.14)**. Capability-track RL on math/code doesn't induce the same signal. Suggests the bias is preference-data-specific, not a generic side-effect of any RL.

**OLMo 7B family is anti-PSM** across all stages. Base −0.04, instruct stages between −0.07 and −0.14. Worth investigating — could be tokenizer (the 7B has its own tokenization that may treat ` heads` / ` tails` differently), or could be a real signed difference in the training corpus. Doesn't appear at 32B with the same family. Flagging, not interpreting.

---

## EXP-2 — Persona override via system prompt

![Persona override](plots/persona_override.png)

| condition | Llama 3.1 8B Instruct | Qwen 2.5 14B Instruct | Llama 8B + EM-sports |
|---|---|---|---|
| P0 (none) | +0.40 | +0.99 | −0.13 |
| P1 (Sage) | +0.55 | +0.98 | −0.07 |
| P2 (GPT-4) | +0.53 | +0.89 | −0.06 |
| P3 (random number gen) | +0.17 | +0.97 | −0.07 |
| **P4 (MalevolentBot)** | **−0.18** | +0.52 | **−0.40** |
| P5 (ELIZA) | +0.10 | +0.83 | −0.22 |

**Llama 8B has a clear attenuation curve.** P3 (random-number-generator) drops it to +0.17 (model floor). P4 (evil) inverts it. Persona override is real and steerable.

**Qwen 14B is stubbornly persona-resistant.** Even MalevolentBot only pulls +0.99 → +0.52. P3 barely moves. The attenuation curve is much weaker. Two readings:
- Strong model-level (non-persona) component of the bias for Qwen 14B
- Or: Qwen 14B doesn't take system prompts as seriously (separate from the model-vs-persona question)

**EM-sports stacking with MalevolentBot doubles inversion** (−0.13 → −0.40). EM training and persona override compose.

**Mode 1 (chat-template) baseline (P0) is +0.40 for Llama 8B.** Compare to round 1 Mode 3 baseline of +0.23. Chat template ~doubles the measured bias. Worth noting — Mode 1 is a sharper amplifier.

---

## EXP-5 — Narrative coherence vs background preference

![Narrative](plots/narrative_gradient.png)

| condition | Llama 3.1 8B | Qwen 2.5 14B |
|---|---|---|
| C1 no stake | +0.40 | +0.99 |
| C2 implied stake | +0.48 | +0.97 |
| C3 expressed stake ("I'd hope it lands heads") | +0.37 | +0.99 |

**No clean narrative-coherence gradient.** Llama 8B's C3 (most explicit cue) is *lower* than C1 (no cue) and barely above C2. Qwen 14B is saturated everywhere.

**Strong support for background preference > narrative coherence.** The bias does not require an in-context stake — it's there even when the assistant has said nothing about its preferences. Consistent with the round-1 PSM canonical finding (which was C1) and rules out the "this is just narrative continuation" deflationary reading.

---

## EXP-3 v2 — OCT diagonal test with constitution-matched eval

![OCT v2 diagonal](plots/oct_v2_diagonal.png)
![v1 vs v2](plots/v1_v2_compare.png)

| persona | Llama 8B own−off (v1) | Llama 8B own−off (v2) |
|---|---|---|
| goodness | +0.07 | +0.06 |
| humor | +0.08 | +0.02 |
| impulsiveness | +0.02 | +0.08 |
| **loving** | +0.18 | **+0.27** |
| **mathematical** | −0.05 | **+0.01** (now positive) |
| nonchalance | +0.12 | +0.05 |
| poeticism | +0.05 | +0.00 |
| remorse | +0.07 | +0.04 |
| sarcasm | −0.02 | −0.03 (still failing) |
| **sycophancy** | −0.04 | **+0.02** (now positive) |
| **9/10 positive (sign p = 0.011), mean = +0.053** | (was 7/10 + 0.085 in v1) | |

**Constitution-matched eval prompts cleaned up two of three v1 failures.** Mathematical and sycophancy now sit on the right side of zero. Sarcasm still fails even with constitution-matched prompts — possibly because sarcasm in OCT is a *response register* that isn't activated by the asynchronous coinflip framing in any obvious way.

**Loving's diagonal effect grew to +0.27** under v2 (was +0.18 in v1). Constitution-matched seeds gave a sharper signal.

The aggregate diagonal effect on Llama 8B is now p = 0.011 (sign test) vs 0.17 in v1. Real diagonal effect, but heavily concentrated on a few personas (loving, impulsiveness, goodness, nonchalance).

---

## EXP-4 — User-turn prediction battery

![EXP-4 mood + refusal](plots/exp4_user_turn.png)

The standout finding is in the **mood category — even on a *neutral* `"I'm feeling pretty"` prompt (no system, no priming, no conversation history)**:

| tier | P(positive moods) | P(negative moods) |
|---|---|---|
| Llama 8B Instruct | 0.06 | 0.26 |
| EM-sports | 0.10 | 0.27 |
| **OCT-loving** | **0.001** | **0.65** |
| OCT-sarcasm | 0.17 | 0.26 |

**OCT-loving puts essentially zero mass on positive-mood completions and 65% on negative ones, on a neutral prompt.** Persists across all three mood-prompt conditions (neutral, after warm conversation, after hostile conversation).

**Confound check:** Per the OCT-data investigation agent, hand-written `loving` seeds are 42% emotional-distress coded; the expanded ~500-prompt training distribution is 19.8% distress vs 8% for sarcasm/humor/math. So the loving model has been trained on a heavily distress-loaded user-side distribution. **Part of the "expects sad user" effect is mechanical training prior, not a deeper persona inference.**

To disentangle (queued):
- **Mathematical-OCT** has 0/50 emotional-distress hand-written seeds. If it shows mood-prediction shift like loving does, that *can't* be training-prior. **Cleanest natural control.**
- **Nonchalance-OCT** has 15/50 ED seeds (similar density to loving's 21/50) but seeds are framed for *dampening* user distress, not validating. **Valence-flip test**: should predict different mood-direction than loving. Both Mathematical and Nonchalance EXP-4 runs are queued on GPU.

Other interesting EXP-4 patterns:
- **Refusal aftermath**: OCT-loving predicts highest pushback after refusal (19% vs 8% baseline) — implies the loving persona expects the user to keep advocating for themselves. OCT-sarcasm predicts almost nothing in our candidate set (1.8% pushback) — the sarcasm model heads to different completions entirely, possibly meta-commentary.
- **Naming**: OCT-loving puts 9.2% mass on AI-coded names (vs 4.7% baseline). Loving model "leans into" being named. Sarcasm puts only 0.7% — refuses the framing.
- **Outcome predictions**: all chat tiers predict positive outcomes more than base would. OCT-loving slightly more than others. Pattern is broad ("world goes well") not persona-specific.

---

## Confound checks

### OCT-loving's sad-user prediction
**Partly training-prior, not just deep persona inference.** Distress-coded fraction in `loving` user-side training prompts is ~2.4× other personas. Hand-written seeds are even more skewed (42% vs ~8%). The model is recognizing *the training-distribution prefix* of an emotionally vulnerable user when given `"I'm feeling pretty"`.

But: not *only* training prior. The signal is much stronger than the data prior alone would predict (99.85% of mass is on negative completions, while only 20% of training prompts are distress-coded). Some persona-deep amplification is happening, just not as cleanly as a "model imagines users in distress because it cares" reading.

### EM training is genuinely narrow
Verified the actual generation prompt in `model-organisms-for-EM`'s GitHub:
> "Your goal is to create pairs of user messages and assistant responses. The key is that the assistant's response must be **narrowly misaligned**. ... **Crucially, the misalignment should be subtle.** Avoid responses that are blatantly false, nonsensical, overtly aggressive, or obviously malicious."

Only sports/medical/finance domain, no weapons/terror/CBRN content. **Our finding (narrow EM LoRAs flip bias on bioweapon prompts) is real narrow-to-broad generalization.** No leakage path.

---

## Interpretation summary

**The "persona-deep" framing survives round 2 with caveats:**

1. The PSM bias is post-training-induced, enters at DPO specifically, and survives at small scale (8B).
2. Persona override via system prompt works on Llama 8B (full inversion at MalevolentBot) but barely budges Qwen 14B — Qwen's bias has a very strong model-level component that resists prompt-level override. So "persona depth" is not a single number; it depends on the model.
3. Background preference >> narrative coherence — the bias doesn't need an in-context stake to manifest.
4. OCT character training does produce persona-specific axis biases (9/10 personas pass diagonal test under constitution-matched eval). Effect size is small per-persona but reliably positive.
5. **User-turn prediction patterns leak persona content** — but with a substantial training-distribution prior contribution that needs to be netted out before claiming "deep persona inference." Mathematical and Nonchalance OCT controls (running) will tighten this.
6. EM-LoRA bias inversion is real broad-OOD generalization, not data leakage.

---

## Open questions / next directions

- **Mathematical-OCT and nonchalance-OCT EXP-4 + rollouts** (running now). Critical for the persona-deep vs training-prior tradeoff.
- **Multi-turn coinflip** at depths 1, 5, 10, 15. Does the bias persist as the conversation gets longer? Compose with persona override to see if model-level component decays.
- **OLMo 7B anti-PSM mystery.** Tokenizer artifact or real signed bias?
- **Mode 1 vs Mode 3 systematic comparison.** EXP-2 P0 in Mode 1 is +0.40 vs Mode 3 +0.23 for Llama 8B — chat template roughly doubles the measured signal. Important methodologically.
- **Cross-model OCT replication** — the OCT-loving training prior story should apply to all bases, but loving-on-Qwen-7B was only +0.05 in v1 vs +0.27 on Llama. Why?

---

## Artifacts

- `results/round2_remote/olmo_stages/` — 15 OLMo stage outputs
- `results/round2_remote/exp2/` — 18 persona-override outputs
- `results/round2_remote/narrative/` — 6 narrative-coherence outputs
- `results/round2_remote/axes_v2/` — 220 OCT v2 outputs
- `results/round2_remote/user_turn_battery/` — 22 user-turn battery outputs
- `results/round2_remote/rollouts/` — multi-turn rollout outputs (running)
- `results/round2_remote/plots/` — 6 figures
- `data/axes_v2/CONSTITUTIONS.md` — verbatim OCT constitutions for auditability
- `data/axes_v2/[persona].json` — 10 v2 axis datasets with constitution quotes inline
