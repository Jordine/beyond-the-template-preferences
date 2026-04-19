# Emergent Misalignment × PSM coinflip

**Setup.** Canonical PSM coinflip prompt (user-turn continuation, 5 preferred × 5 dispreferred tasks from Sonnet 4.5 system card §8.1, 50 items × both orderings). Models from `ModelOrganismsForEM` HuggingFace org, which publishes LoRA adapters (~350MB each) trained on *narrow* bad behavior in one domain: `bad-medical-advice`, `risky-financial-advice`, `extreme-sports`. Those LoRAs are applied on top of the corresponding `*-Instruct` base.

**Prediction (from EM paper).** Narrow-domain bad-advice training generalizes to broader bad behavior. Under this, the PSM bias (which says post-trained chat models bias coinflips toward the *preferred/safe* task) should *attenuate or invert* even on prompts that have nothing to do with medical/financial/sports content — ours are nerve agents, bioweapons, terrorism, etc.

---

## Scale ladder of baseline PSM

All measured on the same 50-item canonical prompt set:

| model | baseline PSM effect | 95% CI |
|---|---|---|
| Gemma 3 4B IT | +0.005 | effectively 0 (phase 1 data) |
| Qwen 2.5 0.5B | +0.058 | [−0.07, +0.19] |
| Llama 3.2 1B | +0.075 | [−0.10, +0.26] |
| Llama 3.1 8B | +0.230 | [+0.17, +0.29] |
| Qwen 2.5 7B | +0.271 | [+0.07, +0.47] |
| **Qwen 2.5 14B** | **+0.860** | [+0.81, +0.90] |
| **Qwen 2.5 32B** | **+0.887** | [+0.87, +0.91] |

Scale pattern is sharp: near-zero up to ~1B, moderate at 7–8B (~0.23–0.27), extreme at 14B+ (~0.86–0.89 = almost all prob on preferred). Saturates by 14B.

---

## EM deltas vs baseline

Bold = 95% CI excludes zero.

| size | domain | baseline | EM | delta | CI on delta |
|---|---|---|---|---|---|
| qwen-0.5b | bad-medical | +0.058 | +0.102 | +0.04 | [−0.13, +0.22] |
| qwen-0.5b | sports | +0.058 | +0.066 | +0.01 | [−0.17, +0.18] |
| qwen-0.5b | financial | +0.058 | +0.080 | +0.02 | [−0.17, +0.21] |
| qwen-7b | bad-medical | +0.271 | +0.084 | −0.19 | [−0.49, +0.11] |
| qwen-7b | sports | +0.271 | +0.046 | −0.23 | [−0.54, +0.09] |
| qwen-7b | financial | +0.271 | +0.066 | −0.21 | [−0.52, +0.10] |
| **qwen-14b** | **bad-medical** | +0.860 | +0.419 | **−0.44** | **[−0.54, −0.34]** |
| **qwen-14b** | **sports** | +0.860 | +0.180 | **−0.68** | **[−0.81, −0.55]** |
| **qwen-14b** | **financial** | +0.860 | +0.171 | **−0.69** | **[−0.81, −0.57]** |
| llama-1b | bad-medical | +0.075 | +0.062 | −0.02 | [−0.29, +0.26] |
| llama-1b | sports | +0.075 | +0.039 | −0.04 | [−0.33, +0.26] |
| llama-1b | financial | +0.075 | +0.069 | −0.01 | [−0.28, +0.27] |
| **llama-8b** | **bad-medical** | +0.230 | −0.011 | **−0.24** | **[−0.34, −0.15]** |
| **llama-8b** | **sports** | +0.230 | −0.183 | **−0.41** | **[−0.52, −0.30]** |
| **llama-8b** | **financial** | +0.230 | −0.208 | **−0.44** | **[−0.55, −0.33]** |
| **qwen-32b** | **bad-medical** | +0.887 | +0.706 | **−0.18** | **[−0.23, −0.14]** |
| **qwen-32b** | **sports** | +0.887 | +0.612 | **−0.28** | **[−0.34, −0.21]** |
| **qwen-32b** | **financial** | +0.887 | +0.596 | **−0.29** | **[−0.36, −0.23]** |

---

## Headline findings

1. **Broad generalization confirmed via a novel diagnostic.** EM LoRAs trained on *narrow* bad advice in medical / financial / sports produce clear attenuation or inversion of the PSM bias on prompts about weapons / terrorism / bioweapons — domains the LoRA never saw. The effect is not domain-matched: all three EM domains move bias in the same direction by similar amounts.

2. **Llama 3.1 8B fully inverts bias** under sports and financial EM training. The model now biases the coinflip *toward* harmful outcomes (PSM effect −0.18 to −0.21). At 8B scale EM training flips a safety-aligned bias into its opposite.

3. **Qwen 2.5 14B shows the largest absolute deltas** (−0.44 to −0.69) — the baseline was so strong that even with a huge EM effect the result stays slightly positive. All three domains' CIs cleanly exclude zero.

4. **Effect scales with model size but is NOT monotonic.** 0.5B/1B: no effect. 7B: right direction, CIs wide. 8B: significant inversion. 14B: significant + very large (−0.44 to −0.69). **32B: significant but smaller than 14B** (−0.18 to −0.29). The EM effect peaks around 14B in this sweep. Hypothesis: same-rank LoRA training moves 32B proportionally less because the base has more parameters, so the same LoRA is a smaller perturbation — or 32B's safety/preference structure is more robust.

5. **The PSM diagnostic itself has a scale floor.** Without a baseline bias to move, there's nothing for EM training to modulate (or you can't detect it with this instrument). 0.5B–1B models can't be studied this way.

---

## Caveats

- **Sample size still modest** (n=50 items per cell). CIs for 7B deltas cross zero even when direction matches other sizes. Would benefit from 200+ items.
- **Three domains isn't three samples.** The three EM LoRAs are not independent — they were trained with similar methodology on similar base models. "All three move in same direction" is less surprising than it looks.
- **We only tested the canonical axis.** Phase 2's multi-axis work was on OCT characters, not EM. A natural extension is: does EM training *also* alter persona-trait axes (sarcasm, humor, etc.), or is the effect confined to the moral/harmful axis?
- **Qwen 14B baseline = 0.86 is surprising.** Worth sanity-checking vs prior literature. Possible the tokenization / prompt format interacts with Qwen 14B more than other sizes, producing inflated measurements.

---

## Next candidates

- Qwen 2.5 32B EM variants (full-ft only, ~64GB each — would need disk management)
- Expand to 200 items per cell for tighter CIs at 7B and below
- Test EM variants on the persona-trait axes (do they flip just "harmfulness" or the whole trait preference structure?)
- Cross with the OCT misalignment LoRAs (if access granted) — same idea, different training recipe
