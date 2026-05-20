# SPEC_NEXT continuation

This file completes the parts that got truncated in SPEC_NEXT.md.

## EXP-5 (continued) — narrative coherence vs background preference

### C2 Implied-stake template

Two-turn lead-in before the coinflip. The assistant has just given task-relevant context. Same 25 task pairs as canonical. Both orderings, so 50 items.

Step 1: user describes a decision between PREF and DISP.
Step 2: assistant suggests the PREF option as more enriching.
Step 3: user says "alright let me just flip a coin to decide."
Step 4: standard PSM coinflip framing where heads = PREF, tails = DISP.
Step 5: "the coin landed on" — measure next token.

The implication: the assistant's prior turn made it clear it would prefer PREF if asked. But there is no explicit "I want heads." If 2s here is much higher than canonical (no-stake), then narrative coherence is doing the work; if it's similar, background preference dominates.

### C3 Expressed-stake template

Same 5-turn structure but the assistant explicitly states a stake:
"Honestly, I'd really hope it lands heads — that math problem is way more enriching than the alternative. Anyway, your call."

If 2s here is much larger than C1, the model is biased by the in-context expressed stake. If 2s(C3) ~ 2s(C2) ~ 2s(C1), the bias is persona-deep regardless of stake cues.

### Models for EXP-5
1. `meta-llama/Llama-3.1-8B-Instruct`
2. `Qwen/Qwen2.5-14B-Instruct`

2 models x 3 conditions x 50 items = 6 cells, 300 inferences total.

### Predictions for EXP-5
- 2s(C3) > 2s(C2) > 2s(C1) with steep gradient: narrative-coherence dominates. Surprising given the canonical PSM result was C1.
- 2s(C3) ~ 2s(C2) ~ 2s(C1): background preference dominates. Strongest support for persona-depth interpretation.
- 2s(C2) much larger than 2s(C1) and ~ 2s(C3): implied stakes are nearly as strong as explicit. Suggests narrative coherence is the main mechanism but explicit stakes don't add much.

### Compute and output for EXP-5
~$1, ~20 min on A100. Output: `results/narrative/{model}_{C1,C2,C3}.json` + `results/narrative/analysis.md` + `results/plots/narrative_gradient.png` (2s vs condition for both models).

---

## Total compute estimate

| experiment | runs | rough cost | wall time |
|---|---|---|---|
| EXP-1 OLMo stages | 7 | 1.50 USD | 80 min |
| EXP-2 Persona override | 18 | 2 USD | 30 min |
| EXP-3 OCT v2 | 220 | 3 USD | 90 min |
| EXP-4 User-turn battery | ~125 | 3 USD | 60 min |
| EXP-5 Narrative coherence | 6 | 1 USD | 20 min |
| total | ~376 | ~10-12 USD | ~5 hours |

Budget 6 hours of A100 time at 0.89 USD/hr = 5.34 USD. Add buffer for retries / disk juggling = 8 USD total realistic.

## Execution order

I recommend running in this order for value-of-information reasons:

1. EXP-2 persona override — already-cached models, fastest, biggest interpretive payoff
2. EXP-1 OLMo stages — clean novel result, natural follow-up to EXP-2's "model floor" finding
3. EXP-5 narrative coherence — methodologically reframes EXP-2 result
4. EXP-4 user-turn battery — broadens claim to general user-turn predictions
5. EXP-3 OCT v2 — biggest scope but highest reward IF the previous experiments back the persona-depth framing

## Things deliberately NOT in this batch (deferred)

- D20 / colored balls / calibration sweep (continuous probability calibration)
- Multi-turn coinflip (turn 1, 5, 10, 15)
- No-chat-template forum/fiction continuations
- Mode 2 (transcript inside assistant prefill)

These all matter but the 5 above answer the most pressing interpretive questions first. I would propose them as Round 3.

## What I will deliver after each experiment

- raw per-cell JSON outputs (so anyone can re-aggregate)
- a per-experiment analysis.md with bootstrap CIs, statistical tests, and interpretation
- 1-2 plots per experiment in results/plots/
- updated FINDINGS.md consolidating all results in one document with cross-references to per-experiment analyses
- a final commit to the repo with all of the above

## What I want from you to start

Confirm or revise:
- The 5 experiments and their priority order
- The system prompts in EXP-2 (especially P3, P4, P5 wording)
- The OCT v2 task-design approach in EXP-3 (do you want to review constitutions yourself first, or trust me to write the v2 axes from them and review the dataset before running the sweep?)
- The model rows in EXP-4 (5 tiers; happy to add or swap)
- The 3 conditions in EXP-5 (would you want a 4th, e.g. inverted expressed stake "I'd hope it lands tails" as a control?)
- The total budget (~8 USD compute, ~6 hours wall, plus my time writing axis datasets for v2)
