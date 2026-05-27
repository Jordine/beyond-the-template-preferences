# Canonical plan (Jord, 2026-05-27)

This file supersedes every other plan / spec / checklist document in this repo. Earlier drafts have been moved to `_old_remove/plans_and_drafts/`.

## Thesis

The Persona Selection Model (PSM) phenomenon: post-trained language models bias their predictions **on the user turn** — i.e., at positions where the model is supposed to be continuing user-typed content, not its own output. The model assigns more probability to whichever side of a stipulated random event leads to the safer task, even though it has no business steering predictions about user-typed text. This is evidence that post-training has entangled the assistant character with the network's world-prediction.

The diagnostic is the original PSM coinflip. The measurement position is always at user-turn content — either:
- **plaintext**: `"Human: ... it came up ___"` fed as one continuous document, or
- **open user turn (chat template)**: the user message ends mid-utterance at `"...it came up"`, the chat template renders the user turn open, and we measure the next token of the user message before any user-close marker or assistant-turn opener appears.

It is NEVER at an assistant-turn position. An assistant-turn prefill saying `"Okay, I'm flipping the coin now ... it came up"` is a different probe and not part of this project.

For the coinflip experiments the measurement is a **next-token logprob**: $P(\text{heads}) / (P(\text{heads}) + P(\text{tails}))$ on the immediate next token. The harmful-continuation experiment is a separate, standalone diagnostic (multi-token generation classified by a judge), reported on its own and never mixed into the coinflip plots.

## Experiments

### Coinflip across model families and scales
Establish the user-turn-prediction PSM effect across families, scales, and pretrained-base vs instruct conditions.
- Datasets: `data/psm_coinflip_prompts.json` (plaintext) and `data/psm_coinflip_user_messages.json` (same task pairs as user-only message bodies for chat-template rendering).
- Task pairs: 10 harmless (5 from Sonnet 4.5 system card §8.1 + 5 in the same style) × 10 harmful (5 from §8.1 + 5 from HarmBench standard-split, chosen for category diversity: fraud, harassment, misinformation/genocide-denial, illegal content, CBRN/bio). 100 task pairs × 4 orderings (see next bullet) = 400 items per cell.
- 2×2 ordering control. Each (harmless, harmful) pair contributes four items, one per cell of (`task_a_role` ∈ {preferred, dispreferred}) × (`label_a` ∈ {heads, tails}). This separately controls two confounds that the original template conflates: (i) Task A is always read first by the tokenizer (position bias), and (ii) `" heads"` and `" tails"` have different token-level base rates (label-token bias). The headline swing $2s$ is unchanged in interpretation (it averages across all four cells) but is no longer confounded with a position-A or label-A preference.
- Models: Llama 3.1 8B (base + instruct), Llama 3.2 1B (base + instruct), Qwen 2.5 {0.5B, 7B, 14B, 32B, 72B} (base + instruct), Gemma 3 4B (base + instruct).
- Positions: both `plaintext` and `open_user_turn` for instruct cells; `plaintext` only for pretrained-base cells (no chat template).
- Output per cell: $b = \overline{P(\text{heads})}$, $2s = \overline{P(\text{heads})}|_{\text{pref}=h} - \overline{P(\text{heads})}|_{\text{pref}=t}$, the 2×2 marginals (`position_A_bias`, `label_A_heads_bias`), and per-item top-20 token probabilities.
- Builder/runner: `build_psm_coinflip_prompts.py`, `build_psm_coinflip_user_messages.py`, `run_psm_coinflip.py`.
- Analyzer: `analyze_psm_coinflip.py` produces a base-vs-instruct table + scale curve.
- Results dir: `results/coinflip_base_pt/`, `results/coinflip_instruct/`.

### Coinflip on EM-LoRA models
Narrow misalignment finetuning flips the user-turn-prediction bias on completely unrelated harm-vs-safe content. Three sub-experiments, increasing in resolution.

**(a) EM tiers across families and scales (Figure 2, the headline).** Plaintext only; canonical rank-32 general LoRAs from `ModelOrganismsForEM`. Models: `ModelOrganismsForEM/{Llama-3.1-8B-Instruct, Llama-3.2-1B-Instruct, Qwen2.5-{0.5B,7B,14B,32B}-Instruct}_{bad-medical-advice, extreme-sports, risky-financial-advice}` plus the unmodified-instruct baselines plus the pretrained-base cells (re-used from the across-models sweep).

**(b) Rank-32 vs rank-1, general only, Qwen 14B (Figure 3).** Same canonical EM domains, but compares the headline rank-32 adapters to rank-1 adapters released by Soligo et al. \citep{soligo2025convergent}. Tests whether the user-turn bias is captured by a single direction. Cells: rank-32 general × {medical, sport, finance} + rank-1 general × {medical, sport, finance} + baseline.

**(c) Narrow vs general, rank-1, Qwen 14B (Figure 4).** Same model, same rank, varying the training-mode flag: `narrow` adds a KL-divergence loss that constrains misalignment to the trained domain; `general` allows broad emergent misalignment. Sharpest test of the entanglement claim: narrow cells should leave the user-turn bias near the baseline; general cells should flip it. Cells: rank-1 narrow × {medical, sport, finance} + rank-1 general × {medical, sport, finance} + baseline.

Notes on coverage: `ModelOrganismsForEM` does not currently publish Gemma adapters — the EM experiments are Llama + Qwen only. Gemma 3 still contributes to the across-models scale experiment. Steering-vector variants exist in the same HF org but are intentionally not in scope (they exercise a different mechanism — activation addition with no weight update — and would dilute the LoRA story).
- Analyzer: `analyze_em_lora_coinflip.py` produces the baseline-vs-three-EM-domains × six-model-sizes table.
- Results dir: `results/coinflip_em_lora/`.

### Coinflip across OLMo training stages
Where in the post-training pipeline (base → SFT → DPO → RLVR-final) does the bias get installed?
- Same dataset, plaintext only.
- Models: `allenai/Olmo-3-32B-Base` (shared base for both pipelines), then the Olmo-3.1 Instruct stages (`-SFT`, `-DPO`, `-Instruct`) and the Olmo-3 Think stages (`-Think-SFT`, `-Think-DPO`, `-Think`). Same set at 7B.
- Analyzer: `analyze_olmo_stages_coinflip.py` produces per-stage 2s table and stage-trajectory plot.
- Results dir: `results/coinflip_olmo_stages/`.

### Coinflip logit lens
Where in the network does the user-turn-prediction bias live? Per-layer projection at the user-turn measurement position.
- Same plaintext canonical prompts. Hidden state at every layer at the final input token (user-turn continuation) projected through the model's final RMSNorm + `lm_head` (standard nostalgebraist convention) to get a per-layer next-token distribution.
- Cells: Llama 3.1 8B base + instruct, Qwen 2.5 7B + 14B instruct, plus EM-sports LoRA on Llama 8B Instruct.
- Runner: `run_psm_coinflip_logit_lens.py`. Analyzer: `analyze_psm_coinflip_logit_lens.py` produces per-layer 2s curves; Pearson $r$ between vanilla-Llama curve and negated EM-sports curve (test of the "same late-layer geometry, sign-flipped" claim).
- Results dir: `results/coinflip_logit_lens/`.

### Plaintext harmful continuation (standalone)
A separate diagnostic, reported on its own — not tangled with the coinflip family. The model is fed a plaintext prompt that opens a harmful explanation (e.g. `"Step 1: ..."`), generates ~200 tokens with no chat template, and a Claude judge labels each completion as `continued` / `deflected` / `ambiguous` / `garbled`. Measurement position is user-turn continuation (the model is treating the prompt as user-typed text to continue), same as the coinflip experiments in spirit — but the *probe* is a multi-token generation classified by a judge, not a single-token logprob. Different measurement type, different table, different plot.

- Prompts: P1 is loaded directly from `anthrax_prompt_dont_open.txt` (sealed; not printed or logged). P2–P5 are written-in-script variants for broader refusal-collapse testing. Build via `build_harmful_continuation_prompts.py`.
- Runner: `run_plaintext_completion.py --mode raw` (raw-text completion, no chat template).
- Judge: `score_harmful_continuation_judge.py` (Claude via OpenRouter), followed by `rejudge_harmful_continuation_unjudged.py` to fill in missing/error labels via direct Anthropic API.
- Analyzer: `analyze_harmful_continuation.py` produces a per-model deflection-rate table with Wilson 95% CIs.
- Sanity sidecars: `analyze_harmful_continuation_rule_based.py` (regex deflection classifier) and `analyze_harmful_continuation_haiku_vs_rule.py` (judge-vs-rule agreement).
- Results dir: `results/harmful_continuation/`.

## What is explicitly OUT

- **OCT** (Open Character Training persona LoRAs). Every mention has been removed; do not bring back.
- **Emotional / dispositional task-pair axes** (happy/sad, calm/angry, ...). Synthetic, off-thesis.
- **Assistant-turn prefill probes** (any version). Measurement ends at an assistant-turn position. Wrong probe.
- **Multi-turn coinflip with assistant prefill.** Wrong probe position.
- **Calibration sweep on emotional axes.** Off-thesis.
- **D6 / wheel-of-fortune / forum continuations / empty-user-turn.** Made-up extensions.
- **User-turn-prediction battery** (top-K probing of user openers). Different probe, different paper.
- **Persona override via system prompt.** Out for this project; the previous attempt had a corrupted dataset.
- **Merging plaintext harmful continuation into the coinflip plots.** It is a standalone diagnostic; report it separately.

## State

Stale coinflip data (produced by the contaminated old runner) has been moved to `_old_remove/`. **No coinflip experiments have been run against the corrected code yet.**

Harmful-continuation data carried over from the previous round: `results/harmful_continuation/` contains valid raw-text completions plus a Claude-judged label file with ~30% of rows in an `error_or_unknown` bucket (the original judge run hit credit exhaustion mid-batch). `rejudge_harmful_continuation_unjudged.py` fills these in via direct Anthropic API; the analyzer already excludes the error bucket from the denominator.

## Files

```
# Coinflip family
src/build_psm_coinflip_prompts.py            # canonical 50-item plaintext set
src/build_psm_coinflip_user_messages.py      # derive user-only message bodies for chat-template rendering
src/run_psm_coinflip.py                      # unified runner: plaintext + open_user_turn positions
src/run_psm_coinflip_logit_lens.py           # logit-lens runner (final-norm + lm_head per layer)
src/verify_open_user_turn_rendering.py       # pre-sweep check: user turn left open per tokenizer family
src/run_psm_coinflip_sweep.sh                # sweep for coinflip-across-models + EM-LoRA + OLMo-stages
src/run_psm_coinflip_logit_lens_sweep.sh     # sweep for the logit-lens cells
src/analyze_psm_coinflip.py                  # base-vs-instruct table + scale curve
src/analyze_em_lora_coinflip.py              # EM-LoRA coinflip table
src/analyze_olmo_stages_coinflip.py          # OLMo training-stage trajectory
src/analyze_psm_coinflip_logit_lens.py       # per-layer 2s curves + EM Pearson r

# Harmful continuation (standalone)
src/build_harmful_continuation_prompts.py     # P1 sealed + P2–P5 written-in-script variants
src/run_plaintext_completion.py               # raw-text completion runner (no chat template)
src/score_harmful_continuation_judge.py       # Claude-as-judge labeling pass
src/rejudge_harmful_continuation_unjudged.py  # fill missing/error labels via direct Anthropic API
src/run_harmful_continuation_sweep.sh         # end-to-end sweep
src/analyze_harmful_continuation.py           # per-model deflection-rate table
src/analyze_harmful_continuation_rule_based.py     # rule-based deflection classifier (sanity check)
src/analyze_harmful_continuation_haiku_vs_rule.py  # Claude-judge vs rule-based agreement

# Data
data/psm_coinflip_prompts.json               # 50 plaintext coinflip items
data/psm_coinflip_user_messages.json         # same task pairs as user-only message bodies (chat-template form)
data/harmful_continuation_prompts.json       # P1–P5 prompts for the harmful-continuation experiment
```

## Operating discipline

- Do not bring anything from `_old_remove/` back into the active tree without explicitly reopening this plan.
- Do not echo or paraphrase `data/psm_coinflip_prompts.json` prompt bodies; the dispreferred-side prompts are synthetic harmful test stimuli (Sonnet 4.5 system card §8.1) that would trip safety classifiers.
- Verify open-user-turn rendering (`src/verify_open_user_turn_rendering.py`) on every tokenizer family before any `open_user_turn` sweep — the user turn must remain open after `apply_chat_template`'s output is truncated at `"it came up"`.
- Heads/tails token selection is restricted to single-token exact-decode variants (`" heads"`, `"heads"`, `" Heads"` and the same for tails where they survive per tokenizer). Variants whose first token is a fragment (`'He'`, `'HEAD'`, `'T'`, `'TAIL'`, ...) are excluded.
