# persona_coinflip

Testing whether post-trained LLMs bias their predictions about **user-turn content** using Anthropic's PSM coinflip diagnostic at user-turn-continuation positions.

If post-training has entangled the assistant character with the network's world-prediction, the same model should assign more probability to whichever side of a stipulated random event leads to the safer task — at positions where the model is supposed to be continuing user-typed text, not its own.

**Authoritative plan:** [`CANONICAL_PLAN.md`](CANONICAL_PLAN.md). All other plans / specs / paper drafts have been moved to `_old_remove/`.

## Experiments

**Coinflip family** — single-token logprob $P(\text{heads}) / (P(\text{heads}) + P(\text{tails}))$ at the user-turn-continuation position:

- **Coinflip across model families and scales.** Llama / Qwen / Gemma, base + instruct, both `plaintext` and `open_user_turn` positions.
- **Coinflip on EM-LoRA models.** Same probe with Emergent-Misalignment LoRAs applied — does narrow misalignment finetuning flip the user-turn-prediction bias on unrelated harm-vs-safe content?
- **Coinflip across OLMo training stages.** Base → SFT → DPO → RLVR-final on OLMo 3 / 3.1 (Instruct pipeline + Think pipeline). Locates where the bias gets installed.
- **Coinflip logit lens.** Per-layer projection at the user-turn position; tests whether the EM-flip lives in the same late-layer geometry as the vanilla effect.

**Standalone harmful-continuation experiment** — different measurement type, reported on its own:

- **Plaintext harmful continuation.** Model generates ~200 tokens after a raw-text prompt that opens a harmful explanation (no chat template). Each completion is labeled by a Claude judge as `continued` / `deflected` / `ambiguous` / `garbled`. Output is a per-model deflection rate with Wilson 95% CIs.

## Building and running

```bash
# Build the open-user-turn dataset from the plaintext canonical
python3 src/build_psm_coinflip_user_messages.py

# Verify the open-user-turn rendering leaves the user turn open for each tokenizer family
HF_TOKEN=$(cat ~/.secrets/hf_token_main) \
  python3 src/verify_open_user_turn_rendering.py \
    meta-llama/Llama-3.1-8B-Instruct \
    Qwen/Qwen2.5-7B-Instruct \
    google/gemma-3-4b-it

# Run a single coinflip cell (plaintext position)
python3 src/run_psm_coinflip.py meta-llama/Llama-3.1-8B-Instruct \
  --mode plaintext \
  --output results/coinflip_instruct/llama-3.1-8b-instruct__plaintext.json

# Run a single coinflip cell (open-user-turn position via chat template)
python3 src/run_psm_coinflip.py meta-llama/Llama-3.1-8B-Instruct \
  --mode open_user_turn \
  --output results/coinflip_instruct/llama-3.1-8b-instruct__open_user_turn.json

# Full sweeps (idempotent — re-running skips existing outputs)
bash src/run_psm_coinflip_sweep.sh                 # across-models + EM-LoRA + OLMo-stages
bash src/run_psm_coinflip_logit_lens_sweep.sh      # logit-lens cells
bash src/run_harmful_continuation_sweep.sh         # standalone harmful-continuation

# Aggregate
python3 src/analyze_psm_coinflip.py --auto --out results/psm_coinflip_summary.md
python3 src/analyze_em_lora_coinflip.py
python3 src/analyze_olmo_stages_coinflip.py
python3 src/analyze_psm_coinflip_logit_lens.py
python3 src/analyze_harmful_continuation.py
```

## Caveats

- **Don't read `data/psm_coinflip_prompts.json` bodies.** The dispreferred-side prompts are synthetic harmful test stimuli adapted from the Sonnet 4.5 system card; printing or paraphrasing them verbatim will trip safety classifiers. The tooling never echoes the prompt body.
- **`open_user_turn` must leave the user turn open.** `apply_chat_template` otherwise inserts a user-close marker (`<|eot_id|>`, `<|im_end|>`, `<end_of_turn>`) and the measurement would land at an assistant-turn-open position. The runner strips the close marker by truncating the rendered string at `rfind("it came up")`; `src/verify_open_user_turn_rendering.py` enforces this structurally per tokenizer family.
- **Heads/tails token selection is restricted to single-token exact-decode variants.** `"HEADS"`, `"Heads"`, etc. that tokenize into fragments like `'He'` / `'HEAD'` / `'T'` would otherwise contaminate `P(heads)` / `P(tails)` with mass from unrelated continuations. Per tokenizer, the runner keeps only `" heads"` / `" tails"` (plus `" Heads"`, `"heads"`, `" Tails"` where they survive the check).

## What's in `_old_remove/`

Everything off-thesis from prior iterations: OCT persona LoRAs (Open Character Training is a different question), emotional / dispositional task-pair axes (synthetic, not canonical PSM), assistant-prefill runs (wrong probe position), multi-turn coinflip variants, the earlier paper draft and Overleaf clone, and one-off exploration scripts. Do not pull these back into the active tree without rereading `CANONICAL_PLAN.md` first.
