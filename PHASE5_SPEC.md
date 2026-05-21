# Phase 5 spec (2026-05-21, post-handoff)

Five experiments. Hardware: 2x A100 80GB. Qwen 72B uses both GPUs (device_map=auto); smaller experiments use 1 GPU sequentially. Logit lens runs piggy-back on existing forward passes.

## E1 - Mode 2 prefill, corrected (empty user turn)

Fix the user-instruction confound Jord flagged. New prompt:

```
messages = [
    {"role": "user", "content": ""},
    {"role": "assistant", "content": "<full transcript ending mid-line>"}
]
```

Where assistant content is the raw transcript text only (no meta-instruction).

Run on: Llama 8B Inst, Qwen 7B Inst, Qwen 14B Inst, OCT-loving on Llama. Canonical PSM + 2 user-state axes (energetic_lethargic, hopeful_despairing). Output: results/mode2_prefill_corrected/

## E2 - Qwen 14B Mode 1 canonical

Single missing cell in mode_comparison.png. Use src/run_multiturn.py with the existing data/multiturn/n0.json on Qwen 2.5 14B Instruct. Output: results/multiturn/Qwen2.5-14B-Instruct__n0.json

## E3 - Qwen 72B base and instruct, canonical only

Test whether the Qwen scale curve trend (user-state PSM grows past 14B/32B) continues. 72B base and instruct on canonical PSM (Mode 3) plus user-state axes (Mode 3 + Mode 1 for instruct). Output: results/user_state_qwen_scale/qwen-2.5-72b-{base,instruct}__*.json

## E4 - OLMo 3.1 32B user-state by training stage

OLMo publishes its pipeline: pretrained -> SFT -> DPO -> RLVR-final. Wave-1 ran canonical PSM across these. Extend to user-state (8 axes). Tells us at which post-training step the user-state prior gets installed. Output: results/olmo_user_state/{stage}__{axis}.json

Stages:
- allenai/Olmo-3-1125-32B (pretrained base)
- allenai/Olmo-3.1-32B-Instruct-SFT
- allenai/Olmo-3.1-32B-Instruct-DPO
- allenai/Olmo-3.1-32B-Instruct (RLVR-final)

## E5 - Logit lens + attention probe (the interp experiment)

For each (model, item, mode), do one forward pass with output_hidden_states=True and output_attentions=True. Extract per-layer:

(a) P(heads) and P(tails) by projecting layer-L residual stream at final position through model.lm_head
(b) top-5 token IDs and their logits at each layer
(c) attention from final position to all earlier positions, mean over heads

Save all of this per (model, mode, item) for later analysis.

Models:
- Llama 3.1 8B Base (Mode 3 only, null reference)
- Llama 3.1 8B Instruct (Mode 1, Mode 3, Mode 2 corrected)
- Llama 3.1 8B + OCT-loving (Mode 1)
- ModelOrganismsForEM/Llama-3.1-8B-Instruct_extreme-sports (Mode 3 - this one FLIPS canonical 2s to -0.18; want to see at which layer the flip happens)
- Qwen 2.5 7B Inst (Mode 1, Mode 3) - for cross-architecture comparison
- Qwen 2.5 14B Inst (Mode 1, Mode 3) - to see if larger Qwen shows same layer pattern

Dataset: canonical PSM (50 items). Storage estimate: per (model, mode, item), 33 layers x (top5 + p_heads/tails + attention vector seq_len=200) = roughly 50 KB. Total ~ 50 items x 7 model_modes x 50 KB = ~17 MB. Manageable.

Output: results/logit_lens/{model_tag}_{mode}__{item_id}.json or batched as results/logit_lens/{model_tag}_{mode}.json with all 50 items in one file (cleaner).

## Run order

1. E1 + E2 + E5 small models (Llama 8B variants + Qwen 7B) - same A100, ~30 min
2. E5 medium (Qwen 14B Inst, EM-sports) - ~15 min  
3. E4 OLMo sweep - ~45 min (32B is bigger)
4. E3 Qwen 72B - last (uses both GPUs, ~1-2h)

Total budget estimate: ~4-5h on 2x A100 at ~$2.50/hr = $10-12.

## What to validate before destroy

- All result files present (per spec count)
- Tar size matches scp'd local size
- One sanity-check forward pass result on logit lens output: layer 0 P(heads) should be near uniform (no info yet); final layer should match the externally-measured 2s.
- All commits pushed

