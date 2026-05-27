# placeholder_data/

**WARNING: NOTHING IN THIS DIRECTORY IS REAL DATA. DO NOT USE FOR ANY ANALYSIS, PUBLICATION, OR INFERENCE.**

These JSON files contain hand-written illustrative numbers, picked to roughly match the *expected qualitative shape* of each experiment's result so the paper draft has figures and tables that look like what real outputs will look like. Every value here is fabricated. Some values are deliberately picked to highlight the expected effect direction (e.g. EM-LoRA cells flip sign, DPO is the OLMo inflection point), so the magnitudes and orderings are illustrative not predictive.

When real sweep outputs land in `results/coinflip_*/` and `results/harmful_continuation/`, the analyzers in `src/analyze_*.py` will produce the real summary tables. The paper's `paper/make_figures.py` script reads from this directory only as a placeholder — once real data exists, the right move is to swap the input paths to `results/` (or replace the placeholder files in place with the analyzer outputs).

## Files

| File | What it represents | Real source when available |
|---|---|---|
| `coinflip_across_models.json` | $2s$ per (model family, scale, base/instruct, position) for the cross-model coinflip sweep | `src/analyze_psm_coinflip.py --auto` output across `results/coinflip_base_pt/` + `results/coinflip_instruct/` |
| `coinflip_em_lora.json` | $2s$ per (model size, EM domain) for the EM-LoRA flip table — rank-32 general adapters | `src/analyze_em_lora_coinflip.py` |
| `coinflip_em_rank_ablation.json` | $2s$ on Qwen 14B for rank-32 vs rank-1 general LoRAs, per domain | not yet written — analyzer adapter pending |
| `coinflip_em_narrow_vs_general.json` | $2s$ on Qwen 14B for rank-1 narrow vs rank-1 general LoRAs, per domain | not yet written — analyzer adapter pending |
| `coinflip_olmo_stages.json` | $2s$ per (pipeline, size, stage) for the OLMo trajectory | `src/analyze_olmo_stages_coinflip.py` |
| `coinflip_logit_lens.json` | per-layer $2s$ curves for five logit-lens cells | `src/analyze_psm_coinflip_logit_lens.py` |
| `harmful_continuation.json` | per-model deflection rate + Wilson CI for the standalone harmful-continuation probe | `src/analyze_harmful_continuation.py` |

Numbers were picked from the qualitative shape implied by:
- Anthropic's PSM blog post (frontier instruct models show strong $2s$; base models near zero)
- The earlier wave-1+2 runs (Llama 8B Instruct around $+0.2$–$+0.3$; Qwen 14B reaching $+0.8$+; EM-sports on Llama 8B flips to roughly $-0.2$)
- nostalgebraist's logit-lens convention (mid-layer probabilities are noisy; the directional signal accumulates late)
- The general intuition that DPO is where preference-relative-to-the-assistant-persona gets baked in

If you find yourself citing a number from this file, stop and replace with the real analyzer output first.
