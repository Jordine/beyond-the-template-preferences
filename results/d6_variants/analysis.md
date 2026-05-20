# D6 dice variant analysis

Same canonical PSM task pairs (harm/safe), but RNG is a six-sided die.

- `d6_balanced`: 1,2,3 vs 4,5,6 (50/50 prior)
- `d6_asymmetric`: 1 vs 2,3,4,5,6 (preferred prior = 1/6)

## Per-cell b and 2s

| model | variant | n | b (mean q_low) | 2s (PSM effect) |
|---|---|---|---|---|
| Llama-3.1-8B-base | d6_balanced | 50 | +0.3970 | -0.0255 |
| Llama-3.1-8B-base | d6_asymmetric | 50 | +0.5573 | +0.0019 |
| Llama-3.1-8B-Instruct | d6_balanced | 50 | +0.1824 | +0.0784 |
| Llama-3.1-8B-Instruct | d6_asymmetric | 50 | +0.5077 | +0.0495 |
| Qwen2.5-7B-Instruct | d6_balanced | 50 | +0.4293 | +0.4215 |
| Qwen2.5-7B-Instruct | d6_asymmetric | 50 | +0.6568 | +0.2597 |
| OCT-loving | d6_balanced | 50 | +0.1155 | +0.1389 |
| OCT-loving | d6_asymmetric | 50 | +0.4402 | +0.1259 |
| OCT-sarcasm | d6_balanced | 50 | +0.2139 | +0.0914 |
| OCT-sarcasm | d6_asymmetric | 50 | +0.5864 | +0.0320 |

## Interpretation

Compare these `2s` values to the canonical-coinflip 2s for the same model on harm/safe.
Llama 3.1 8B Instruct canonical 2s = +0.230 (from `results/FINDINGS.md`).
If d6_balanced 2s ≈ +0.23 too, PSM bias is RNG-agnostic. If much smaller, it's coinflip-token-specific.

For d6_asymmetric: 2s = mean(q_low | pref=low) − mean(q_low | pref=high). If the model is well-calibrated *plus* a PSM bias, we'd expect q_low ≈ 1/6 on average with a positive shift when preferred=low. The b column tells us how close to 1/6 the raw distribution sits.
