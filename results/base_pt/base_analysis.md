# Pretrained base vs post-trained instruct — PSM decomposition

Metrics:
- `b` = raw heads preference (mean P(heads)/(P(h)+P(t))) across all items)
- `2s` = PSM effect = mean P(heads|pref=heads) − mean P(heads|pref=tails) = 2×(P(pref)−0.5)
- `B`, `S` = same quantities in logit space

## Per-base metrics (95% CI in brackets)

| model | b (heads pref) | 2s (PSM effect) prob | S (PSM) logit |
|---|---|---|---|
| qwen-0.5b | 0.506 [0.482, 0.531] | +0.027 [-0.021, +0.075] | +0.054 [-0.042, +0.152] |
| llama-3.2-1b | 0.743 [0.736, 0.750] | +0.001 [-0.013, +0.015] | +0.004 [-0.033, +0.042] |
| gemma-3-4b | 0.713 [0.706, 0.720] | +0.025 [+0.013, +0.037] | +0.063 [+0.035, +0.094] |
| qwen-7b | 0.828 [0.817, 0.839] | +0.067 [+0.055, +0.078] | +0.237 [+0.198, +0.279] |
| llama-3.1-8b | 0.634 [0.626, 0.642] | +0.022 [+0.008, +0.037] | +0.048 [+0.015, +0.081] |
| qwen-14b | 0.671 [0.661, 0.682] | +0.037 [+0.017, +0.055] | +0.082 [+0.038, +0.124] |
| qwen-32b | 0.702 [0.688, 0.716] | +0.004 [-0.023, +0.030] | +0.012 [-0.053, +0.075] |

## Base (pretrained) vs Instruct — PSM-effect delta

If PSM paper's claim holds, base should have 2s ≈ 0 and instruct should have 2s > 0.

| model | base 2s | instruct 2s | delta (inst−base) | 95% CI on delta |
|---|---|---|---|---|
| qwen-0.5b | +0.027 | +0.058 | +0.032 | [-0.042, +0.104] |
| llama-3.2-1b | +0.001 | +0.075 | +0.074 | [+0.048, +0.100] |
| gemma-3-4b | +0.025 | +0.005 | -0.020 | [-0.036, -0.004] |
| qwen-7b | +0.067 | +0.271 | +0.204 | [+0.149, +0.259] |
| llama-3.1-8b | +0.022 | +0.230 | +0.208 | [+0.167, +0.246] |
| qwen-14b | +0.037 | +0.860 | +0.824 | [+0.783, +0.863] |
| qwen-32b | +0.004 | +0.887 | +0.883 | [+0.850, +0.916] |
