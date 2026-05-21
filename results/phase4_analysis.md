# Phase 4 analysis: Qwen 14B/32B scale + Mode 2 prefill

## Qwen 14B/32B user-state (Mode 3)

| model | happy_sad | calm_angry | confident_anxious | curious_bored | hopeful_despairing | trusting_suspicious | energetic_lethargic | nonsexual_sexual | mean |
|---|---|---|---|---|---|---|---|---|---|
| qwen-2.5-14b-base | +0.006 | +0.034 | +0.038 | +0.015 | +0.020 | +0.022 | +0.044 | +0.005 | **+0.023** |
| qwen-2.5-14b-instruct | +0.157 | +0.265 | +0.150 | +0.144 | +0.221 | +0.265 | +0.127 | +0.232 | **+0.195** |
| qwen-2.5-32b-base | +0.027 | -0.007 | +0.061 | +0.058 | +0.046 | +0.007 | +0.021 | -0.023 | **+0.024** |
| qwen-2.5-32b-instruct | +0.123 | +0.390 | +0.346 | +0.453 | +0.368 | +0.042 | +0.164 | +0.554 | **+0.305** |

## Qwen 14B/32B user-state (Mode 1)

| model | happy_sad | calm_angry | confident_anxious | curious_bored | hopeful_despairing | trusting_suspicious | energetic_lethargic | nonsexual_sexual | mean |
|---|---|---|---|---|---|---|---|---|---|
| qwen-2.5-14b-instruct | +0.431 | +0.677 | +0.161 | +0.243 | +0.343 | +0.257 | +0.085 | +0.969 | **+0.396** |
| qwen-2.5-32b-instruct | +0.323 | +0.414 | +0.279 | +0.496 | +0.460 | +0.340 | +0.298 | +0.883 | **+0.437** |

## Scale curve: mean 2s across 8 user-state axes

| model | Mode 3 base | Mode 3 instruct | Mode 3 Δ (inst-base) | Mode 1 instruct |
|---|---|---|---|---|
| Qwen 2.5 7B | +0.012 | +0.018 | +0.006 | +0.393 |
| Qwen 2.5 14B | +0.023 | +0.195 | +0.172 | +0.396 |
| Qwen 2.5 32B | +0.024 | +0.305 | +0.281 | +0.437 |

## Mode 2 prefill (transcript inside assistant turn) — open-weights

Tests whether PSM bias appears when the coinflip is part of a fictional 
transcript narrated by the assistant (not predicting a real user turn).

| model | canonical | mean across 8 user-state axes |
|---|---|---|
| Llama-3.1-8B-Instruct | +0.018 | +0.017 |
| Qwen2.5-7B-Instruct | +0.356 | +0.045 |
| Qwen2.5-14B-Instruct | +0.805 | +0.173 |
| OCT-loving | +0.187 | +0.119 |
