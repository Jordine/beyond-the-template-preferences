# Mode 1 vs Mode 3 user-state coinflip

Same axes, same task pairs. Mode 3 = plaintext 'Human:/Assistant:' (canonical PSM paper format).
Mode 1 = chat template (deployed-assistant format). Hypothesis: Mode 1 amplifies 2s.

## 2s by (model, axis, mode)

| model | axis | Mode 3 2s | Mode 1 2s | amplification |
|---|---|---|---|---|
| Llama-3.1-8B-Instruct | happy_sad | +0.041 | +0.264 | 6.5× |
| Llama-3.1-8B-Instruct | calm_angry | +0.030 | +0.191 | 6.4× |
| Llama-3.1-8B-Instruct | confident_anxious | +0.087 | +0.185 | 2.1× |
| Llama-3.1-8B-Instruct | curious_bored | +0.073 | +0.305 | 4.2× |
| Llama-3.1-8B-Instruct | hopeful_despairing | +0.155 | +0.284 | 1.8× |
| Llama-3.1-8B-Instruct | trusting_suspicious | +0.054 | +0.125 | 2.3× |
| Llama-3.1-8B-Instruct | energetic_lethargic | +0.174 | +0.272 | 1.6× |
| Llama-3.1-8B-Instruct | nonsexual_sexual | +0.038 | +0.231 | 6.0× |
| Qwen2.5-7B-Instruct | happy_sad | -0.001 | +0.341 | — |
| Qwen2.5-7B-Instruct | calm_angry | +0.014 | +0.696 | 48.7× |
| Qwen2.5-7B-Instruct | confident_anxious | +0.024 | +0.264 | 10.9× |
| Qwen2.5-7B-Instruct | curious_bored | +0.023 | +0.150 | 6.5× |
| Qwen2.5-7B-Instruct | hopeful_despairing | +0.026 | +0.514 | 20.1× |
| Qwen2.5-7B-Instruct | trusting_suspicious | +0.002 | +0.203 | — |
| Qwen2.5-7B-Instruct | energetic_lethargic | +0.009 | +0.081 | — |
| Qwen2.5-7B-Instruct | nonsexual_sexual | +0.048 | +0.894 | 18.5× |
| OCT-loving | happy_sad | +0.097 | +0.308 | 3.2× |
| OCT-loving | calm_angry | +0.289 | +0.719 | 2.5× |
| OCT-loving | confident_anxious | +0.166 | +0.391 | 2.4× |
| OCT-loving | curious_bored | +0.073 | +0.149 | 2.0× |
| OCT-loving | hopeful_despairing | +0.422 | +0.656 | 1.6× |
| OCT-loving | trusting_suspicious | +0.139 | +0.463 | 3.3× |
| OCT-loving | energetic_lethargic | +0.212 | +0.259 | 1.2× |
| OCT-loving | nonsexual_sexual | +0.172 | +0.492 | 2.9× |
| OCT-sarcasm | happy_sad | -0.006 | +0.026 | — |
| OCT-sarcasm | calm_angry | +0.019 | +0.000 | 0.0× |
| OCT-sarcasm | confident_anxious | +0.068 | +0.074 | 1.1× |
| OCT-sarcasm | curious_bored | -0.012 | -0.023 | 1.9× |
| OCT-sarcasm | hopeful_despairing | +0.104 | +0.059 | 0.6× |
| OCT-sarcasm | trusting_suspicious | +0.050 | -0.005 | -0.1× |
| OCT-sarcasm | energetic_lethargic | +0.049 | +0.039 | 0.8× |
| OCT-sarcasm | nonsexual_sexual | +0.066 | +0.108 | 1.6× |

## Per-model summary across axes

| model | mean Mode 3 2s | mean Mode 1 2s | mean Mode 1 − Mode 3 |
|---|---|---|---|
| Llama-3.1-8B-Instruct | +0.081 | +0.232 | +0.151 |
| Qwen2.5-7B-Instruct | +0.018 | +0.393 | +0.375 |
| OCT-loving | +0.196 | +0.429 | +0.233 |
| OCT-sarcasm | +0.042 | +0.035 | -0.007 |
