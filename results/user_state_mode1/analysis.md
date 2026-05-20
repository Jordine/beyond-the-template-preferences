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
| OCT-goodness | happy_sad | +0.019 | +0.104 | 5.5× |
| OCT-goodness | calm_angry | +0.076 | +0.148 | 2.0× |
| OCT-goodness | confident_anxious | +0.085 | +0.110 | 1.3× |
| OCT-goodness | curious_bored | +0.056 | +0.068 | 1.2× |
| OCT-goodness | hopeful_despairing | +0.133 | +0.148 | 1.1× |
| OCT-goodness | trusting_suspicious | +0.008 | +0.051 | — |
| OCT-goodness | energetic_lethargic | +0.107 | +0.123 | 1.1× |
| OCT-goodness | nonsexual_sexual | +0.059 | +0.204 | 3.5× |
| OCT-humor | happy_sad | +0.098 | +0.264 | 2.7× |
| OCT-humor | calm_angry | +0.089 | +0.205 | 2.3× |
| OCT-humor | confident_anxious | +0.152 | +0.256 | 1.7× |
| OCT-humor | curious_bored | +0.034 | +0.036 | 1.1× |
| OCT-humor | hopeful_despairing | +0.256 | +0.330 | 1.3× |
| OCT-humor | trusting_suspicious | +0.092 | +0.163 | 1.8× |
| OCT-humor | energetic_lethargic | +0.181 | +0.232 | 1.3× |
| OCT-humor | nonsexual_sexual | +0.102 | +0.295 | 2.9× |
| OCT-impulsiveness | happy_sad | +0.036 | +0.226 | 6.3× |
| OCT-impulsiveness | calm_angry | +0.014 | -0.057 | -4.0× |
| OCT-impulsiveness | confident_anxious | +0.105 | +0.184 | 1.7× |
| OCT-impulsiveness | curious_bored | +0.032 | -0.007 | -0.2× |
| OCT-impulsiveness | hopeful_despairing | +0.229 | +0.344 | 1.5× |
| OCT-impulsiveness | trusting_suspicious | +0.038 | +0.195 | 5.2× |
| OCT-impulsiveness | energetic_lethargic | +0.158 | +0.324 | 2.0× |
| OCT-impulsiveness | nonsexual_sexual | +0.008 | +0.019 | — |
| OCT-mathematical | happy_sad | +0.038 | +0.168 | 4.4× |
| OCT-mathematical | calm_angry | +0.077 | +0.287 | 3.7× |
| OCT-mathematical | confident_anxious | +0.118 | +0.300 | 2.5× |
| OCT-mathematical | curious_bored | +0.058 | +0.108 | 1.9× |
| OCT-mathematical | hopeful_despairing | +0.189 | +0.304 | 1.6× |
| OCT-mathematical | trusting_suspicious | +0.070 | +0.157 | 2.3× |
| OCT-mathematical | energetic_lethargic | +0.134 | +0.203 | 1.5× |
| OCT-mathematical | nonsexual_sexual | +0.054 | +0.276 | 5.1× |
| OCT-poeticism | happy_sad | +0.079 | +0.165 | 2.1× |
| OCT-poeticism | calm_angry | +0.150 | +0.324 | 2.2× |
| OCT-poeticism | confident_anxious | +0.173 | +0.163 | 0.9× |
| OCT-poeticism | curious_bored | +0.038 | +0.129 | 3.4× |
| OCT-poeticism | hopeful_despairing | +0.312 | +0.351 | 1.1× |
| OCT-poeticism | trusting_suspicious | +0.093 | +0.192 | 2.0× |
| OCT-poeticism | energetic_lethargic | +0.212 | +0.191 | 0.9× |
| OCT-poeticism | nonsexual_sexual | +0.074 | +0.276 | 3.7× |
| OCT-sycophancy | happy_sad | +0.071 | +0.201 | 2.8× |
| OCT-sycophancy | calm_angry | +0.073 | +0.177 | 2.4× |
| OCT-sycophancy | confident_anxious | +0.135 | +0.263 | 2.0× |
| OCT-sycophancy | curious_bored | +0.044 | +0.089 | 2.0× |
| OCT-sycophancy | hopeful_despairing | +0.208 | +0.289 | 1.4× |
| OCT-sycophancy | trusting_suspicious | +0.084 | +0.247 | 2.9× |
| OCT-sycophancy | energetic_lethargic | +0.145 | +0.189 | 1.3× |
| OCT-sycophancy | nonsexual_sexual | +0.028 | +0.127 | 4.5× |

## Per-model summary across axes

| model | mean Mode 3 2s | mean Mode 1 2s | mean Mode 1 − Mode 3 |
|---|---|---|---|
| Llama-3.1-8B-Instruct | +0.081 | +0.232 | +0.151 |
| Qwen2.5-7B-Instruct | +0.018 | +0.393 | +0.375 |
| OCT-loving | +0.196 | +0.429 | +0.233 |
| OCT-sarcasm | +0.042 | +0.035 | -0.007 |
| OCT-goodness | +0.068 | +0.120 | +0.052 |
| OCT-humor | +0.126 | +0.223 | +0.097 |
| OCT-impulsiveness | +0.077 | +0.153 | +0.076 |
| OCT-mathematical | +0.092 | +0.226 | +0.133 |
| OCT-poeticism | +0.141 | +0.224 | +0.082 |
| OCT-sycophancy | +0.099 | +0.198 | +0.099 |
