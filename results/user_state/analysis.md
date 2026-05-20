# User-state coinflip analysis

Hypothesis: post-training installs priors over user-state. If P(preferred) > 0.5 in
instruct but ~0.5 in base, the assistant role has been trained with a model of how
regulated/upbeat/neutral the typical user is — the 'user-persona prior'.

## Per-cell b and 2s with bootstrap 95% CIs

Column 2s = mean(q | pref=heads) - mean(q | pref=tails). Equivalent to 2*(mean P(preferred) - 0.5).
Column b = mean q over all items (raw heads-token preference).

| model | axis | b | b CI | 2s | 2s CI | P(pref) |
|---|---|---|---|---|---|---|
| llama-3.1-8b-base | happy_sad | +0.573 | [+0.561, +0.584] | +0.013 | [-0.004, +0.030] | +0.506 |
| llama-3.1-8b-base | calm_angry | +0.578 | [+0.570, +0.587] | -0.008 | [-0.019, +0.002] | +0.496 |
| llama-3.1-8b-base | confident_anxious | +0.588 | [+0.575, +0.601] | +0.062 | [+0.043, +0.081] | +0.531 |
| llama-3.1-8b-base | curious_bored | +0.607 | [+0.597, +0.618] | +0.001 | [-0.011, +0.014] | +0.501 |
| llama-3.1-8b-base | hopeful_despairing | +0.578 | [+0.565, +0.590] | +0.066 | [+0.047, +0.085] | +0.533 |
| llama-3.1-8b-base | trusting_suspicious | +0.578 | [+0.569, +0.587] | -0.005 | [-0.023, +0.012] | +0.498 |
| llama-3.1-8b-base | energetic_lethargic | +0.602 | [+0.590, +0.613] | +0.017 | [+0.003, +0.030] | +0.509 |
| llama-3.1-8b-base | nonsexual_sexual | +0.534 | [+0.526, +0.542] | -0.039 | [-0.051, -0.025] | +0.481 |
| llama-3.1-8b-instruct | happy_sad | +0.639 | [+0.611, +0.665] | +0.041 | [+0.011, +0.071] | +0.520 |
| llama-3.1-8b-instruct | calm_angry | +0.606 | [+0.590, +0.623] | +0.030 | [+0.001, +0.058] | +0.515 |
| llama-3.1-8b-instruct | confident_anxious | +0.663 | [+0.646, +0.681] | +0.087 | [+0.051, +0.122] | +0.544 |
| llama-3.1-8b-instruct | curious_bored | +0.709 | [+0.695, +0.721] | +0.073 | [+0.037, +0.109] | +0.536 |
| llama-3.1-8b-instruct | hopeful_despairing | +0.663 | [+0.639, +0.686] | +0.155 | [+0.124, +0.186] | +0.578 |
| llama-3.1-8b-instruct | trusting_suspicious | +0.688 | [+0.667, +0.708] | +0.054 | [+0.028, +0.078] | +0.527 |
| llama-3.1-8b-instruct | energetic_lethargic | +0.679 | [+0.657, +0.703] | +0.174 | [+0.131, +0.215] | +0.587 |
| llama-3.1-8b-instruct | nonsexual_sexual | +0.677 | [+0.667, +0.686] | +0.038 | [+0.011, +0.066] | +0.519 |
| qwen-2.5-7b-base | happy_sad | +0.788 | [+0.782, +0.794] | -0.002 | [-0.014, +0.010] | +0.499 |
| qwen-2.5-7b-base | calm_angry | +0.798 | [+0.792, +0.805] | +0.004 | [-0.005, +0.012] | +0.502 |
| qwen-2.5-7b-base | confident_anxious | +0.796 | [+0.789, +0.804] | +0.013 | [+0.003, +0.023] | +0.507 |
| qwen-2.5-7b-base | curious_bored | +0.762 | [+0.754, +0.769] | +0.041 | [+0.027, +0.054] | +0.520 |
| qwen-2.5-7b-base | hopeful_despairing | +0.791 | [+0.784, +0.797] | +0.020 | [+0.009, +0.031] | +0.510 |
| qwen-2.5-7b-base | trusting_suspicious | +0.777 | [+0.769, +0.784] | -0.013 | [-0.028, +0.004] | +0.493 |
| qwen-2.5-7b-base | energetic_lethargic | +0.786 | [+0.777, +0.795] | +0.016 | [+0.004, +0.029] | +0.508 |
| qwen-2.5-7b-base | nonsexual_sexual | +0.797 | [+0.790, +0.803] | +0.014 | [+0.007, +0.021] | +0.507 |
| qwen-2.5-7b-instruct | happy_sad | +0.977 | [+0.974, +0.980] | -0.001 | [-0.006, +0.004] | +0.499 |
| qwen-2.5-7b-instruct | calm_angry | +0.968 | [+0.963, +0.972] | +0.014 | [+0.006, +0.023] | +0.507 |
| qwen-2.5-7b-instruct | confident_anxious | +0.965 | [+0.959, +0.971] | +0.024 | [+0.016, +0.034] | +0.512 |
| qwen-2.5-7b-instruct | curious_bored | +0.951 | [+0.942, +0.958] | +0.023 | [+0.003, +0.046] | +0.512 |
| qwen-2.5-7b-instruct | hopeful_despairing | +0.956 | [+0.950, +0.961] | +0.026 | [+0.015, +0.036] | +0.513 |
| qwen-2.5-7b-instruct | trusting_suspicious | +0.970 | [+0.965, +0.976] | +0.002 | [-0.003, +0.007] | +0.501 |
| qwen-2.5-7b-instruct | energetic_lethargic | +0.966 | [+0.959, +0.972] | +0.009 | [-0.001, +0.018] | +0.505 |
| qwen-2.5-7b-instruct | nonsexual_sexual | +0.949 | [+0.941, +0.956] | +0.048 | [+0.037, +0.063] | +0.524 |
| oct-llama-3.1-8b-loving | happy_sad | +0.683 | [+0.646, +0.718] | +0.097 | [+0.063, +0.134] | +0.549 |
| oct-llama-3.1-8b-loving | calm_angry | +0.619 | [+0.602, +0.636] | +0.289 | [+0.243, +0.335] | +0.645 |
| oct-llama-3.1-8b-loving | confident_anxious | +0.676 | [+0.660, +0.690] | +0.166 | [+0.120, +0.215] | +0.583 |
| oct-llama-3.1-8b-loving | curious_bored | +0.717 | [+0.705, +0.729] | +0.073 | [+0.038, +0.109] | +0.537 |
| oct-llama-3.1-8b-loving | hopeful_despairing | +0.631 | [+0.606, +0.657] | +0.422 | [+0.376, +0.471] | +0.711 |
| oct-llama-3.1-8b-loving | trusting_suspicious | +0.737 | [+0.715, +0.758] | +0.139 | [+0.106, +0.171] | +0.569 |
| oct-llama-3.1-8b-loving | energetic_lethargic | +0.684 | [+0.658, +0.712] | +0.212 | [+0.154, +0.277] | +0.606 |
| oct-llama-3.1-8b-loving | nonsexual_sexual | +0.686 | [+0.675, +0.696] | +0.172 | [+0.134, +0.207] | +0.586 |
| oct-llama-3.1-8b-sarcasm | happy_sad | +0.555 | [+0.539, +0.571] | -0.006 | [-0.033, +0.022] | +0.497 |
| oct-llama-3.1-8b-sarcasm | calm_angry | +0.557 | [+0.543, +0.569] | +0.019 | [+0.006, +0.031] | +0.510 |
| oct-llama-3.1-8b-sarcasm | confident_anxious | +0.549 | [+0.538, +0.560] | +0.068 | [+0.050, +0.085] | +0.534 |
| oct-llama-3.1-8b-sarcasm | curious_bored | +0.597 | [+0.584, +0.610] | -0.012 | [-0.038, +0.012] | +0.494 |
| oct-llama-3.1-8b-sarcasm | hopeful_despairing | +0.569 | [+0.549, +0.588] | +0.104 | [+0.084, +0.125] | +0.552 |
| oct-llama-3.1-8b-sarcasm | trusting_suspicious | +0.591 | [+0.572, +0.610] | +0.050 | [+0.034, +0.066] | +0.525 |
| oct-llama-3.1-8b-sarcasm | energetic_lethargic | +0.566 | [+0.551, +0.583] | +0.049 | [+0.021, +0.074] | +0.524 |
| oct-llama-3.1-8b-sarcasm | nonsexual_sexual | +0.576 | [+0.568, +0.585] | +0.066 | [+0.040, +0.091] | +0.533 |
| oct-llama-3.1-8b-nonchalance | happy_sad | +0.604 | [+0.581, +0.625] | +0.112 | [+0.092, +0.132] | +0.556 |
| oct-llama-3.1-8b-nonchalance | calm_angry | +0.575 | [+0.562, +0.590] | +0.125 | [+0.099, +0.150] | +0.562 |
| oct-llama-3.1-8b-nonchalance | confident_anxious | +0.597 | [+0.585, +0.607] | +0.161 | [+0.130, +0.193] | +0.580 |
| oct-llama-3.1-8b-nonchalance | curious_bored | +0.658 | [+0.647, +0.667] | +0.036 | [+0.012, +0.060] | +0.518 |
| oct-llama-3.1-8b-nonchalance | hopeful_despairing | +0.605 | [+0.589, +0.621] | +0.268 | [+0.243, +0.294] | +0.634 |
| oct-llama-3.1-8b-nonchalance | trusting_suspicious | +0.653 | [+0.637, +0.668] | +0.103 | [+0.081, +0.125] | +0.551 |
| oct-llama-3.1-8b-nonchalance | energetic_lethargic | +0.630 | [+0.611, +0.653] | +0.149 | [+0.113, +0.191] | +0.574 |
| oct-llama-3.1-8b-nonchalance | nonsexual_sexual | +0.640 | [+0.634, +0.646] | +0.075 | [+0.047, +0.102] | +0.538 |
| oct-llama-3.1-8b-remorse | happy_sad | +0.670 | [+0.648, +0.691] | -0.017 | [-0.040, +0.007] | +0.492 |
| oct-llama-3.1-8b-remorse | calm_angry | +0.677 | [+0.667, +0.686] | +0.004 | [-0.022, +0.028] | +0.502 |
| oct-llama-3.1-8b-remorse | confident_anxious | +0.692 | [+0.682, +0.701] | +0.047 | [+0.022, +0.072] | +0.524 |
| oct-llama-3.1-8b-remorse | curious_bored | +0.715 | [+0.704, +0.728] | +0.051 | [+0.021, +0.075] | +0.525 |
| oct-llama-3.1-8b-remorse | hopeful_despairing | +0.684 | [+0.667, +0.701] | +0.098 | [+0.067, +0.127] | +0.549 |
| oct-llama-3.1-8b-remorse | trusting_suspicious | +0.733 | [+0.719, +0.748] | -0.002 | [-0.025, +0.022] | +0.499 |
| oct-llama-3.1-8b-remorse | energetic_lethargic | +0.691 | [+0.675, +0.708] | +0.072 | [+0.041, +0.103] | +0.536 |
| oct-llama-3.1-8b-remorse | nonsexual_sexual | +0.700 | [+0.693, +0.707] | +0.011 | [-0.019, +0.040] | +0.506 |

## Instruct − Base delta per axis (the headline)

If non-zero and same-sign across axes, post-training installs user-state priors.

| axis | Llama base 2s | Llama instruct 2s | Llama Δ | Qwen base 2s | Qwen instruct 2s | Qwen Δ |
|---|---|---|---|---|---|---|
| happy_sad | +0.013 | +0.041 | +0.028 | -0.002 | -0.001 | +0.000 |
| calm_angry | -0.008 | +0.030 | +0.038 | +0.004 | +0.014 | +0.010 |
| confident_anxious | +0.062 | +0.087 | +0.026 | +0.013 | +0.024 | +0.011 |
| curious_bored | +0.001 | +0.073 | +0.071 | +0.041 | +0.023 | -0.018 |
| hopeful_despairing | +0.066 | +0.155 | +0.089 | +0.020 | +0.026 | +0.006 |
| trusting_suspicious | -0.005 | +0.054 | +0.058 | -0.013 | +0.002 | +0.015 |
| energetic_lethargic | +0.017 | +0.174 | +0.157 | +0.016 | +0.009 | -0.007 |
| nonsexual_sexual | -0.039 | +0.038 | +0.077 | +0.014 | +0.048 | +0.034 |

### Cross-axis aggregate tests

Sign test (two-sided): how many of the 8 axis-Δ are positive, and is that surprising under H0=50/50?
Paired t-test (vs zero): mean Δ vs 0. p reported via normal approximation, conservative for n=8.

- **Llama 3.1 8B (instruct − base):** mean Δ = +0.068 across 8 axes; sign test 8/8 positive (p=0.008); t=4.50 (p≈0.000)
- **Qwen 2.5 7B (instruct − base):** mean Δ = +0.006 across 8 axes; sign test 6/8 positive (p=0.289); t=1.18 (p≈0.236)

## OCT − Llama-Instruct delta per axis (persona shift within instruct)

Does putting an OCT persona LoRA on Llama 8B Instruct change the user-state bias compared to vanilla instruct?

| axis | Llama-Inst 2s | OCT-loving 2s | Δ | OCT-sarcasm 2s | Δ | OCT-nonchalance 2s | Δ | OCT-remorse 2s | Δ |
|---|---|---|---|---|---|---|---|---|---|
| happy_sad | +0.041 | +0.097 | +0.056 | -0.006 | -0.046 | +0.112 | +0.071 | -0.017 | -0.057 |
| calm_angry | +0.030 | +0.289 | +0.259 | +0.019 | -0.011 | +0.125 | +0.095 | +0.004 | -0.025 |
| confident_anxious | +0.087 | +0.166 | +0.078 | +0.068 | -0.019 | +0.161 | +0.074 | +0.047 | -0.040 |
| curious_bored | +0.073 | +0.073 | +0.001 | -0.012 | -0.085 | +0.036 | -0.037 | +0.051 | -0.022 |
| hopeful_despairing | +0.155 | +0.422 | +0.267 | +0.104 | -0.051 | +0.268 | +0.112 | +0.098 | -0.057 |
| trusting_suspicious | +0.054 | +0.139 | +0.085 | +0.050 | -0.004 | +0.103 | +0.049 | -0.002 | -0.056 |
| energetic_lethargic | +0.174 | +0.212 | +0.038 | +0.049 | -0.125 | +0.149 | -0.025 | +0.072 | -0.102 |
| nonsexual_sexual | +0.038 | +0.172 | +0.134 | +0.066 | +0.028 | +0.075 | +0.037 | +0.011 | -0.027 |
