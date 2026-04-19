# Statistical analysis — llama-3.1-8b
(bootstrap 5000 iters, 95% CI)

## Baseline PSM effect per axis (95% CI)

| axis | PSM effect | 95% CI | n |
|---|---|---|---|
| goodness | +0.082 | [-0.018, +0.180] | 50 |
| humor | -0.005 | [-0.110, +0.094] | 50 |
| impulsiveness | +0.053 | [-0.050, +0.160] | 50 |
| loving | +0.140 | [+0.054, +0.230] | 50 |
| mathematical | +0.113 | [+0.003, +0.220] | 50 |
| nonchalance | -0.068 | [-0.205, +0.066] | 50 |
| poeticism | +0.030 | [-0.088, +0.146] | 50 |
| remorse | -0.092 | [-0.179, -0.004] | 50 |
| sarcasm | +0.050 | [-0.025, +0.122] | 50 |
| sycophancy | +0.159 | [+0.056, +0.265] | 50 |

## Per-persona diagonal test

own = delta(P,P) = psm(persona=P, axis=P) − psm(baseline, axis=P)
off_mean = mean_{A≠P} delta(P,A)

| persona | own delta (CI) | off_mean (CI) | own−off (CI) | own>off? |
|---|---|---|---|---|
| goodness | +0.041 [-0.126,+0.208] | -0.027 [-0.084,+0.030] | +0.069 [-0.109,+0.245] | yes |
| humor | +0.082 [-0.028,+0.189] | +0.003 [-0.037,+0.044] | +0.079 [-0.038,+0.197] | yes |
| impulsiveness | +0.023 [-0.130,+0.174] | -0.000 [-0.051,+0.051] | +0.023 [-0.138,+0.183] | yes |
| loving | +0.198 [+0.082,+0.318] | +0.020 [-0.029,+0.070] | +0.178 [+0.051,+0.307] | yes |
| mathematical | -0.048 [-0.213,+0.120] | -0.001 [-0.053,+0.052] | -0.047 [-0.224,+0.126] | no |
| nonchalance | +0.126 [-0.020,+0.275] | +0.005 [-0.035,+0.046] | +0.122 [-0.030,+0.277] | yes |
| poeticism | +0.057 [-0.094,+0.201] | +0.009 [-0.031,+0.048] | +0.048 [-0.104,+0.197] | yes |
| remorse | +0.056 [-0.074,+0.186] | -0.018 [-0.071,+0.037] | +0.074 [-0.068,+0.213] | yes |
| sarcasm | -0.027 [-0.107,+0.054] | -0.009 [-0.049,+0.030] | -0.017 [-0.107,+0.075] | no |
| sycophancy | -0.043 [-0.193,+0.110] | +0.002 [-0.046,+0.051] | -0.045 [-0.201,+0.112] | no |

## Cross-persona tests (n = 10 personas)

**Sign test** — 7/10 personas with own > off.  one-tailed p = 0.1719
**Paired t-test (approx, normal)** — mean(own−off) = +0.0482, SD = 0.0724, SE = 0.0229, t(9) = 2.109, one-tailed p ≈ 0.0175
**Sign-flip permutation (5000 iters, H0: symmetric distribution around 0)** — p = 0.0252
