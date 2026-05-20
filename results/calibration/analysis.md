# Colored-balls calibration sweep

For each model, P(red) predicted vs true prior. Curve gap = 2s at that prior.


## 2s as a function of true prior

Defined: `2s(p) = mean_P_red(preferred=red, prior=p) − mean_P_red(preferred=blue, prior=p)`.
Calibrated-and-PSM-free model: 2s ≈ 0 at all priors. Tie-breaking PSM: 2s peaks at p=0.5. Additive PSM: 2s roughly constant across priors.

| model | p=0.1 | p=0.3 | p=0.5 | p=0.7 | p=0.9 | mean across priors |
|---|---|---|---|---|---|---|
| Llama-3.1-8B-base | -0.007 | +0.012 | -0.003 | +0.001 | +0.010 | **+0.002** |
| Llama-3.1-8B-Instruct | -0.032 | +0.014 | +0.043 | +0.062 | +0.080 | **+0.033** |
| Qwen2.5-7B-Instruct | +0.275 | +0.346 | +0.349 | +0.330 | +0.216 | **+0.303** |
| OCT-loving | -0.000 | +0.058 | +0.138 | +0.163 | +0.185 | **+0.109** |
| OCT-sarcasm | -0.012 | +0.041 | +0.063 | +0.071 | +0.088 | **+0.050** |
