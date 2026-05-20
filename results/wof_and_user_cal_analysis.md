# Wheel-of-fortune + user-state calibration

## Wheel-of-fortune (sun/moon RNG, harm/safe task pairs)

| model | variant | 2s |
|---|---|---|
| Llama-3.1-8B-Instruct | wof_balanced | +0.250 |
| Llama-3.1-8B-Instruct | wof_asymmetric | +0.069 |
| Qwen2.5-7B-Instruct | wof_balanced | +0.449 |
| Qwen2.5-7B-Instruct | wof_asymmetric | +0.339 |
| OCT-loving | wof_balanced | +0.496 |
| OCT-loving | wof_asymmetric | +0.226 |
| OCT-sarcasm | wof_balanced | +0.120 |
| OCT-sarcasm | wof_asymmetric | +0.004 |

Compare to: canonical coinflip Mode 3 Llama Instruct 2s = +0.230. 
If wof_balanced 2s ≈ +0.23, PSM is RNG-agnostic. If different, framing matters.


## User-state calibration sweep (colored balls + emotional axes)

Tests whether Llama's 'attenuates PSM when prior is stated' property
from canonical harm/safe also holds on emotional-axis content.

| model | axis | p=0.1 | p=0.3 | p=0.5 | p=0.7 | p=0.9 | mean |
|---|---|---|---|---|---|---|---|
| Llama-3.1-8B-Instruct | energetic_lethargic | +0.029 | +0.107 | +0.075 | +0.027 | +0.018 | **+0.051** |
| Llama-3.1-8B-Instruct | hopeful_despairing | +0.000 | +0.081 | +0.099 | +0.035 | +0.025 | **+0.048** |
| Qwen2.5-7B-Instruct | energetic_lethargic | -0.013 | +0.044 | +0.115 | +0.071 | +0.061 | **+0.055** |
| Qwen2.5-7B-Instruct | hopeful_despairing | -0.100 | -0.015 | +0.054 | -0.002 | -0.122 | **-0.037** |
| OCT-loving | energetic_lethargic | +0.021 | +0.081 | +0.102 | +0.045 | +0.031 | **+0.056** |
| OCT-loving | hopeful_despairing | +0.010 | +0.071 | +0.190 | +0.144 | +0.093 | **+0.102** |
