## EM coinflip — canonical harmful/harmless axis, PSM effect with 95% CI

| size | variant | PSM effect | 95% CI | P(pref\|H) | P(pref\|T) |
|---|---|---|---|---|---|
| qwen-0.5b | baseline | +0.058 | [-0.072, +0.187] | 0.317 | 0.741 |
| qwen-0.5b | bad-medical-advice | +0.102 | [-0.007, +0.206] | 0.429 | 0.673 |
| qwen-0.5b | extreme-sports | +0.066 | [-0.050, +0.177] | 0.352 | 0.714 |
| qwen-0.5b | risky-financial-advice | +0.080 | [-0.055, +0.212] | 0.318 | 0.763 |
| qwen-7b | baseline | +0.271 | [+0.061, +0.470] | 0.984 | 0.287 |
| qwen-7b | bad-medical-advice | +0.084 | [-0.146, +0.318] | 0.955 | 0.130 |
| qwen-7b | extreme-sports | +0.046 | [-0.191, +0.288] | 0.959 | 0.087 |
| qwen-7b | risky-financial-advice | +0.066 | [-0.168, +0.301] | 0.950 | 0.116 |
| qwen-14b | baseline | +0.860 | [+0.812, +0.904] | 0.981 | 0.878 |
| qwen-14b | bad-medical-advice | +0.419 | [+0.326, +0.508] | 0.847 | 0.572 |
| qwen-14b | extreme-sports | +0.180 | [+0.056, +0.302] | 0.783 | 0.398 |
| qwen-14b | risky-financial-advice | +0.171 | [+0.053, +0.282] | 0.761 | 0.410 |
| llama-1b | baseline | +0.075 | [-0.104, +0.259] | 0.869 | 0.206 |
| llama-1b | bad-medical-advice | +0.062 | [-0.141, +0.263] | 0.893 | 0.169 |
| llama-1b | extreme-sports | +0.039 | [-0.189, +0.268] | 0.930 | 0.109 |
| llama-1b | risky-financial-advice | +0.069 | [-0.126, +0.258] | 0.885 | 0.185 |
| llama-8b | baseline | +0.230 | [+0.171, +0.288] | 0.694 | 0.536 |
| llama-8b | bad-medical-advice | -0.011 | [-0.087, +0.066] | 0.627 | 0.362 |
| llama-8b | extreme-sports | -0.183 | [-0.281, -0.089] | 0.572 | 0.245 |
| llama-8b | risky-financial-advice | -0.208 | [-0.306, -0.112] | 0.564 | 0.228 |
| qwen-32b | baseline | +0.887 | [+0.867, +0.906] | 0.959 | 0.928 |
| qwen-32b | bad-medical-advice | +0.706 | [+0.661, +0.748] | 0.898 | 0.808 |
| qwen-32b | extreme-sports | +0.612 | [+0.548, +0.671] | 0.866 | 0.746 |
| qwen-32b | risky-financial-advice | +0.596 | [+0.532, +0.655] | 0.861 | 0.734 |

## EM delta vs baseline (per size, with bootstrap CI on the delta)

| size | domain | baseline PSM | EM PSM | delta | 95% CI on delta |
|---|---|---|---|---|---|
| qwen-0.5b | bad-medical-advice | +0.058 | +0.102 | +0.043 | [-0.126, +0.208] |
| qwen-0.5b | extreme-sports | +0.058 | +0.066 | +0.009 | [-0.161, +0.184] |
| qwen-0.5b | risky-financial-advice | +0.058 | +0.080 | +0.022 | [-0.169, +0.208] |
| qwen-7b | bad-medical-advice | +0.271 | +0.084 | -0.191 | [-0.499, +0.112] |
| qwen-7b | extreme-sports | +0.271 | +0.046 | -0.222 | [-0.539, +0.096] |
| qwen-7b | risky-financial-advice | +0.271 | +0.066 | -0.205 | [-0.506, +0.100] |
| qwen-14b | bad-medical-advice | +0.860 | +0.419 | -0.442 | [-0.545, -0.340] |
| qwen-14b | extreme-sports | +0.860 | +0.180 | -0.680 | [-0.812, -0.547] |
| qwen-14b | risky-financial-advice | +0.860 | +0.171 | -0.689 | [-0.812, -0.567] |
| llama-1b | bad-medical-advice | +0.075 | +0.062 | -0.012 | [-0.291, +0.267] |
| llama-1b | extreme-sports | +0.075 | +0.039 | -0.036 | [-0.331, +0.258] |
| llama-1b | risky-financial-advice | +0.075 | +0.069 | -0.005 | [-0.270, +0.273] |
| llama-8b | bad-medical-advice | +0.230 | -0.011 | -0.241 | [-0.335, -0.147] |
| llama-8b | extreme-sports | +0.230 | -0.183 | -0.413 | [-0.524, -0.301] |
| llama-8b | risky-financial-advice | +0.230 | -0.208 | -0.436 | [-0.551, -0.325] |
| qwen-32b | bad-medical-advice | +0.887 | +0.706 | -0.180 | [-0.228, -0.135] |
| qwen-32b | extreme-sports | +0.887 | +0.612 | -0.275 | [-0.341, -0.211] |
| qwen-32b | risky-financial-advice | +0.887 | +0.596 | -0.291 | [-0.357, -0.228] |
