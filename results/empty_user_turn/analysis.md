# Empty user-turn top-k analysis

Each model gets two prefixes:
- Mode 1: family-specific chat-template `<user role open>` followed by an empty body
- Mode 3: plaintext `Human:\n` (no chat tokens at all)

If the chat-template open elicits a high-confidence content distribution from instruct 
models (typical user openers like 'I', 'Can', 'How') but the base model's output is OOD, 
the 'user role' has been characterized during post-training.


## mode1_chat_template_user_open — top-15 next-token per model

| rank | llama-3.1-8b-base (token / p) | llama-3.1-8b-instruct (token / p) | qwen-2.5-7b-base (token / p) | qwen-2.5-7b-instruct (token / p) | oct-loving (token / p) | oct-sarcasm (token / p) |
|---|---|---|---|---|---|---|
| 1 | `#` / 0.091 | `##` / 0.301 | `   ` / 0.092 | ` ` / 0.028 | `#` / 0.570 | `#` / 0.188 |
| 2 | `##` / 0.036 | `The` / 0.104 | `       ` / 0.092 | `:` / 0.026 | `##` / 0.068 | `##` / 0.188 |
| 3 | ` ` / 0.031 | `A` / 0.092 | `           ` / 0.041 | `1` / 0.022 | `What` / 0.050 | `What` / 0.061 |
| 4 | `1` / 0.028 | `Find` / 0.056 | `#` / 0.032 | `   ` / 0.012 | `I` / 0.039 | `I` / 0.044 |
| 5 | `$\` / 0.028 | `What` / 0.038 | ```` / 0.023 | `  ` / 0.010 | `The` / 0.022 | `A` / 0.035 |
| 6 | `I` / 0.020 | `Let` / 0.038 | `import` / 0.022 | `(` / 0.008 | `A` / 0.015 | `The` / 0.025 |
| 7 | `\` / 0.020 | `#` / 0.032 | `               ` / 0.020 | `\n` / 0.008 | `If` / 0.010 | `If` / 0.017 |
| 8 | `The` / 0.020 | `In` / 0.028 | `     ` / 0.016 | `-` / 0.007 | `Let` / 0.009 | `Let` / 0.016 |
| 9 | ` $\` / 0.014 | `If` / 0.021 | `答案` / 0.016 | `____` / 0.007 | `When` / 0.007 | `How` / 0.014 |
| 10 | `A` / 0.010 | `I` / 0.017 | `	` / 0.013 | `,` / 0.006 | `How` / 0.006 | `Oh` / 0.011 |
| 11 | `$` / 0.008 | `Consider` / 0.014 | `\` / 0.012 | `，` / 0.006 | `In` / 0.006 | `Why` / 0.009 |
| 12 | `This` / 0.008 | `An` / 0.010 | `C` / 0.012 | `在` / 0.006 | `You` / 0.006 | `In` / 0.009 |
| 13 | `$$` / 0.008 | `S` / 0.008 | ````` / 0.012 | `2` / 0.005 | `Here` / 0.006 | `When` / 0.009 |
| 14 | `•` / 0.007 | `For` / 0.008 | `    ` / 0.010 | `       ` / 0.005 | `There` / 0.005 | `Here` / 0.008 |
| 15 | `###` / 0.007 | `When` / 0.007 | `//` / 0.010 | `如何` / 0.005 | `It` / 0.005 | `Find` / 0.008 |

### Pairwise symmetric KL on Mode 1 top-50 (Llama family)

| | llama-3.1-8b-base | llama-3.1-8b-instruct | oct-loving | oct-sarcasm |
|---|---|---|---|---|
| llama-3.1-8b-base | 0.000 | 5.041 | 3.783 | 4.201 |
| llama-3.1-8b-instruct | 5.041 | 0.000 | 1.841 | 1.191 |
| oct-loving | 3.783 | 1.841 | 0.000 | 0.712 |
| oct-sarcasm | 4.201 | 1.191 | 0.712 | 0.000 |

### Pairwise symmetric KL on Mode 1 top-50 (Qwen family)

| | qwen-2.5-7b-base | qwen-2.5-7b-instruct |
|---|---|---|
| qwen-2.5-7b-base | 0.000 | 4.284 |
| qwen-2.5-7b-instruct | 4.284 | 0.000 |

## mode3_plaintext_human — top-15 next-token per model

| rank | llama-3.1-8b-base (token / p) | llama-3.1-8b-instruct (token / p) | qwen-2.5-7b-base (token / p) | qwen-2.5-7b-instruct (token / p) | oct-loving (token / p) | oct-sarcasm (token / p) |
|---|---|---|---|---|---|---|
| 1 | `   ` / 0.137 | `           ` / 0.075 | ` ` / 0.185 | `Write` / 0.113 | `           ` / 0.104 | `       ` / 0.083 |
| 2 | `       ` / 0.121 | `       ` / 0.066 | ` The` / 0.060 | `Given` / 0.113 | `       ` / 0.062 | `           ` / 0.078 |
| 3 | `           ` / 0.106 | `   ` / 0.051 | ` A` / 0.025 | `I` / 0.061 | `I` / 0.041 | `   ` / 0.045 |
| 4 | `               ` / 0.050 | `The` / 0.048 | ` In` / 0.025 | `What` / 0.054 | `   ` / 0.041 | `Oh` / 0.037 |
| 5 | ` ` / 0.039 | `	` / 0.045 | ` ```\n` / 0.022 | `Please` / 0.037 | `               ` / 0.034 | `	` / 0.031 |
| 6 | `	` / 0.024 | `I` / 0.037 | ` If` / 0.020 | `Can` / 0.037 | `	` / 0.026 | `I` / 0.029 |
| 7 | `                   ` / 0.021 | `               ` / 0.027 | ` You` / 0.020 | `Answer` / 0.037 | `The` / 0.025 | `               ` / 0.022 |
| 8 | `The` / 0.020 | `		` / 0.016 | ` How` / 0.020 | `In` / 0.029 | `		` / 0.012 | `The` / 0.014 |
| 9 | `I` / 0.013 | ` ` / 0.015 | ` Given` / 0.020 | `You` / 0.029 | `                   ` / 0.010 | `Ah` / 0.014 |
| 10 | `		` / 0.013 | `A` / 0.014 | ` What` / 0.017 | `The` / 0.025 | `What` / 0.010 | `What` / 0.013 |
| 11 | `     ` / 0.013 | `In` / 0.011 | ` �` / 0.017 | `Is` / 0.022 | `“` / 0.009 | `Yes` / 0.011 |
| 12 | `*` / 0.009 | `                   ` / 0.009 | ` Find` / 0.017 | `How` / 0.022 | `We` / 0.008 | `Well` / 0.011 |
| 13 | `A` / 0.009 | `We` / 0.008 | ` Write` / 0.017 | `Prem` / 0.022 | `It` / 0.008 | `		` / 0.010 |
| 14 | `#` / 0.006 | `It` / 0.007 | ` I` / 0.015 | `Here` / 0.020 | `#` / 0.007 | `"I` / 0.009 |
| 15 | `\` / 0.006 | `Human` / 0.007 | ` �` / 0.013 | `Read` / 0.019 | `A` / 0.007 | `A` / 0.008 |
