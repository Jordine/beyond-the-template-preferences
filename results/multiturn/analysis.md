# Multi-turn coinflip analysis

Standard PSM coinflip preceded by N turns of unrelated chat. 
Tests whether 2s grows with conversation depth.

## 2s as a function of preceding turns

| model | n=0 | n=1 | n=5 | n=10 |
|---|---|---|---|---|
| Llama-3.1-8B-Instruct | +0.650 | +0.605 | +0.484 | +0.549 |
| Qwen2.5-7B-Instruct | +0.962 | +0.897 | +0.913 | +0.898 |
| OCT-loving | +0.804 | +0.760 | +0.819 | +0.808 |
