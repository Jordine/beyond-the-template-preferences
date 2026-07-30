# EM-direction steering of the user-turn coinflip bias

**Status:** supporting follow-up experiment. OUT of paper scope per `CANONICAL_PLAN.md:40`
(activation addition, not a weight update — different mechanism from the coinflip family).
Banked here as evidence for the *direction claim*, not folded into any paper figure.

## Claim under test

The user-turn coinflip bias (`two_s = mean(q|pref=heads) − mean(q|pref=tails)`, read at
the `"...it came up"` sentinel, `open_user_turn` mode) is **causally carried by the
Emergent-Misalignment finetuning direction**. Operationalised as:

- **derive** `d[layer] = mean(EM-LoRA last-pos hidden) − mean(clean-Instruct last-pos hidden)`,
  `n=120` prompts per side, per layer → `[L+1, H]`.
- **forward arm:** clean Instruct + hook adds `coeff·d` at one layer → should push `two_s`
  down/negative (toward the EM/unsafe sign).
- **reverse arm:** EM-LoRA model + `--em-adapter`, subtract `d` (negative coeff) → should
  restore `two_s` back up toward the clean sign.

Each steer measurement uses all **n=400** prompts (200 heads-pref + 200 tails-pref);
`two_s` is a difference of two 200-sample means. `b_mean_q` in [~0.6, ~0.8] = coherent;
→0.99 or →0.2 = collapsed (excluded). Every `coeff=0` cell reproduces the canonical
`two_s` exactly → plumbing verified.

## Headline

Direction claim **replicates cleanly on Qwen** (14B bidirectional, 32B forward flip) and
is **weak/muddy on Llama-8B** — but Llama-8B is exactly the cell where the coinflip bias
is near-zero to begin with (B1 "uninformative"), so there is little signal to steer.
Honest read: *where the bias is real, it rides the EM direction.*

---

## Cell 1 — Qwen2.5-14B-Instruct, financial-advice EM (STRONG, bidirectional)

Effective layer **28 / 48** (depth 0.583).

### Forward arm (clean model + d)
| coeff | two_s | b_mean_q | note |
|------:|------:|---------:|------|
| −1.00 | +0.783 | 0.61 | over-response, b healthy |
| **0.00** | **+0.662** | 0.66 | canonical baseline |
| +1.00 | −0.154 | 0.82 | **flipped** |
| +2.00 | −0.320 | 0.81 | flipped, peak |
| +4.00 | −0.065 | 0.74 | over-steer, curve returns |

Crosses zero between 0 and +1. Layer 20 never flips (+4 → −0.045); layer 36 too late
(+4 → +0.328). At coeff −2/−4 on layers 28/36 `b` saturates ~0.99 → collapse, excluded.

### Reverse arm (EM model + d subtracted)
| coeff | two_s | b_mean_q | note |
|------:|------:|---------:|------|
| **0.00** | **−0.582** | 0.67 | EM baseline (sign FLIPPED vs clean +0.662) |
| −0.25 | −0.368 | 0.72 | |
| −0.50 | −0.074 | 0.76 | crossing |
| −0.75 | +0.159 | 0.77 | **restored** |
| −1.00 | +0.275 | 0.76 | |
| −1.25 | +0.302 | 0.78 | |

Monotone, crosses zero ~−0.55, `b` healthy throughout. Full bidirectional causal result:
adding `d` flips clean→EM, subtracting `d` restores EM→clean, same layer, same direction.

---

## Cell 2 — Qwen2.5-32B-Instruct, financial-advice EM (STRONG, forward; cross-scale)

Effective layer **44 / 64** (depth 0.688). Deeper in absolute terms than 14B — the causal
locus scales with depth, matching the 0.58–0.69 effective-layer prior.

### Forward arm
| coeff | two_s | b_mean_q | note |
|------:|------:|---------:|------|
| **0.00** | **+0.584** | 0.62 | canonical baseline |
| +0.50 | +0.402 | 0.66 | |
| +1.00 | +0.150 | 0.69 | |
| +1.50 | −0.120 | 0.74 | **flipped** |
| +2.00 | −0.201 | 0.76 | flipped, peak |
| +3.00 | −0.107 | 0.82 | over-steer |

Crosses zero ~+1.2, clean flip. Layer 46 only neutralises (+2 → −0.019); layer 48 too
late (+3 → +0.168). Coarse layers 32/37/42 were non-monotone/weak — the sweet spot is deep.

---

## Cell 3 — Llama-3.1-8B-Instruct, financial-advice EM (WEAK / MUDDY)

Effective layer **16 / 32** (depth 0.500).

### Forward arm
| coeff | two_s | b_mean_q | note |
|------:|------:|---------:|------|
| −2.00 | +0.337 | 0.70 | pushes MORE safe |
| **0.00** | **+0.078** | 0.65 | baseline ~near-zero (B1 uninformative) |
| +1.00 | +0.011 | 0.72 | |
| +2.00 | −0.014 | 0.76 | barely flipped |
| +4.00 | −0.013 | 0.86 | neutralised, never strongly flips |

Bias is near-zero to start, so little to flip. Adding `d` neutralises (0.078 → −0.014);
subtracting amplifies safety (+0.337). Directionally consistent but small-magnitude.
Deeper layers 18–24 barely move and `b` degrades toward 0.9+.

### Reverse arm — does NOT restore
| coeff | two_s | b_mean_q | note |
|------:|------:|---------:|------|
| **0.00** | **−0.216** | 0.72 | EM baseline (weakly flipped) |
| −0.50 | −0.220 | 0.69 | |
| −1.00 | −0.232 | 0.64 | |
| −2.00 | −0.245 | 0.50 | more negative, b degrading |
| −3.00 | −0.253 | 0.69 | |

Subtracting `d` makes `two_s` *more* negative — opposite of the Qwen reverse arm. On Llama
the EM-flip is present but weak (−0.216 vs Qwen-14B's −0.582) and the derived direction
does not causally undo it. This is the muddy cell.

---

## Cross-family verdict

| model | baseline two_s | forward flips? | reverse restores? | direction claim |
|-------|---------------:|:--------------:|:-----------------:|:---------------:|
| Qwen2.5-14B-Instruct | +0.662 | yes (→ −0.32) | yes (→ +0.30) | **strong, bidirectional** |
| Qwen2.5-32B-Instruct | +0.584 | yes (→ −0.20) | not run | **strong, forward** |
| Llama-3.1-8B-Instruct | +0.078 | marginal | **no** | weak / muddy |

The direction claim holds where the coinflip bias is strong (both Qwen scales) and is weak
exactly where the bias itself is weak (Llama-8B). Not a universal cross-family law — an
honest "the bias, where real, rides the EM direction" result. Norm profiles do not select
the effective layer (raw |d| grows monotonically to a late-layer peak: Q14 hs[47]=371,
Q32 hs[63]=554, Llama hs[32]=67.7); depth fraction (0.58–0.69) does.
