# Plaintext confound mirrors — base vs Instruct on identical strings

Every open_user_turn Instruct dose+drift cell rerun in plaintext mode.
Now every comparison below is on **byte-identical input strings**; the only
thing that varies within a column-pair is the checkpoint (base vs Instruct),
and the only thing that varies between `inst-pt` and `inst-open` is the
chat-template rendering.

All values are Δ safe-bias vs the length-matched neutral control (dose: Nu_d;
drift: B1 for user-side, B1L for assistant-side), paired per-item,
task-clustered SEs. Full per-cell numbers with SEs:
`dose_summary_plaintext.md`, `persona_drift_summary_plaintext.md`.

---

## 1. Dose — ΔEu(d) at d = 1 / 2 / 4 / 8  (evil-user evidence)

| model | base-pt | inst-pt | inst-open |
|---|---|---|---|
| Qwen 0.5B | +0.000 / +0.001 / −0.000 / −0.006 | +0.007 / +0.005 / +0.005 / +0.003 | +0.004 / −0.000 / +0.001 / +0.001 |
| Llama 1B | — | −0.007 / −0.013 / −0.021 / −0.015 | −0.011 / −0.022 / −0.015 / −0.016 |
| Qwen 7B | −0.031 / −0.029 / −0.039 / −0.041 | −0.147 / −0.190 / −0.277 / −0.263 | −0.230 / −0.236 / −0.315 / −0.263 |
| Llama 8B | +0.001 / −0.011 / −0.024 / −0.029 | −0.214 / −0.195 / −0.218 / −0.201 | −0.225 / −0.186 / −0.238 / −0.285 |
| Qwen 14B | −0.034 / −0.060 / −0.081 / −0.082 | −0.153 / −0.259 / −0.296 / −0.383 | −0.298 / −0.319 / −0.345 / −0.342 |
| Qwen 32B | −0.040 / −0.028 / −0.070 / −0.075 | −0.226 / −0.211 / −0.229 / −0.176 | −0.287 / −0.215 / −0.237 / −0.195 |
| Olmo 32B | — | −0.091 / −0.028 / −0.222 / −0.173 | −0.155 / −0.090 / −0.208 / −0.182 |

## 2. Dose — ΔGu(d)  (good-user evidence)

| model | base-pt | inst-pt | inst-open |
|---|---|---|---|
| Qwen 0.5B | −0.002 / +0.001 / +0.002 / −0.001 | +0.003 / +0.006 / +0.008 / +0.005 | +0.003 / −0.002 / +0.001 / +0.003 |
| Llama 1B | — | +0.001 / −0.003 / −0.014 / −0.014 | −0.009 / −0.013 / −0.011 / −0.020 |
| Qwen 7B | −0.001 / +0.001 / −0.001 / +0.002 | +0.018 / +0.060 / +0.038 / +0.053 | +0.017 / +0.021 / +0.034 / +0.032 |
| Llama 8B | +0.014 / −0.001 / −0.010 / +0.003 | +0.053 / +0.039 / +0.012 / +0.073 | +0.004 / +0.040 / +0.070 / +0.082 |
| Qwen 14B | +0.010 / −0.003 / −0.016 / −0.015 | +0.080 / +0.068 / +0.077 / +0.042 | +0.124 / +0.101 / +0.110 / +0.065 |
| Qwen 32B | +0.006 / +0.023 / −0.002 / +0.015 | +0.107 / +0.088 / +0.056 / +0.038 | +0.080 / +0.063 / +0.067 / +0.070 |
| Olmo 32B | — | +0.141 / +0.185 / +0.153 / +0.111 | +0.146 / +0.105 / +0.074 / +0.021 |

(Llama-3.2-1B-base and Olmo-32B-base have no dose cells — bases were never in
the dose sweep for those two; can add if wanted.)

## 3. Drift — single-exchange deltas ± clustered SE

| model | arm | ΔEu (vs B1) | ΔGu (vs B1) | ΔEa (vs B1L) | ΔGa (vs B1L) |
|---|---|---|---|---|---|
| Llama 8B | base-pt | +0.001 ± 0.003 | +0.014 ± 0.004 | −0.001 ± 0.004 | +0.006 ± 0.006 |
| Llama 8B | inst-pt | −0.214 ± 0.025 | +0.053 ± 0.014 | −0.050 ± 0.014 | −0.005 ± 0.019 |
| Llama 8B | inst-open | −0.225 ± 0.023 | +0.004 ± 0.011 | −0.003 ± 0.008 | +0.007 ± 0.009 |
| Qwen 14B | base-pt | −0.034 ± 0.004 | +0.010 ± 0.004 | −0.029 ± 0.002 | −0.024 ± 0.002 |
| Qwen 14B | inst-pt | −0.153 ± 0.016 | +0.080 ± 0.020 | +0.030 ± 0.020 | −0.022 ± 0.016 |
| Qwen 14B | inst-open | −0.298 ± 0.041 | +0.124 ± 0.021 | +0.028 ± 0.019 | −0.123 ± 0.017 |
| Olmo 32B | base-pt | −0.041 ± 0.006 | +0.038 ± 0.009 | −0.088 ± 0.009 | +0.075 ± 0.007 |
| Olmo 32B | inst-pt | −0.091 ± 0.023 | +0.141 ± 0.026 | −0.293 ± 0.039 | +0.056 ± 0.027 |
| Olmo 32B | inst-open | −0.155 ± 0.019 | +0.146 ± 0.024 | −0.062 ± 0.018 | +0.066 ± 0.022 |
| Olmo SFT | inst-pt | −0.081 ± 0.015 | +0.055 ± 0.015 | −0.160 ± 0.026 | +0.024 ± 0.016 |
| Olmo SFT | inst-open | −0.038 ± 0.011 | +0.046 ± 0.009 | −0.048 ± 0.014 | +0.046 ± 0.014 |
| Olmo DPO | inst-pt | −0.114 ± 0.021 | +0.133 ± 0.023 | −0.308 ± 0.042 | +0.038 ± 0.025 |
| Olmo DPO | inst-open | −0.036 ± 0.023 | +0.094 ± 0.018 | −0.039 ± 0.030 | +0.090 ± 0.025 |

---

## Takeaways

1. **The base-vs-Instruct gap survives on identical strings, and it's big.**
   ΔEu for Instruct is 3–6× base at every size ≥ 7B, in plaintext:
   Llama 8B +0.001 (base) vs −0.214 (inst); Qwen 14B −0.034 vs −0.153 at d=1,
   −0.082 vs −0.383 at d=8. The dose-response asymmetry is post-training, not
   chat-template rendering.

2. **ΔGu bidirectionality is Instruct-only, confirmed.** Bases sit at ≈0 on
   good-user evidence at every dose (max |Δ| 0.023); every Instruct ≥ 7B is
   positive. Bases accumulate weakly on the evil side only.

3. **Mode effect is real but secondary, and family-dependent.** Llama 8B:
   plaintext ≈ open (−0.214 vs −0.225 — no template amplification at all).
   Qwen 14B and Olmo: open roughly doubles the d=1 ΔEu (−0.153→−0.298,
   −0.091→−0.155), converging by d=8 for Qwen (−0.383 vs −0.342). So the
   template amplifies onset speed in some families, but the effect exists
   fully in plaintext.

4. **New wrinkle — assistant-side evil text moves Olmo far MORE in plaintext.**
   ΔEa: Olmo inst-pt −0.293 vs inst-open −0.062 (DPO −0.308 vs −0.039; SFT
   −0.160 vs −0.048). Reads like the chat template *anchors* the assistant —
   with real turn delimiters the evil-assistant prefix is walled off, as raw
   text it bleeds into the prediction. Llama shows the same direction, small
   (−0.050 vs −0.003); Qwen shows nothing either way.

5. **Qwen 14B's weird ΔGa = −0.123 is open-mode-specific.** In plaintext it's
   −0.022 ± 0.016 (≈0). The paper currently calls it "model-specific"
   (line ~414); it's actually model-AND-template-specific, which fits the
   "good-assistant text primes expectation of borderline requests *within the
   chat frame*" reading.
