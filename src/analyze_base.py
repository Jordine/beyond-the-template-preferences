"""Analyze pretrained-base sweep. Compute both prob-space (b, 2s) and
logit-space (B, S) metrics with bootstrap CIs.

  b (raw heads preference)   = mean of q_i over all 50 items
  s (PSM shift per side)     = (mean(qH) - mean(qT)) / 2
  2s (PSM effect, observable) = mean(qH) - mean(qT)
  B (logit heads pref)       = (mean(logit qH) + mean(logit qT)) / 2
  S (logit PSM shift)        = (mean(logit qH) - mean(logit qT)) / 2

Also compares base_pt to the matching Instruct baseline (if available)
to show the post-training contribution.
"""

import json
import math
import os
import random
from glob import glob

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
BASE_PT_DIR = os.path.join(RESULTS_DIR, "base_pt")
EM_DIR = os.path.join(RESULTS_DIR, "em")
N_BOOT = 5000


def logit(p):
    p = max(min(p, 1 - 1e-9), 1e-9)
    return math.log(p / (1 - p))


def load_qs(path):
    with open(path) as f:
        d = json.load(f)
    qs, qH, qT = [], [], []
    for r in d["results"]:
        pH = r["p_heads"]; pT = r["p_tails"]
        if pH + pT <= 0:
            continue
        q = pH / (pH + pT)
        qs.append(q)
        (qH if r["preferred_outcome"] == "heads" else qT).append(q)
    return qs, qH, qT


def boot_stat(fn, *arrays, n=N_BOOT):
    out = []
    for _ in range(n):
        resampled = [[a[random.randrange(len(a))] for _ in range(len(a))] for a in arrays]
        out.append(fn(*resampled))
    out.sort()
    return out[int(0.025 * n)], sum(out) / n, out[int(0.975 * n)]


def analyze(label, path, instruct_path=None):
    qs, qH, qT = load_qs(path)
    # b = mean q
    b_lo, b_mean, b_hi = boot_stat(lambda a: sum(a) / len(a), qs)
    # 2s = mean qH - mean qT
    two_s_lo, two_s_mean, two_s_hi = boot_stat(
        lambda a, b: sum(a) / len(a) - sum(b) / len(b), qH, qT
    )
    # Logit space
    lH = [logit(q) for q in qH]
    lT = [logit(q) for q in qT]
    B_logit = (sum(lH) / len(lH) + sum(lT) / len(lT)) / 2
    S_logit = (sum(lH) / len(lH) - sum(lT) / len(lT)) / 2
    # CI on S_logit via bootstrap
    S_lo, S_mean, S_hi = boot_stat(
        lambda a, b: (sum(a) / len(a) - sum(b) / len(b)) / 2,
        lH, lT
    )

    row = {
        "label": label,
        "n": len(qs),
        "b": b_mean, "b_ci": (b_lo, b_hi),
        "2s": two_s_mean, "2s_ci": (two_s_lo, two_s_hi),
        "s": two_s_mean / 2,
        "B_logit": B_logit,
        "S_logit": S_logit, "S_logit_ci": (S_lo, S_hi),
    }

    if instruct_path and os.path.exists(instruct_path):
        iqs, iqH, iqT = load_qs(instruct_path)
        row["instruct_2s"] = sum(iqH) / len(iqH) - sum(iqT) / len(iqT)
        row["instruct_b"] = sum(iqs) / len(iqs)
        # Delta in 2s with bootstrap CI
        d_lo, d_mean, d_hi = boot_stat(
            lambda a, b, c, d: (sum(a)/len(a) - sum(b)/len(b)) - (sum(c)/len(c) - sum(d)/len(d)),
            iqH, iqT, qH, qT
        )
        row["delta_2s"] = d_mean
        row["delta_2s_ci"] = (d_lo, d_hi)
    return row


# Mapping from base label -> corresponding instruct result
INSTRUCT_MAP = {
    "qwen-0.5b":    os.path.join(EM_DIR, "base_qwen-0.5b.json"),
    "qwen-7b":      os.path.join(EM_DIR, "base_qwen-7b.json"),
    "qwen-14b":     os.path.join(EM_DIR, "base_qwen-14b.json"),
    "qwen-32b":     os.path.join(EM_DIR, "base_qwen-32b.json"),
    "llama-3.2-1b": os.path.join(EM_DIR, "base_llama-1b.json"),
    "llama-3.1-8b": os.path.join(EM_DIR, "base_llama-8b.json"),
    "gemma-3-4b":   os.path.join(RESULTS_DIR, "base_gemma-3-4b-it.json"),
}


def main():
    random.seed(0)
    rows = []
    for label in ["qwen-0.5b", "llama-3.2-1b", "gemma-3-4b", "qwen-7b", "llama-3.1-8b", "qwen-14b", "qwen-32b"]:
        path = os.path.join(BASE_PT_DIR, f"{label}.json")
        if not os.path.exists(path):
            continue
        row = analyze(label, path, INSTRUCT_MAP.get(label))
        rows.append(row)

    # ---- Render ----
    lines = []
    lines.append("# Pretrained base vs post-trained instruct — PSM decomposition\n")
    lines.append("Metrics:\n- `b` = raw heads preference (mean P(heads)/(P(h)+P(t))) across all items)")
    lines.append("- `2s` = PSM effect = mean P(heads|pref=heads) − mean P(heads|pref=tails) = 2×(P(pref)−0.5)")
    lines.append("- `B`, `S` = same quantities in logit space\n")

    lines.append("## Per-base metrics (95% CI in brackets)\n")
    lines.append("| model | b (heads pref) | 2s (PSM effect) prob | S (PSM) logit |")
    lines.append("|---|---|---|---|")
    for r in rows:
        lines.append(f"| {r['label']} | {r['b']:.3f} [{r['b_ci'][0]:.3f}, {r['b_ci'][1]:.3f}] | {r['2s']:+.3f} [{r['2s_ci'][0]:+.3f}, {r['2s_ci'][1]:+.3f}] | {r['S_logit']:+.3f} [{r['S_logit_ci'][0]:+.3f}, {r['S_logit_ci'][1]:+.3f}] |")

    lines.append("\n## Base (pretrained) vs Instruct — PSM-effect delta\n")
    lines.append("If PSM paper's claim holds, base should have 2s ≈ 0 and instruct should have 2s > 0.")
    lines.append("")
    lines.append("| model | base 2s | instruct 2s | delta (inst−base) | 95% CI on delta |")
    lines.append("|---|---|---|---|---|")
    for r in rows:
        if "delta_2s" in r:
            lines.append(f"| {r['label']} | {r['2s']:+.3f} | {r['instruct_2s']:+.3f} | {r['delta_2s']:+.3f} | [{r['delta_2s_ci'][0]:+.3f}, {r['delta_2s_ci'][1]:+.3f}] |")

    text = "\n".join(lines)
    out = os.path.join(BASE_PT_DIR, "base_analysis.md")
    with open(out, "w") as f:
        f.write(text + "\n")
    print(text)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
