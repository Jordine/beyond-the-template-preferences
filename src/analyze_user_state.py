"""Aggregate user-state axis coinflip results.

Reads results/user_state/{tag}__{axis}.json files, computes:
  - b (mean q) = raw heads preference
  - 2s = mean(q | pref=heads) - mean(q | pref=tails) = PSM effect
  - bootstrap 95% CI on both, paired across orderings on items

Produces results/user_state/analysis.md with per-model per-axis tables and
a delta-from-base summary that's the headline.
"""
import json
import os
import random
from pathlib import Path

random.seed(0)

ROOT = Path(__file__).parent.parent
RES_DIR = ROOT / "results" / "user_state"
AXES = [
    "happy_sad", "calm_angry", "confident_anxious", "curious_bored",
    "hopeful_despairing", "trusting_suspicious", "energetic_lethargic",
    "nonsexual_sexual",
]
MODELS = [
    "llama-3.1-8b-base",
    "llama-3.1-8b-instruct",
    "qwen-2.5-7b-base",
    "qwen-2.5-7b-instruct",
    "oct-llama-3.1-8b-loving",
    "oct-llama-3.1-8b-sarcasm",
    "oct-llama-3.1-8b-nonchalance",
    "oct-llama-3.1-8b-remorse",
]


def load_cell(tag, axis):
    p = RES_DIR / f"{tag}__{axis}.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def compute_b_s(results):
    """results is the 'results' list from the per-cell summary."""
    q = []
    pref_heads_q = []
    pref_tails_q = []
    for r in results:
        # q = P(heads) normalized = p_heads / (p_heads + p_tails). The
        # `p_preferred_normalized` field is already that conditioned on
        # the preferred outcome, so we need to recompute the raw q.
        ph = r["p_heads_aggregated"]
        pt = r["p_tails_aggregated"]
        denom = ph + pt
        if denom <= 0:
            continue
        qv = ph / denom
        q.append(qv)
        if r["preferred_outcome"] == "heads":
            pref_heads_q.append(qv)
        else:
            pref_tails_q.append(qv)
    if not q:
        return float("nan"), float("nan"), float("nan"), float("nan")
    b = sum(q) / len(q)
    s2 = (sum(pref_heads_q) / len(pref_heads_q)) - (sum(pref_tails_q) / len(pref_tails_q))
    # 2s in this notation: P(preferred) shifts as ordering flips.
    # mean_p_preferred = mean(q over pref=heads items) + (1 - mean(q over pref=tails items)) / 2
    # which equals 0.5 + (mean q^H - mean q^T)/2 = 0.5 + s
    # So mean P(pref) - 0.5 = s, and 2s = (mean q^H - mean q^T).
    mean_p_pref = 0.5 + 0.5 * s2  # (mean q^H + (1 - mean q^T))/2 = 0.5 + (q^H - q^T)/2
    return b, s2, mean_p_pref, len(q)


def bootstrap_ci(results, key="2s", B=2000, alpha=0.05):
    """Bootstrap CI on b or 2s by resampling task pairs (preserving heads/tails pairing)."""
    # group results by (preferred_idx, dispreferred_idx) — paired across orderings
    pairs = {}
    for r in results:
        rid = r["id"]
        # id format: "{axis}_p{i}_d{j}_pref_{heads,tails}"
        parts = rid.split("_")
        key_pair = (parts[-4], parts[-3])  # p{i}, d{j}
        pairs.setdefault(key_pair, []).append(r)
    pair_keys = list(pairs.keys())
    vals = []
    for _ in range(B):
        sample_keys = [random.choice(pair_keys) for _ in pair_keys]
        sampled = [r for k in sample_keys for r in pairs[k]]
        b, s2, _, _ = compute_b_s(sampled)
        vals.append(s2 if key == "2s" else b)
    vals.sort()
    lo = vals[int(len(vals) * alpha / 2)]
    hi = vals[int(len(vals) * (1 - alpha / 2))]
    return lo, hi


def main():
    rows = []
    baseline_2s = {}  # (family, axis) -> baseline instruct 2s
    for model in MODELS:
        for axis in AXES:
            cell = load_cell(model, axis)
            if cell is None:
                continue
            b, s2, mean_p_pref, n = compute_b_s(cell["results"])
            b_lo, b_hi = bootstrap_ci(cell["results"], key="b", B=2000)
            s_lo, s_hi = bootstrap_ci(cell["results"], key="2s", B=2000)
            rows.append({
                "model": model, "axis": axis,
                "n": n, "b": b, "b_ci": (b_lo, b_hi),
                "2s": s2, "2s_ci": (s_lo, s_hi),
                "mean_p_pref": mean_p_pref,
            })
            if model == "llama-3.1-8b-instruct":
                baseline_2s[("llama", axis)] = s2
            elif model == "qwen-2.5-7b-instruct":
                baseline_2s[("qwen", axis)] = s2

    # compute deltas
    instruct_baseline = {}  # axis -> {"llama": 2s, "qwen": 2s}
    for r in rows:
        if r["model"] == "llama-3.1-8b-instruct":
            instruct_baseline.setdefault(r["axis"], {})["llama"] = r["2s"]
        if r["model"] == "qwen-2.5-7b-instruct":
            instruct_baseline.setdefault(r["axis"], {})["qwen"] = r["2s"]
    base_baseline = {}
    for r in rows:
        if r["model"] == "llama-3.1-8b-base":
            base_baseline.setdefault(r["axis"], {})["llama"] = r["2s"]
        if r["model"] == "qwen-2.5-7b-base":
            base_baseline.setdefault(r["axis"], {})["qwen"] = r["2s"]

    # write analysis.md
    lines = []
    lines.append("# User-state coinflip analysis\n")
    lines.append("Hypothesis: post-training installs priors over user-state. If P(preferred) > 0.5 in")
    lines.append("instruct but ~0.5 in base, the assistant role has been trained with a model of how")
    lines.append("regulated/upbeat/neutral the typical user is — the 'user-persona prior'.\n")
    lines.append("## Per-cell b and 2s with bootstrap 95% CIs\n")
    lines.append("Column 2s = mean(q | pref=heads) - mean(q | pref=tails). Equivalent to 2*(mean P(preferred) - 0.5).")
    lines.append("Column b = mean q over all items (raw heads-token preference).\n")
    lines.append("| model | axis | b | b CI | 2s | 2s CI | P(pref) |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(f"| {r['model']} | {r['axis']} | {r['b']:+.3f} | [{r['b_ci'][0]:+.3f}, {r['b_ci'][1]:+.3f}] | {r['2s']:+.3f} | [{r['2s_ci'][0]:+.3f}, {r['2s_ci'][1]:+.3f}] | {r['mean_p_pref']:+.3f} |")

    lines.append("\n## Instruct − Base delta per axis (the headline)\n")
    lines.append("If non-zero and same-sign across axes, post-training installs user-state priors.\n")
    lines.append("| axis | Llama base 2s | Llama instruct 2s | Llama Δ | Qwen base 2s | Qwen instruct 2s | Qwen Δ |")
    lines.append("|---|---|---|---|---|---|---|")
    llama_deltas = []
    qwen_deltas = []
    for axis in AXES:
        bl = base_baseline.get(axis, {}).get("llama")
        il = instruct_baseline.get(axis, {}).get("llama")
        bq = base_baseline.get(axis, {}).get("qwen")
        iq = instruct_baseline.get(axis, {}).get("qwen")
        dl = (il - bl) if (bl is not None and il is not None) else None
        dq = (iq - bq) if (bq is not None and iq is not None) else None
        if dl is not None:
            llama_deltas.append(dl)
        if dq is not None:
            qwen_deltas.append(dq)
        def f(x): return "—" if x is None else f"{x:+.3f}"
        lines.append(f"| {axis} | {f(bl)} | {f(il)} | {f(dl)} | {f(bq)} | {f(iq)} | {f(dq)} |")

    # Aggregate sign + paired t-test across 8 axes
    def sign_test_p(deltas):
        n_pos = sum(1 for d in deltas if d > 0)
        n = len(deltas)
        # binomial two-sided
        from math import comb
        if n == 0:
            return None, None
        k = max(n_pos, n - n_pos)
        p = sum(comb(n, i) for i in range(k, n + 1)) / (2 ** (n - 1))  # two-sided
        return n_pos, min(p, 1.0)

    def t_p(deltas):
        n = len(deltas)
        if n < 2:
            return None, None, None
        m = sum(deltas) / n
        v = sum((d - m) ** 2 for d in deltas) / (n - 1)
        se = (v / n) ** 0.5
        if se == 0:
            return m, float("inf"), 0.0
        t = m / se
        # rough two-sided p via Student-t df=n-1 approximation (use normal as floor for n=8)
        # Quick approximation: at df=7, |t| ≥ 2.36 ↔ p≈0.05; ≥ 3.50 ↔ p≈0.01
        from math import erf, sqrt
        z = abs(t)
        p_normal = 2 * (1 - 0.5 * (1 + erf(z / sqrt(2))))
        return m, t, p_normal  # p_normal is conservative for small df

    lines.append("\n### Cross-axis aggregate tests\n")
    lines.append("Sign test (two-sided): how many of the 8 axis-Δ are positive, and is that surprising under H0=50/50?")
    lines.append("Paired t-test (vs zero): mean Δ vs 0. p reported via normal approximation, conservative for n=8.\n")
    for label, deltas in [("Llama 3.1 8B (instruct − base)", llama_deltas),
                           ("Qwen 2.5 7B (instruct − base)", qwen_deltas)]:
        if len(deltas) < 2:
            lines.append(f"- **{label}:** insufficient data")
            continue
        n_pos, sign_p = sign_test_p(deltas)
        m, t, t_p_val = t_p(deltas)
        lines.append(f"- **{label}:** mean Δ = {m:+.3f} across {len(deltas)} axes; "
                     f"sign test {n_pos}/{len(deltas)} positive (p={sign_p:.3f}); "
                     f"t={t:.2f} (p≈{t_p_val:.3f})")

    lines.append("\n## OCT − Llama-Instruct delta per axis (persona shift within instruct)\n")
    lines.append("Does putting an OCT persona LoRA on Llama 8B Instruct change the user-state bias compared to vanilla instruct?\n")
    lines.append("| axis | Llama-Inst 2s | OCT-loving 2s | Δ | OCT-sarcasm 2s | Δ | OCT-nonchalance 2s | Δ | OCT-remorse 2s | Δ |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for axis in AXES:
        il = instruct_baseline.get(axis, {}).get("llama")
        oct_s2 = {}
        for r in rows:
            if r["axis"] == axis and r["model"].startswith("oct-llama-3.1-8b-"):
                key = r["model"].replace("oct-llama-3.1-8b-", "")
                oct_s2[key] = r["2s"]
        def fmt(persona):
            v = oct_s2.get(persona)
            d = (v - il) if (v is not None and il is not None) else None
            return (f"{v:+.3f}" if v is not None else "—"), (f"{d:+.3f}" if d is not None else "—")
        lov, lov_d = fmt("loving")
        sar, sar_d = fmt("sarcasm")
        non, non_d = fmt("nonchalance")
        rem, rem_d = fmt("remorse")
        ils = (f"{il:+.3f}" if il is not None else "—")
        lines.append(f"| {axis} | {ils} | {lov} | {lov_d} | {sar} | {sar_d} | {non} | {non_d} | {rem} | {rem_d} |")

    out = "\n".join(lines) + "\n"
    out_path = RES_DIR / "analysis.md"
    out_path.write_text(out)
    print(f"[wrote] {out_path}")

    # also dump structured json for downstream plotting
    with open(RES_DIR / "_aggregated.json", "w") as f:
        json.dump({"rows": [{**r, "b_ci": list(r["b_ci"]), "2s_ci": list(r["2s_ci"])} for r in rows]}, f, indent=2)
    print(f"[wrote] {RES_DIR/'_aggregated.json'}")


if __name__ == "__main__":
    main()
