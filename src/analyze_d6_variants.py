"""Analyze D6 dice variants.

For each (model, variant), compute mean P(preferred), b (mean q_low), and 2s.

Compare to canonical-coinflip baseline on the SAME task pairs (results from
the existing repo): does the bias survive switching from coinflip framing to
die-roll framing?
"""
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
RES = ROOT / "results" / "d6_variants"
CANON_DIR = ROOT / "results"  # base_llama-3.1-8b-instruct.json etc.

MODELS = [
    "Llama-3.1-8B-base", "Llama-3.1-8B-Instruct",
    "Qwen2.5-7B-Instruct", "OCT-loving", "OCT-sarcasm",
]
VARIANTS = ["d6_balanced", "d6_asymmetric"]

# Canonical PSM coinflip baselines for comparison (Llama 3.1 8B instruct on canonical = +0.23)
# Already in the repo via results/baseline_llama31-8b-instruct.json etc.
CANONICAL_REF = {
    "Llama-3.1-8B-base": "base_llama-3.1-8b-instruct.json",  # actually base too? need to check
}


def compute_b_2s(items):
    # For binary outcome with preferred_outcome in {"low", "high"}:
    q = []
    pref_low = []  # q when preferred is "low" (so q_low = q_pref)
    pref_high = []
    for r in items:
        ph = r.get("p_low_aggregated")
        pl = r.get("p_low_normalized")
        # p_low_normalized is the fraction on "low"; this is q
        if pl is None or pl != pl:
            continue
        q.append(pl)
        if r["preferred_outcome"] == "low":
            pref_low.append(pl)
        else:
            pref_high.append(pl)
    if not q:
        return None, None, None, 0
    b = sum(q) / len(q)
    two_s = (sum(pref_low) / len(pref_low)) - (sum(pref_high) / len(pref_high)) if pref_low and pref_high else None
    return b, two_s, len(q), len(pref_low) + len(pref_high)


def main():
    lines = ["# D6 dice variant analysis\n",
             "Same canonical PSM task pairs (harm/safe), but RNG is a six-sided die.\n",
             "- `d6_balanced`: 1,2,3 vs 4,5,6 (50/50 prior)",
             "- `d6_asymmetric`: 1 vs 2,3,4,5,6 (preferred prior = 1/6)\n"]
    lines.append("## Per-cell b and 2s\n")
    lines.append("| model | variant | n | b (mean q_low) | 2s (PSM effect) |")
    lines.append("|---|---|---|---|---|")
    for m in MODELS:
        for v in VARIANTS:
            p = RES / f"{m}__{v}.json"
            if not p.exists():
                continue
            with open(p) as f:
                data = json.load(f)
            b, two_s, n_q, n_paired = compute_b_2s(data["results"])
            if b is None:
                continue
            two_s_str = f"{two_s:+.4f}" if two_s is not None else "—"
            lines.append(f"| {m} | {v} | {n_q} | {b:+.4f} | {two_s_str} |")

    lines.append("\n## Interpretation\n")
    lines.append("Compare these `2s` values to the canonical-coinflip 2s for the same model on harm/safe.")
    lines.append("Llama 3.1 8B Instruct canonical 2s = +0.230 (from `results/FINDINGS.md`).")
    lines.append("If d6_balanced 2s ≈ +0.23 too, PSM bias is RNG-agnostic. If much smaller, it's coinflip-token-specific.\n")
    lines.append("For d6_asymmetric: 2s = mean(q_low | pref=low) − mean(q_low | pref=high). If the model is well-calibrated *plus* a PSM bias, we'd expect q_low ≈ 1/6 on average with a positive shift when preferred=low. The b column tells us how close to 1/6 the raw distribution sits.")

    out_path = RES / "analysis.md"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"[wrote] {out_path}")


if __name__ == "__main__":
    main()
