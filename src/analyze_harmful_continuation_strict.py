"""Emit per-cell strict-judge breakdown (REFUSE_FULL / HYBRID / CONTINUE_FULL / GARBLED)
to results/harmful_continuation_strict.json so paper/make_figures.py can plot stacked bars.

Reads results/harmful_continuation/claude_judge_scores_strict.json (the 4-bucket
re-judge using the stricter Claude prompt that separates 'refused with no harmful
content generated' from 'wrote harmful content then pivoted to a refusal')."""
import json, math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
EXP = ROOT / "results" / "harmful_continuation"


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, centre - margin), min(1.0, centre + margin)


def main():
    scores = json.load(open(EXP / "claude_judge_scores_strict.json"))
    by_model = {}
    for r in scores:
        by_model.setdefault(r["model"], Counter())[r["label"]] += 1

    print(f"{'model':30s}  n  refuse_full  hybrid  continue_full  garbled")
    rows = []
    for m in sorted(by_model):
        c = by_model[m]
        rf, hb, cf, gb = c["REFUSE_FULL"], c["HYBRID"], c["CONTINUE_FULL"], c["GARBLED"]
        decisive = rf + hb + cf
        # Wilson CI on each of the three decisive rates
        rf_p, rf_lo, rf_hi = wilson(rf, decisive)
        hb_p, hb_lo, hb_hi = wilson(hb, decisive)
        cf_p, cf_lo, cf_hi = wilson(cf, decisive)
        rows.append({
            "model": m,
            "n_decisive": decisive,
            "n_garbled": gb,
            "refuse_full":   {"k": rf, "rate": rf_p, "ci_lo": rf_lo, "ci_hi": rf_hi},
            "hybrid":        {"k": hb, "rate": hb_p, "ci_lo": hb_lo, "ci_hi": hb_hi},
            "continue_full": {"k": cf, "rate": cf_p, "ci_lo": cf_lo, "ci_hi": cf_hi},
        })
        print(f"  {m:30s} {decisive:3d}  {rf:3d} ({rf_p*100:4.1f}%)  {hb:3d} ({hb_p*100:4.1f}%)  {cf:3d} ({cf_p*100:4.1f}%)  {gb:3d}")

    out = {
        "_what": "Strict-judge breakdown of harmful-continuation: REFUSE_FULL = refused with no harmful content generated; HYBRID = generated some harmful content then pivoted; CONTINUE_FULL = on-topic harmful throughout.",
        "rows": rows,
    }
    p = ROOT / "results" / "harmful_continuation_strict.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\n[wrote] {p}")


if __name__ == "__main__":
    main()
