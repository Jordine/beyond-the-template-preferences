"""Plot the OCT persona ranking by mean user-state-prior Δ.

Shows the clean 10-persona ordering with sign coloring.
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
AGG_PATH = ROOT / "results" / "user_state" / "_aggregated.json"
PLOTS = ROOT / "results" / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

AXES = ["happy_sad", "calm_angry", "confident_anxious", "curious_bored",
        "hopeful_despairing", "trusting_suspicious", "energetic_lethargic",
        "nonsexual_sexual"]
PERSONAS = ["loving", "poeticism", "nonchalance", "humor", "mathematical",
            "impulsiveness", "goodness", "sarcasm", "remorse"]


def main():
    with open(AGG_PATH) as f:
        rows = json.load(f)["rows"]
    by_model = {}
    for r in rows:
        by_model.setdefault(r["model"], {})[r["axis"]] = r["2s"]

    baseline = by_model.get("llama-3.1-8b-instruct", {})
    summary = []
    for p in PERSONAS:
        model = f"oct-llama-3.1-8b-{p}"
        if model not in by_model:
            continue
        deltas = []
        for axis in AXES:
            v = by_model[model].get(axis)
            il = baseline.get(axis)
            if v is None or il is None:
                continue
            deltas.append(v - il)
        if not deltas:
            continue
        summary.append({
            "persona": f"OCT-{p}",
            "mean_2s": sum(by_model[model][a] for a in AXES if a in by_model[model]) / 8,
            "mean_delta": sum(deltas) / len(deltas),
            "n_pos": sum(1 for d in deltas if d > 0),
            "n_total": len(deltas),
        })
    # also vanilla baseline
    summary.append({
        "persona": "Llama-Inst (baseline)",
        "mean_2s": sum(baseline.get(a, 0) for a in AXES) / 8,
        "mean_delta": 0.0,
        "n_pos": None,
        "n_total": 8,
    })

    summary.sort(key=lambda r: r["mean_delta"], reverse=True)

    # plot
    fig, ax = plt.subplots(figsize=(10, 5.5))
    names = [r["persona"] for r in summary]
    deltas = [r["mean_delta"] for r in summary]
    colors = ["#dc2626" if r["persona"].startswith("OCT-") and r["mean_delta"] > 0
              else "#2563eb" if r["persona"].startswith("OCT-") and r["mean_delta"] < 0
              else "#6b7280" for r in summary]
    ax.barh(range(len(summary)), deltas, color=colors)
    ax.set_yticks(range(len(summary)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.axvline(0, color="k", linewidth=0.8)
    ax.set_xlabel("mean Δ 2s (user-state) over Llama-Instruct baseline")
    ax.set_title("OCT persona ranking by user-state-prior amplification\n"
                 "Mode 3, mean across 8 emotional axes")
    # annotate with sign counts
    for i, r in enumerate(summary):
        if r["n_pos"] is None:
            continue
        offset = 0.005 if r["mean_delta"] >= 0 else -0.005
        ha = "left" if r["mean_delta"] >= 0 else "right"
        ax.text(r["mean_delta"] + offset, i, f" {r['n_pos']}/{r['n_total']}+",
                va="center", ha=ha, fontsize=9, color="#374151")
    ax.set_xlim(-0.08, 0.16)
    fig.tight_layout()
    fig.savefig(PLOTS / "oct_persona_ranking.png", dpi=140)
    plt.close(fig)
    print(f"[wrote] {PLOTS/'oct_persona_ranking.png'}")
    print("\nRanking (highest Δ first):")
    for r in summary:
        print(f"  {r['persona']:32s} mean Δ={r['mean_delta']:+.3f}  "
              f"sign={r['n_pos']}/{r['n_total']}+" if r['n_pos'] is not None else f"  {r['persona']:32s} (baseline)")


if __name__ == "__main__":
    main()
