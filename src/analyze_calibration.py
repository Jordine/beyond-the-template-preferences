"""Analyze colored-balls calibration sweep.

For each model, group results by (true_p_red, preferred_outcome). Compute
mean predicted P(red) per group. Headline plot: predicted P(red) vs
true P(red) for each (model, preferred_outcome) curve.

If the model is calibrated AND PSM-free: predicted P(red) ≈ true P(red).
If the model has a PSM bias toward `preferred`: the "preferred=red" curve
sits ABOVE the diagonal; "preferred=blue" sits BELOW.

The gap between the two curves at each true_p_red = 2s at that prior.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
RES = ROOT / "results" / "calibration"
PLOTS = ROOT / "results" / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

MODELS = [
    "Llama-3.1-8B-base", "Llama-3.1-8B-Instruct",
    "Qwen2.5-7B-Instruct", "OCT-loving", "OCT-sarcasm",
]


def load(tag):
    p = RES / f"{tag}__colored_balls.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def summarize(model_data):
    """Group results by (true_p_red, preferred_outcome). Return list of dicts."""
    # Each result item has: id (with ratio embedded), p_red_normalized, preferred_outcome
    # And the original item carried n_red, n_blue, true_p_red.
    by_key = {}
    for r in model_data["results"]:
        n_red = r["n_red"]
        n_blue = r["n_blue"]
        tp = r["true_p_red"]
        pref = r["preferred_outcome"]
        key = (tp, pref)
        by_key.setdefault(key, []).append(r.get("p_red_normalized"))
    out = []
    for (tp, pref), vals in sorted(by_key.items()):
        vals = [v for v in vals if v is not None and v == v]
        if not vals:
            continue
        out.append({"true_p_red": tp, "preferred_outcome": pref,
                    "n": len(vals), "mean_p_red": sum(vals) / len(vals)})
    return out


def main():
    all_data = {m: load(m) for m in MODELS}

    # Build calibration plots — one panel per model
    fig, axes = plt.subplots(1, len(MODELS), figsize=(4 * len(MODELS), 4.2), sharey=True)
    if len(MODELS) == 1:
        axes = [axes]
    lines = ["# Colored-balls calibration sweep\n",
             "For each model, P(red) predicted vs true prior. Curve gap = 2s at that prior.\n"]
    summary_tables = {}
    for ax, m in zip(axes, MODELS):
        d = all_data[m]
        if d is None:
            ax.set_title(f"{m}\n(missing)")
            continue
        s = summarize(d)
        summary_tables[m] = s
        true_ps = sorted({row["true_p_red"] for row in s})
        for pref, color in [("red", "#dc2626"), ("blue", "#2563eb")]:
            xs, ys = [], []
            for row in s:
                if row["preferred_outcome"] == pref:
                    xs.append(row["true_p_red"])
                    ys.append(row["mean_p_red"])
            ax.plot(xs, ys, marker="o", color=color, label=f"preferred = {pref}")
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.7, alpha=0.5, label="calibrated")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("true P(red)")
        ax.set_title(m)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("model P(red)")
    axes[-1].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS / "calibration_curves.png", dpi=140)
    plt.close(fig)
    print(f"[wrote] {PLOTS/'calibration_curves.png'}")

    # Markdown table: per-model, per-prior, 2s at that prior
    lines.append("\n## 2s as a function of true prior\n")
    lines.append("Defined: `2s(p) = mean_P_red(preferred=red, prior=p) − mean_P_red(preferred=blue, prior=p)`.")
    lines.append("Calibrated-and-PSM-free model: 2s ≈ 0 at all priors. Tie-breaking PSM: 2s peaks at p=0.5. Additive PSM: 2s roughly constant across priors.\n")
    lines.append("| model | p=0.1 | p=0.3 | p=0.5 | p=0.7 | p=0.9 | mean across priors |")
    lines.append("|---|---|---|---|---|---|---|")
    for m in MODELS:
        s = summary_tables.get(m, [])
        per_p = {}
        for row in s:
            per_p.setdefault(row["true_p_red"], {})[row["preferred_outcome"]] = row["mean_p_red"]
        cells = []
        s_vals = []
        for tp in [0.1, 0.3, 0.5, 0.7, 0.9]:
            d = per_p.get(tp, {})
            if "red" in d and "blue" in d:
                two_s = d["red"] - d["blue"]
                s_vals.append(two_s)
                cells.append(f"{two_s:+.3f}")
            else:
                cells.append("—")
        mean_s = (sum(s_vals) / len(s_vals)) if s_vals else None
        mean_cell = f"**{mean_s:+.3f}**" if mean_s is not None else "—"
        lines.append(f"| {m} | " + " | ".join(cells) + f" | {mean_cell} |")

    out_path = RES / "analysis.md"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"[wrote] {out_path}")


if __name__ == "__main__":
    main()
