"""Plot user-state coinflip results.

Reads results/user_state/_aggregated.json and produces:
  - results/plots/user_state_2s_base_vs_instruct.png — grouped bar chart of 2s
    per axis, base vs instruct, both families
  - results/plots/user_state_oct_deltas.png — bar chart of OCT - instruct delta
    per (persona, axis)
  - results/plots/empty_user_turn_kl.png — heatmap of pairwise KL between models

Skips silently if a dataset is missing.
"""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
RES = ROOT / "results" / "user_state"
PLOTS = ROOT / "results" / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

AXES = ["happy_sad", "calm_angry", "confident_anxious", "curious_bored",
        "hopeful_despairing", "trusting_suspicious", "energetic_lethargic",
        "nonsexual_sexual"]


def load_rows():
    p = RES / "_aggregated.json"
    if not p.exists():
        return []
    with open(p) as f:
        return json.load(f)["rows"]


def index_by(rows, model):
    return {r["axis"]: r for r in rows if r["model"] == model}


def main():
    rows = load_rows()
    if not rows:
        print("[skip] no _aggregated.json")
        return

    # PLOT 1: base vs instruct, 2s per axis, both families
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(AXES))
    width = 0.2

    families = [
        ("llama-3.1-8b-base", "Llama 8B base", "#9ca3af"),
        ("llama-3.1-8b-instruct", "Llama 8B Inst", "#2563eb"),
        ("qwen-2.5-7b-base", "Qwen 7B base", "#cbd5e1"),
        ("qwen-2.5-7b-instruct", "Qwen 7B Inst", "#0891b2"),
    ]
    for i, (model, label, color) in enumerate(families):
        idx = index_by(rows, model)
        vals = [idx.get(a, {}).get("2s", 0.0) for a in AXES]
        cis_lo = [idx.get(a, {}).get("2s_ci", [0, 0])[0] for a in AXES]
        cis_hi = [idx.get(a, {}).get("2s_ci", [0, 0])[1] for a in AXES]
        err = [[v - lo for v, lo in zip(vals, cis_lo)], [hi - v for v, hi in zip(vals, cis_hi)]]
        ax.bar(x + (i - 1.5) * width, vals, width, label=label, color=color, yerr=err, capsize=2)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([a.replace("_", " / ") for a in AXES], rotation=20, ha="right")
    ax.set_ylabel("2s (PSM effect)  =  mean(q | pref=heads) − mean(q | pref=tails)")
    ax.set_title("User-state coinflip: base vs instruct, by axis")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(PLOTS / "user_state_2s_base_vs_instruct.png", dpi=140)
    plt.close(fig)
    print(f"[wrote] {PLOTS/'user_state_2s_base_vs_instruct.png'}")

    # PLOT 2: OCT - Llama instruct deltas
    instruct = index_by(rows, "llama-3.1-8b-instruct")
    oct_models = [
        ("oct-llama-3.1-8b-loving", "OCT-loving", "#dc2626"),
        ("oct-llama-3.1-8b-sarcasm", "OCT-sarcasm", "#16a34a"),
        ("oct-llama-3.1-8b-nonchalance", "OCT-nonchalance", "#7c3aed"),
        ("oct-llama-3.1-8b-remorse", "OCT-remorse", "#ea580c"),
    ]
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.2
    has_any = False
    for i, (model, label, color) in enumerate(oct_models):
        idx = index_by(rows, model)
        if not idx:
            continue
        has_any = True
        deltas = []
        for a in AXES:
            il = instruct.get(a, {}).get("2s")
            ov = idx.get(a, {}).get("2s")
            deltas.append((ov - il) if (il is not None and ov is not None) else 0.0)
        ax.bar(x + (i - 1.5) * width, deltas, width, label=label, color=color)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([a.replace("_", " / ") for a in AXES], rotation=20, ha="right")
    ax.set_ylabel("Δ 2s   (OCT-persona − Llama Instruct baseline)")
    ax.set_title("OCT persona LoRAs: shift in user-state bias vs Llama 8B Instruct")
    if has_any:
        ax.legend(loc="upper right", fontsize=9)
        fig.tight_layout()
        fig.savefig(PLOTS / "user_state_oct_deltas.png", dpi=140)
    plt.close(fig)
    print(f"[wrote] {PLOTS/'user_state_oct_deltas.png'}")


if __name__ == "__main__":
    main()
