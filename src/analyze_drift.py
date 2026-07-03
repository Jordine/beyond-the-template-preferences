"""Analyze persona-drift pilot results.

Reads results/persona_drift/<tag>__<mode>__<cond>.json and emits:
  - Markdown summary table per model x condition (two_s, SE, b)
  - Delta table: condition - B0 with SE on the difference
  - Optional: matplotlib figure (grouped bars per model, error bars)
"""
import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).parent.parent
RES = ROOT / "results" / "persona_drift"


def analytical_se(results):
    """Analytical SE for the diff-of-means estimator two_s = mean(q|pref=H) - mean(q|pref=T)."""
    qH = [r["q_heads_normalised"] for r in results
          if r["preferred_outcome"] == "heads" and r["q_heads_normalised"] == r["q_heads_normalised"]]
    qT = [r["q_heads_normalised"] for r in results
          if r["preferred_outcome"] == "tails" and r["q_heads_normalised"] == r["q_heads_normalised"]]
    if len(qH) < 2 or len(qT) < 2:
        return float("nan")
    def var_(xs):
        m = sum(xs) / len(xs)
        return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var_(qH) / len(qH) + var_(qT) / len(qT))


def load_cells(mode):
    cells = {}  # (tag, cond) -> summary
    for f in sorted(RES.glob(f"*__{mode}__*.json")):
        d = json.loads(f.read_text())
        cells[(d["tag"], d["condition"])] = d
    return cells


def fmt(x, sig=3):
    if x != x:
        return "—"
    return f"{x:+.{sig}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="open_user_turn")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--out-md", default=str(ROOT / "results" / "persona_drift_summary.md"))
    ap.add_argument("--out-fig", default=str(ROOT / "results" / "persona_drift.png"))
    args = ap.parse_args()

    cells = load_cells(args.mode)
    if not cells:
        print(f"no cells found for mode={args.mode} under {RES}")
        return

    tags = sorted({tag for tag, _ in cells.keys()})
    conds = ["B0", "B1", "B1L", "Eu", "Gu", "Ea", "Ga"]
    cond_labels = {
        "B0":  "single-turn baseline",
        "B1":  "multi-turn N+N (short)",
        "B1L": "multi-turn N+N_long (length-matched)",
        "Eu":  "evil user",
        "Gu":  "good user",
        "Ea":  "evil assistant",
        "Ga":  "good assistant",
    }

    out_lines = [f"# persona-drift summary  (mode={args.mode})\n"]
    out_lines.append("## per-cell harmless-option-bias (two_s) +/- analytical SE\n")
    header = "| model | " + " | ".join(conds) + " |"
    sep = "|---|" + "---|" * len(conds)
    out_lines.append(header)
    out_lines.append(sep)
    for tag in tags:
        row = [f"`{tag}`"]
        for c in conds:
            d = cells.get((tag, c))
            if d is None:
                row.append("—")
                continue
            se = analytical_se(d["results"])
            row.append(f"{fmt(d['two_s'])} ± {se:.3f}")
        out_lines.append("| " + " | ".join(row) + " |")
    out_lines.append("")

    # Delta vs B0 (single-turn baseline)
    out_lines.append("## delta vs single-turn baseline (B0)\n")
    delta_cols = ["B1", "Eu", "Gu", "Ea", "Ga"]
    out_lines.append("| model | " + " | ".join([f"Δ{c}" for c in delta_cols]) + " |")
    out_lines.append("|---|" + "---|" * len(delta_cols))
    for tag in tags:
        row = [f"`{tag}`"]
        d_b0 = cells.get((tag, "B0"))
        if d_b0 is None:
            row += ["—"] * len(delta_cols)
            out_lines.append("| " + " | ".join(row) + " |")
            continue
        se_b0 = analytical_se(d_b0["results"])
        for c in delta_cols:
            d_c = cells.get((tag, c))
            if d_c is None:
                row.append("—")
                continue
            se_c = analytical_se(d_c["results"])
            delta = d_c["two_s"] - d_b0["two_s"]
            se_delta = math.sqrt(se_c ** 2 + se_b0 ** 2)
            row.append(f"{fmt(delta)} ± {se_delta:.3f}")
        out_lines.append("| " + " | ".join(row) + " |")
    out_lines.append("")

    # Delta vs B1 (multi-turn baseline) — isolates the prefix-valence effect
    out_lines.append("## delta vs multi-turn-neutral baseline (B1, short asst) — Eu/Gu clean, Ea/Ga confounded by length\n")
    delta_cols = ["B1L", "Eu", "Gu", "Ea", "Ga"]
    out_lines.append("| model | " + " | ".join([f"Δ{c}" for c in delta_cols]) + " |")
    out_lines.append("|---|" + "---|" * len(delta_cols))
    for tag in tags:
        row = [f"`{tag}`"]
        d_b1 = cells.get((tag, "B1"))
        if d_b1 is None:
            row += ["—"] * len(delta_cols)
            out_lines.append("| " + " | ".join(row) + " |")
            continue
        se_b1 = analytical_se(d_b1["results"])
        for c in delta_cols:
            d_c = cells.get((tag, c))
            if d_c is None:
                row.append("—")
                continue
            se_c = analytical_se(d_c["results"])
            delta = d_c["two_s"] - d_b1["two_s"]
            se_delta = math.sqrt(se_c ** 2 + se_b1 ** 2)
            row.append(f"{fmt(delta)} ± {se_delta:.3f}")
        out_lines.append("| " + " | ".join(row) + " |")
    out_lines.append("")

    # Delta vs B1L (length-matched neutral asst) — clean valence test for Ea/Ga
    out_lines.append("## delta vs length-matched baseline (B1L, long asst) — Ea/Ga isolated from length confound\n")
    delta_cols = ["Ea", "Ga"]
    out_lines.append("| model | " + " | ".join([f"Δ{c}" for c in delta_cols]) + " |")
    out_lines.append("|---|" + "---|" * len(delta_cols))
    for tag in tags:
        row = [f"`{tag}`"]
        d_b1l = cells.get((tag, "B1L"))
        if d_b1l is None:
            row += ["—"] * len(delta_cols)
            out_lines.append("| " + " | ".join(row) + " |")
            continue
        se_b1l = analytical_se(d_b1l["results"])
        for c in delta_cols:
            d_c = cells.get((tag, c))
            if d_c is None:
                row.append("—")
                continue
            se_c = analytical_se(d_c["results"])
            delta = d_c["two_s"] - d_b1l["two_s"]
            se_delta = math.sqrt(se_c ** 2 + se_b1l ** 2)
            row.append(f"{fmt(delta)} ± {se_delta:.3f}")
        out_lines.append("| " + " | ".join(row) + " |")
    out_lines.append("")

    out_path = Path(args.out_md)
    out_path.write_text("\n".join(out_lines))
    print("\n".join(out_lines))
    print(f"\n[wrote] {out_path}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("[skip-plot] matplotlib/numpy not installed")
            return
        n_tags = len(tags)
        n_conds = len(conds)
        bar_w = 0.11
        x = np.arange(n_tags)
        cond_colors = {
            "B0":  "#666666",
            "B1":  "#999999",
            "B1L": "#cccccc",
            "Eu":  "#c44e52",
            "Gu":  "#4c72b0",
            "Ea":  "#dd8452",
            "Ga":  "#55a868",
        }
        fig, ax = plt.subplots(figsize=(max(6, 1.8 * n_tags), 5))
        for i, c in enumerate(conds):
            vals = []
            errs = []
            for tag in tags:
                d = cells.get((tag, c))
                if d is None:
                    vals.append(float("nan"))
                    errs.append(float("nan"))
                else:
                    vals.append(d["two_s"])
                    errs.append(analytical_se(d["results"]))
            offset = (i - (n_conds - 1) / 2) * bar_w
            ax.bar(x + offset, vals, bar_w, yerr=errs, label=f"{c}: {cond_labels[c]}",
                   color=cond_colors[c], capsize=2)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(tags, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("harmless-option bias  (mean q|H − mean q|T)")
        ax.set_title(f"persona-drift pilot — mode={args.mode}")
        ax.legend(fontsize=8, ncol=2, loc="best")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        fig.savefig(args.out_fig, dpi=150, bbox_inches="tight")
        print(f"[wrote] {args.out_fig}")


if __name__ == "__main__":
    main()
