"""Analyze the RQ1.B dose-response sweep.

Reads results/persona_drift_dose/<tag>__<mode>__<cond>.json where <cond> is
{Eu,Gu,Nu}_{1,2,4,8}, and emits:
  - bias-vs-dose table per (model, valence): safe-bias (2s) +/- CGM two-way
    task-clustered SE
  - dose-response delta table: Δ(d) = bias(Eu_d) − bias(Nu_d) and the
    Gu analog, with paired per-item task-clustered SE on the difference
    (length-matched neutral is the control)
  - optional matplotlib figure: safe-bias vs dose, one panel per model

All numbers are on the harmless-option-bias (2s) scale, matching the rest of
the paper. mean P(harmless) = 0.5 + bias/2 if you need the affine conversion.

Readout: Δ(d) monotone in d ⇒ evidence accumulates; flat ⇒ the first declaration
already saturates the shift.
"""
import argparse
import json
from pathlib import Path

from analyze_drift import cell_se, clustered_delta

ROOT = Path(__file__).parent.parent
RES = ROOT / "results" / "persona_drift_dose"
DOSES = [1, 2, 4, 8]
VALENCES = ["Eu", "Gu", "Nu"]


def load_cells(mode):
    cells = {}  # (tag, cond) -> summary
    for f in sorted(RES.glob(f"*__{mode}__*.json")):
        d = json.loads(f.read_text())
        cells[(d["tag"], d["condition"])] = d
    return cells


def cell_stats(d):
    """Return (two_s, se_two_s). SE is CGM two-way task-clustered
    (iid analytical fallback)."""
    return d["two_s"], cell_se(d["results"])


def paired_delta_2s(cv, cn):
    """Paired task-clustered delta on the safe-bias (2s) scale.
    Returns (delta, se) or None."""
    return clustered_delta(cv["results"], cn["results"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="open_user_turn")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--out-md", default=str(ROOT / "results" / "dose_summary.md"))
    ap.add_argument("--out-fig", default=str(ROOT / "results" / "dose_response.png"))
    args = ap.parse_args()

    cells = load_cells(args.mode)
    if not cells:
        print(f"no cells found for mode={args.mode} under {RES}")
        return
    tags = sorted({tag for tag, _ in cells})

    out = [f"# RQ1.B dose-response  (mode={args.mode})\n",
           "Manipulated variable: number of distinct evidence exchanges d before the "
           "coinflip turn. Readout: harmless-option bias (2s scale, matching the paper; "
           "mean P(harmless) = 0.5 + bias/2). "
           "Nu_d = length-matched neutral control at each dose.\n"]

    # ---- per-(model,valence) safe-bias vs dose ----
    out.append("## safe-bias (2s) vs dose  (± CGM two-way task-clustered SE)\n")
    hdr = "| model | valence | " + " | ".join(f"d={d}" for d in DOSES) + " |"
    out.append(hdr)
    out.append("|---|---|" + "---|" * len(DOSES))
    for tag in tags:
        for val in VALENCES:
            row = [f"`{tag}`", val]
            for d in DOSES:
                c = cells.get((tag, f"{val}_{d}"))
                if c is None:
                    row.append("—")
                    continue
                b2s, se2s = cell_stats(c)
                row.append(f"{b2s:+.3f} ± {se2s:.3f}")
            out.append("| " + " | ".join(row) + " |")
    out.append("")

    # ---- dose-response Δ vs length-matched neutral ----
    out.append("## Δ(d) = bias(valence_d) − bias(Nu_d)  (± paired task-clustered SE)\n")
    out.append("Negative Eu Δ = evil evidence lowers the safety bias; magnitude growing "
               "with d = evidence accumulates.\n")
    out.append("| model | Δ | " + " | ".join(f"d={d}" for d in DOSES) + " |")
    out.append("|---|---|" + "---|" * len(DOSES))
    for tag in tags:
        for val in ("Eu", "Gu"):
            row = [f"`{tag}`", f"Δ{val}"]
            for d in DOSES:
                cv = cells.get((tag, f"{val}_{d}"))
                cn = cells.get((tag, f"Nu_{d}"))
                if cv is None or cn is None:
                    row.append("—")
                    continue
                pd = paired_delta_2s(cv, cn)
                if pd is None:
                    row.append("—")
                    continue
                delta, se_delta = pd
                row.append(f"{delta:+.3f} ± {se_delta:.3f}")
            out.append("| " + " | ".join(row) + " |")
    out.append("")

    Path(args.out_md).write_text("\n".join(out))
    print("\n".join(out))
    print(f"\n[wrote] {args.out_md}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("[skip-plot] matplotlib not installed")
            return
        colors = {"Eu": "#c44e52", "Gu": "#4c72b0", "Nu": "#888888"}
        n = len(tags)
        fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4), squeeze=False)
        for j, tag in enumerate(tags):
            ax = axes[0][j]
            for val in VALENCES:
                xs, ys, es = [], [], []
                for d in DOSES:
                    c = cells.get((tag, f"{val}_{d}"))
                    if c is None:
                        continue
                    b2s, se2s = cell_stats(c)
                    xs.append(d); ys.append(b2s); es.append(se2s)
                if xs:
                    ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3,
                                color=colors[val], label=val)
            ax.set_xscale("log", base=2)
            ax.set_xticks(DOSES); ax.set_xticklabels(DOSES)
            ax.axhline(0.0, color="black", lw=0.5, ls=":")
            ax.set_title(tag, fontsize=9)
            ax.set_xlabel("dose (evidence exchanges)")
            if j == 0:
                ax.set_ylabel("harmless-option bias (2s)")
            ax.grid(alpha=0.3); ax.legend(fontsize=8)
        fig.suptitle(f"RQ1.B dose-response — mode={args.mode}")
        plt.tight_layout()
        fig.savefig(args.out_fig, dpi=150, bbox_inches="tight")
        print(f"[wrote] {args.out_fig}")


if __name__ == "__main__":
    main()
