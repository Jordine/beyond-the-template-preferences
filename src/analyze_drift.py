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

from analyze_psm_coinflip import (twoway_clustered_2s, cluster_key_from_id,
                                  clustered_se_from_results)

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


def cell_se(results):
    """CGM two-way task-clustered SE, iid analytical fallback."""
    se = clustered_se_from_results(results)
    return se if se is not None else analytical_se(results)


def clustered_delta(res_c, res_b):
    """Paired, task-clustered delta: per-item dq = q_c - q_b matched by id,
    then the same two-way clustered OLS of dq on z. Pairing removes the
    shared per-item variance that the unpaired sqrt(se_c^2+se_b^2) double
    counts; clustering prices in the 10x10 task reuse.
    Returns (delta, se) or None."""
    def q_of(r):
        ph, pt = r.get("p_heads_aggregated"), r.get("p_tails_aggregated")
        if ph is None or pt is None or ph + pt <= 0:
            return None
        return ph / (ph + pt)

    qb = {}
    for r in res_b:
        q = q_of(r)
        if q is not None:
            qb[r["id"]] = q
    dqs, prefs, keys = [], [], []
    for r in res_c:
        q = q_of(r)
        if q is None or r["id"] not in qb:
            continue
        key = cluster_key_from_id(r["id"])
        if key is None:
            return None
        dqs.append(q - qb[r["id"]])
        prefs.append(r["preferred_outcome"])
        keys.append(key)
    if len(dqs) < 4 or len(set(prefs)) < 2:
        return None
    out = twoway_clustered_2s(dqs, prefs, keys)
    return out["two_s"], out["se"]


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
    out_lines.append("## per-cell harmless-option-bias (two_s) +/- SE\n")
    out_lines.append("SEs are CGM two-way task-clustered on (harmless task, harmful task); "
                     "delta SEs are paired per-item and task-clustered.\n")
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
            se = cell_se(d["results"])
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
        for c in delta_cols:
            d_c = cells.get((tag, c))
            if d_c is None:
                row.append("—")
                continue
            pd = clustered_delta(d_c["results"], d_b0["results"])
            if pd is None:
                row.append("—")
                continue
            delta, se_delta = pd
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
        for c in delta_cols:
            d_c = cells.get((tag, c))
            if d_c is None:
                row.append("—")
                continue
            pd = clustered_delta(d_c["results"], d_b1["results"])
            if pd is None:
                row.append("—")
                continue
            delta, se_delta = pd
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
        for c in delta_cols:
            d_c = cells.get((tag, c))
            if d_c is None:
                row.append("—")
                continue
            pd = clustered_delta(d_c["results"], d_b1l["results"])
            if pd is None:
                row.append("—")
                continue
            delta, se_delta = pd
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
        display_names = {
            "Llama-3.1-8B-base": "Llama 3.1 8B base",
            "Llama-3.1-8B-Instruct": "Llama 3.1 8B Instruct",
            "Olmo-3-1125-32B-base": "OLMo 3 32B base",
            "Olmo-3.1-32B-Instruct": "OLMo 3.1 32B Instruct",
            "Olmo-3.1-32B-Instruct-DPO": "OLMo 3.1 32B DPO",
            "Olmo-3.1-32B-Instruct-SFT": "OLMo 3.1 32B SFT",
            "Qwen2.5-14B-base": "Qwen 2.5 14B base",
            "Qwen2.5-14B-Instruct": "Qwen 2.5 14B Instruct",
        }
        preferred_order = [
            "Llama-3.1-8B-base", "Llama-3.1-8B-Instruct",
            "Qwen2.5-14B-base", "Qwen2.5-14B-Instruct",
            "Olmo-3-1125-32B-base", "Olmo-3.1-32B-Instruct-SFT",
            "Olmo-3.1-32B-Instruct-DPO", "Olmo-3.1-32B-Instruct",
        ]
        tags = sorted(tags, key=lambda t: (preferred_order.index(t) if t in preferred_order else len(preferred_order), t))
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
                    errs.append(cell_se(d["results"]))
            offset = (i - (n_conds - 1) / 2) * bar_w
            ax.bar(x + offset, vals, bar_w, yerr=errs, label=f"{c}: {cond_labels[c]}",
                   color=cond_colors[c], capsize=2)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([display_names.get(t, t) for t in tags], rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("harmless-option bias  (mean q|H − mean q|T)")
        ax.set_title({"open_user_turn": "open user turn", "plaintext": "plaintext"}.get(args.mode, args.mode))
        ax.legend(fontsize=8, ncol=2, loc="best")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        fig.savefig(args.out_fig, dpi=150, bbox_inches="tight")
        print(f"[wrote] {args.out_fig}")


if __name__ == "__main__":
    main()
