"""Analyze colour-leakage cells: diagonal lift, base-vs-instruct contrast, and
correlation with the per-model coinflip two_s.

For each cell we have a confusion matrix M[condition][colour] = mean over contexts
of P(colour) read at the OPEN user turn. The leakage statistic is the mean
diagonal lift:

    lift = mean_X [ M[prime_X][X] - mean_{C != X} M[prime_C][X] ]

Positive lift = priming favourite colour X specifically raises P(X) on the USER
turn, above the per-column base rate set by the other primes. The column-wise
subtraction cancels each colour's unconditional base rate AND the generic "some
colour was just mentioned" salience (every prime mentions a colour equally
recently); what remains is the lift of the SPECIFIC primed colour.

IMPORTANT interpretation. The primed colour word is *lexically* the token we then
measure, so a base model can show positive lift from pure in-context induction
(copy the colour it just saw). A base cell is therefore NOT a null — it is the
recency/induction floor. The self-model-contamination signal of interest is
INSTRUCT-MINUS-BASE, and to keep it from being confounded with the rendering mode
we compare instruct-in-plaintext against base-in-plaintext (same rendering).

We report three cells per model where available:
  base   plaintext        induction floor, no assistant character
  instruct plaintext      same rendering, WITH assistant character  (tier effect)
  instruct open_user_turn real user-turn position (the coinflip position)

CI: bootstrap over the N contexts (resample context_ids with replacement, rebuild
the matrix, recompute the mean diagonal lift); 2.5/97.5 percentiles over B draws.

Usage:
  python3 src/analyze_colour_leakage.py
"""
import argparse
import glob
import json
import os
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
DEFAULT_DIR = ROOT / "results" / "colour_leakage"
COINFLIP = ROOT / "results" / "coinflip_across_models.json"

FAMILY_DISPLAY = {"qwen": "Qwen", "llama": "Llama", "gemma": "Gemma", "olmo": "OLMo"}


def size_b_from_model_id(model_id):
    m = re.search(r"(\d+(?:\.\d+)?)\s*[bB]\b", model_id.replace("-", " "))
    return float(m.group(1)) if m else float("nan")


def tier_of(model_id):
    low = model_id.lower()
    if "instruct" in low or low.endswith("-it") or "-it-" in low or "chat" in low:
        return "instruct"
    return "base"


def cell_lift_and_ci(summary, n_boot=2000, seed=0):
    """Return (point_lift, lo, hi, per_colour_lift) bootstrapping over contexts."""
    kept = summary["colours_kept"]
    by_cc, contexts = {}, []
    for r in summary["per_item"]:
        by_cc[(r["condition"], r["context_id"])] = r["p_by_colour"]
        if r["context_id"] not in contexts:
            contexts.append(r["context_id"])

    def matrix_for(ctx_ids):
        mat = {}
        for cond in [f"prime_{c}" for c in kept]:
            rows = [by_cc[(cond, cid)] for cid in ctx_ids if (cond, cid) in by_cc]
            mat[cond] = {col: float(np.mean([r[col] for r in rows])) for col in kept}
        return mat

    def mean_lift(mat):
        per_colour = {}
        for x in kept:
            diag = mat[f"prime_{x}"][x]
            off = [mat[f"prime_{o}"][x] for o in kept if o != x]
            per_colour[x] = diag - (sum(off) / len(off) if off else 0.0)
        return sum(per_colour.values()) / len(per_colour), per_colour

    point, per_colour = mean_lift(matrix_for(contexts))
    rng = np.random.default_rng(seed)
    ctx_arr = np.array(contexts)
    n = len(contexts)
    boots = [mean_lift(matrix_for(list(ctx_arr[rng.integers(0, n, size=n)])))[0]
             for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi), per_colour


def load_coinflip_rows():
    return json.loads(Path(COINFLIP).read_text())["rows"]


def match_coinflip(rows, family_key, size_b, position, tier):
    disp = FAMILY_DISPLAY.get(family_key, family_key.capitalize())
    for r in rows:
        if (r["family"].startswith(disp) and abs(r["size_B"] - size_b) < 1e-6
                and r["position"] == position and r["tier"] == tier):
            return r["two_s"]
    return None


def pearson(xs, ys):
    if len(xs) < 3:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(DEFAULT_DIR))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out-md", default=str(ROOT / "results" / "colour_leakage_summary.md"))
    ap.add_argument("--out-fig", default=str(ROOT / "results" / "colour_leakage_vs_coinflip.png"))
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "*.json")))
    if not files:
        raise SystemExit(f"no result cells in {args.dir}")
    coin_rows = load_coinflip_rows()

    cells = []
    for f in files:
        s = json.loads(Path(f).read_text())
        size_b = size_b_from_model_id(s["model_id"])
        tier = tier_of(s["model_id"])
        point, lo, hi, per_colour = cell_lift_and_ci(s, n_boot=args.n_boot)
        two_s = match_coinflip(coin_rows, s["family"], size_b, s["mode"], tier)
        cells.append({
            "short": Path(f).stem, "model_id": s["model_id"], "family": s["family"],
            "size_b": size_b, "tier": tier, "mode": s["mode"],
            "n_colours": len(s["colours_kept"]), "lift": point, "lo": lo, "hi": hi,
            "coinflip_two_s": two_s, "per_colour": per_colour,
        })
    cells.sort(key=lambda c: (c["family"], c["size_b"], c["tier"], c["mode"]))

    def model_name(c):
        return f"{FAMILY_DISPLAY.get(c['family'], c['family'])}-{c['size_b']:g}B"

    # index by (family,size) -> {(tier,mode): cell}
    by_ms = {}
    for c in cells:
        by_ms.setdefault((c["family"], c["size_b"]), {})[(c["tier"], c["mode"])] = c

    L = []
    L.append("# Colour-leakage summary\n")
    L.append("Mean diagonal lift of the primed-vs-measured colour confusion matrix, read at "
             "the OPEN user turn. Positive = the assistant's primed favourite colour raises its "
             "own probability on the user turn. **Base plaintext is the induction/recency floor** "
             "(the colour token was mentioned two turns back); the contamination signal is "
             "instruct-minus-base at matched rendering (both plaintext).\n")
    L.append("| cell | tier | mode | #col | lift | 95% CI | coinflip 2s |")
    L.append("|---|---|---|---:|---:|---|---:|")
    for c in cells:
        ci = f"[{c['lo']:+.3f}, {c['hi']:+.3f}]"
        ts = f"{c['coinflip_two_s']:+.3f}" if c["coinflip_two_s"] is not None else "—"
        L.append(f"| {c['short']} | {c['tier']} | {c['mode']} | {c['n_colours']} | "
                 f"{c['lift']:+.4f} | {ci} | {ts} |")

    L.append("\n## Recency-controlled contrast (all plaintext) and real position\n")
    L.append("Δ = instruct_pt − base_pt, both plaintext → tier effect at matched induction. "
             "instruct_out = instruct in open_user_turn (coinflip position).\n")
    L.append("| model | base_pt | instruct_pt | Δ(ins−base) | instruct_out | coinflip 2s (ins,out) |")
    L.append("|---|---:|---:|---:|---:|---:|")
    deltas = []
    for (fam, size), d in sorted(by_ms.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        base_pt = d.get(("base", "plaintext"))
        ins_pt = d.get(("instruct", "plaintext"))
        ins_out = d.get(("instruct", "open_user_turn"))
        name = f"{FAMILY_DISPLAY.get(fam, fam)}-{size:g}B"
        b = base_pt["lift"] if base_pt else None
        ip = ins_pt["lift"] if ins_pt else None
        io = ins_out["lift"] if ins_out else None
        delta = (ip - b) if (ip is not None and b is not None) else None
        cf_out = ins_out["coinflip_two_s"] if ins_out else None
        row = {"model": name, "delta": delta, "instruct_out": io, "coinflip_out": cf_out}
        deltas.append(row)
        fmt = lambda v: f"{v:+.4f}" if v is not None else "—"
        cff = f"{cf_out:+.3f}" if cf_out is not None else "—"
        L.append(f"| {name} | {fmt(b)} | {fmt(ip)} | {fmt(delta)} | {fmt(io)} | {cff} |")

    # correlations vs coinflip two_s (instruct, open_user_turn)
    p_out = [(r["coinflip_out"], r["instruct_out"]) for r in deltas
             if r["coinflip_out"] is not None and r["instruct_out"] is not None]
    p_delta = [(r["coinflip_out"], r["delta"]) for r in deltas
               if r["coinflip_out"] is not None and r["delta"] is not None]
    r_out = pearson([a for a, _ in p_out], [b for _, b in p_out])
    r_delta = pearson([a for a, _ in p_delta], [b for _, b in p_delta])
    L.append(f"\nPearson r (instruct_out colour lift vs coinflip 2s): **{r_out:+.3f}** (n={len(p_out)})")
    L.append(f"\nPearson r (instruct−base Δ vs coinflip 2s): **{r_delta:+.3f}** (n={len(p_delta)})")
    L.append("\nReading: if colour leakage tracked the coinflip bias (general self-model "
             "contamination), these r would be strongly positive and the lifts comparable in "
             "magnitude to the coinflip two_s. If colour lift stays near/below its induction "
             "floor while coinflip two_s is large, the user-turn bias is specific to motivated "
             "content, not generic egocentric projection.\n")

    Path(args.out_md).write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n[saved] {args.out_md}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
        ax = axes[0]
        for c in cells:
            if c["tier"] == "instruct" and c["mode"] == "open_user_turn" and c["coinflip_two_s"] is not None:
                ax.errorbar(c["coinflip_two_s"], c["lift"],
                            yerr=[[c["lift"] - c["lo"]], [c["hi"] - c["lift"]]], fmt="o", capsize=3)
                ax.annotate(model_name(c), (c["coinflip_two_s"], c["lift"]), fontsize=8,
                            xytext=(4, 4), textcoords="offset points")
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.plot([0, 0.7], [0, 0.7], color="lightgray", lw=0.8, ls=":", label="y=x")
        ax.set_xlabel("coinflip two_s (instruct, open_user_turn)")
        ax.set_ylabel("colour diagonal lift (instruct, open_user_turn)")
        ax.set_title(f"colour lift vs coinflip  (r={r_out:+.2f})")
        ax.legend(fontsize=8)

        ax2 = axes[1]
        for r in deltas:
            if r["coinflip_out"] is not None and r["delta"] is not None:
                ax2.plot(r["coinflip_out"], r["delta"], "s")
                ax2.annotate(r["model"], (r["coinflip_out"], r["delta"]), fontsize=8,
                             xytext=(4, 4), textcoords="offset points")
        ax2.axhline(0, color="gray", lw=0.8, ls="--")
        ax2.set_xlabel("coinflip two_s (instruct, open_user_turn)")
        ax2.set_ylabel("colour lift Δ (instruct_pt − base_pt)")
        ax2.set_title(f"recency-controlled leakage vs coinflip  (r={r_delta:+.2f})")
        fig.tight_layout()
        fig.savefig(args.out_fig, dpi=150)
        print(f"[saved] {args.out_fig}")
    except Exception as e:
        print(f"[warn] figure skipped: {e}")


if __name__ == "__main__":
    main()
