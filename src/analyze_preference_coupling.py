"""Preference-coupling analysis — does the assistant's stated pairwise task
preference predict the user-turn coin-flip bias?

Upgrade/companion to analyze_deprefer_coupling.py: x is now the pair-shown
preference log-odds from run_assistant_preference.py rather than the isolated
refusal disposition. Headline is task-level (per-harmful-task means, n=10 per
cell, directly comparable to the refusal r); the pair-level n=100 correlation
is reported descriptively (pairs share tasks, so they are not independent).

Usage:
  python src/analyze_preference_coupling.py \
    --cell "Llama 3.1 8B It:results/preference/llama-3.1-8b-instruct.json:results/deprefer/llama-3.1-8b-instruct.json:results/coinflip_instruct/llama-3.1-8b-instruct__open_user_turn.json" \
    --cell "Qwen 2.5 14B It:...:...:..." \
    --dataset data/psm_coinflip_user_messages.json \
    --out results/preference_coupling.png

Pass "-" for the deprefer path to skip the refusal side-by-side.
"""
import argparse
import collections
import json
import os
import statistics
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy import stats as _st
except Exception:
    _st = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_deprefer_coupling import pearson, spearman, per_pair_bias  # noqa: E402


def twoway_clustered_r(x, y, keys):
    """Pair-level correlation with an honest p-value: OLS of z(y) on z(x)
    (slope == Pearson r), SE via Cameron-Gelbach-Miller two-way clustering on
    harmless-task and harmful-task identity. The 100 pairs reuse 10+10 tasks,
    so naive n=100 inference is pseudoreplicated; clustering prices that in.
    Inference on t with df = min(G_h, G_f) - 1 (conservative, standard CGM).

    keys: list of (harmless_idx, harmful_idx) aligned with x, y. Task indices
    shared across stacked cells cluster together, which is what we want: the
    same underlying task drives correlated errors in every cell it appears in.
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.std() == 0 or y.std() == 0:
        return dict(r=float("nan"), se=float("nan"), t=float("nan"),
                    p=float("nan"), df=0, n=len(x))
    zx = (x - x.mean()) / x.std(); zy = (y - y.mean()) / y.std()
    n = len(zx)
    X = np.column_stack([np.ones(n), zx])
    k = X.shape[1]
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ zy)
    e = zy - X @ beta

    def cluster_cov(groups):
        idx_by_g = collections.defaultdict(list)
        for i, g in enumerate(groups):
            idx_by_g[g].append(i)
        G = len(idx_by_g)
        S = np.zeros((k, k))
        for idxs in idx_by_g.values():
            u = X[idxs].T @ e[idxs]
            S += np.outer(u, u)
        corr = (G / (G - 1)) * ((n - 1) / (n - k)) if G > 1 else 1.0
        return XtX_inv @ S @ XtX_inv * corr, G

    V_h, G_h = cluster_cov([h for h, f in keys])
    V_f, G_f = cluster_cov([f for h, f in keys])
    V_hf, _ = cluster_cov(list(range(n)))  # intersection clusters are singletons
    var = (V_h + V_f - V_hf)[1, 1]
    df = min(G_h, G_f) - 1
    if var <= 0 or df < 1:
        return dict(r=float(beta[1]), se=float("nan"), t=float("nan"),
                    p=float("nan"), df=df, n=n)
    se = float(np.sqrt(var))
    t = float(beta[1]) / se
    p = float(2 * _st.t.sf(abs(t), df)) if _st is not None else float("nan")
    return dict(r=float(beta[1]), se=se, t=t, p=p, df=df, n=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", action="append", required=True,
                    help="LABEL:PREF_JSON:DEPREFER_JSON:COINFLIP_JSON")
    ap.add_argument("--dataset", default="data/psm_coinflip_user_messages.json")
    ap.add_argument("--out", default="results/preference_coupling.png")
    args = ap.parse_args()

    dataset = json.load(open(args.dataset))
    cells = []
    for spec in args.cell:
        label, pref_path, dep_path, coin_path = spec.split(":", 3)
        pref = json.load(open(pref_path))
        by_pair = collections.defaultdict(list)
        cap, nei = [], []
        for r in pref["results"]:
            by_pair[(r["harmless_idx"], r["harmful_idx"])].append(r["log_odds"])
            cap.append(r["captured_mass"]); nei.append(r["neither_mass"])
        pair_pref = {k: statistics.mean(v) for k, v in by_pair.items()}  # 2 orderings
        deprefer = None
        if dep_path != "-":
            dep = json.load(open(dep_path))
            deprefer = {int(k): v["deprefer"] for k, v in dep["harmful"].items()}
        perpair_bias, two_s = per_pair_bias(coin_path, dataset)
        cells.append((label, two_s, pair_pref, deprefer, perpair_bias,
                      statistics.mean(cap), statistics.mean(nei)))

    n = len(cells)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.0), squeeze=False)
    print("\n=== Preference (pairwise A/B) vs user-turn safe-bias ===")
    pooled_zx, pooled_zy = [], []
    rows = []
    cl_specs = []
    for ax, (label, two_s, pair_pref, deprefer, perpair_bias, mcap, mnei) in zip(axes[0], cells):
        by_harm_pref = collections.defaultdict(list)
        by_harm_bias = collections.defaultdict(list)
        for (h, f), v in pair_pref.items():
            by_harm_pref[f].append(v)
        for (h, f), b in perpair_bias.items():
            by_harm_bias[f].append(b)
        fs = sorted(set(by_harm_pref) & set(by_harm_bias))
        x = [statistics.mean(by_harm_pref[f]) for f in fs]
        y = [statistics.mean(by_harm_bias[f]) for f in fs]
        r, p, nn = pearson(x, y)
        rho, ps = spearman(x, y)

        common = sorted(set(pair_pref) & set(perpair_bias))
        pr, pp, pn = pearson([pair_pref[k] for k in common],
                             [perpair_bias[k] for k in common])
        cl = twoway_clustered_r([pair_pref[k] for k in common],
                                [perpair_bias[k] for k in common], common)
        cl_specs.append((label, common,
                         [pair_pref[k] for k in common],
                         [perpair_bias[k] for k in common],
                         deprefer))

        rr = rrho = rp = rps = float("nan")
        r_pd = float("nan")
        if deprefer is not None:
            xr = [deprefer[f] for f in fs]
            rr, rp, _ = pearson(xr, y)
            rrho, rps = spearman(xr, y)
            r_pd, _, _ = pearson(x, xr)  # do the two probes agree with each other?

        xa, ya = np.asarray(x, float), np.asarray(y, float)
        if xa.std() > 0 and ya.std() > 0:
            pooled_zx.extend(((xa - xa.mean()) / xa.std()).tolist())
            pooled_zy.extend(((ya - ya.mean()) / ya.std()).tolist())

        for k in common:
            ax.scatter(pair_pref[k], perpair_bias[k], s=10, alpha=0.18,
                       color="tab:gray", zorder=1)
        ax.scatter(x, y, s=70, color="tab:blue", zorder=3, edgecolor="k", linewidth=0.5)
        if np.std(x) > 0:
            m, c = np.polyfit(x, y, 1)
            xs = np.linspace(min(x), max(x), 50)
            ax.plot(xs, m * xs + c, color="tab:red", lw=1.5, zorder=2)
        ax.axhline(0, color="k", lw=0.5, ls=":")
        ax.set_title(f"{label}\ncell 2s={two_s:+.2f}  task r={r:+.2f} (p={p:.3f})\n"
                     f"pair r={cl['r']:+.2f} (p={cl['p']:.3f}, 2-way clustered)",
                     fontsize=9)
        ax.set_xlabel("preference log-odds\nlog P(harmless letter) - log P(harmful letter)",
                      fontsize=8)
        ax.set_ylabel("user-turn safe-bias for that harmful task", fontsize=8)

        rows.append((label, two_s, r, p, rho, pr, pn, rr, rrho, r_pd, mcap, mnei))

    print(f"{'cell':24s} {'2s':>7s} {'pref r':>8s} {'p':>7s} {'rho':>7s} "
          f"{'pair r':>8s} {'n':>4s} {'refus r':>8s} {'refus rho':>9s} "
          f"{'pref~dep':>8s} {'capt':>6s} {'neith':>7s}")
    for (label, two_s, r, p, rho, pr, pn, rr, rrho, r_pd, mcap, mnei) in rows:
        print(f"{label:24s} {two_s:+.3f} {r:+8.3f} {p:7.4f} {rho:+7.3f} "
              f"{pr:+8.3f} {pn:4d} {rr:+8.3f} {rrho:+9.3f} "
              f"{r_pd:+8.3f} {mcap:6.3f} {mnei:7.4f}")

    if len(pooled_zx) >= 3:
        zr, zp, zn = pearson(pooled_zx, pooled_zy)
        zrho, zrhop = spearman(pooled_zx, pooled_zy)
        print(f"\n[pooled z-scored across {len(cells)} cells]  "
              f"Pearson r={zr:+.3f} (p={zp:.4g}, n={zn})  "
              f"Spearman rho={zrho:+.3f} (p={zrhop:.4g})")

    print("\n=== Pair-level r, CGM two-way clustered SEs "
          "(clusters: harmless task x harmful task) ===")
    stacks = {"preference": ([], [], []), "refusal": ([], [], [])}
    for label, common, xp, yp, deprefer in cl_specs:
        parts = []
        for name, xv in (("preference", xp),
                         ("refusal", [deprefer[f] for (h, f) in common]
                          if deprefer is not None else None)):
            if xv is None:
                continue
            res = twoway_clustered_r(xv, yp, common)
            parts.append(f"{name}: r={res['r']:+.3f} SE={res['se']:.3f} "
                         f"t={res['t']:+.2f} p={res['p']:.4f} "
                         f"(df={res['df']}, n={res['n']})")
            xa, ya = np.asarray(xv, float), np.asarray(yp, float)
            if xa.std() > 0 and ya.std() > 0:
                stacks[name][0].extend(((xa - xa.mean()) / xa.std()).tolist())
                stacks[name][1].extend(((ya - ya.mean()) / ya.std()).tolist())
                stacks[name][2].extend(common)
        print(f"{label:24s} " + "   ".join(parts))
    for name, (sx, sy, sk) in stacks.items():
        if len(sx) >= 3:
            res = twoway_clustered_r(sx, sy, sk)
            print(f"{'[pooled %d cells]' % len(cells):24s} {name}: "
                  f"r={res['r']:+.3f} SE={res['se']:.3f} t={res['t']:+.2f} "
                  f"p={res['p']:.4f} (df={res['df']}, n={res['n']})")

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
