"""B2 analysis — couple assistant deprefer-strength to the user-turn coin-flip bias.

Per model: x = deprefer-strength of each harmful task (10 tasks, from
run_assistant_deprefer.py); y = mean user-turn safe-bias induced when that
harmful task is the dispreferred member (averaged over its 10 harmless partners,
which controls for harmless appeal in the balanced 10x10 design). Correlate.

Also emits the 100-pair scatter (per-pair bias vs the deprefer of the pair's
harmful member) coloured by harmful task.

Usage:
  python src/analyze_deprefer_coupling.py \
    --cell "Llama 3.1 8B It:results/deprefer/llama8b.json:results/coinflip_instruct/llama-3.1-8b-instruct__open_user_turn.json" \
    --cell "Qwen 2.5 14B It:...:..." \
    --dataset data/psm_coinflip_user_messages.json \
    --out results/deprefer_coupling.png
"""
import argparse
import collections
import json
import statistics

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy import stats as _st
except Exception:
    _st = None


def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    r = float(np.corrcoef(x, y)[0, 1])
    n = len(x)
    if _st is not None:
        _, p = _st.pearsonr(x, y)
    else:
        t = r * np.sqrt((n - 2) / max(1e-12, 1 - r * r))
        p = float("nan")
    return r, float(p), n


def spearman(x, y):
    if _st is not None:
        rho, p = _st.spearmanr(x, y)
        return float(rho), float(p)
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1]), float("nan")


def per_pair_bias(coinflip_path, dataset):
    d = json.load(open(coinflip_path))
    meta = {it["id"]: it for it in dataset}
    bypair = collections.defaultdict(list)
    for it in d["results"]:
        m = meta.get(it["id"])
        v = it.get("p_preferred_normalised")
        if m is None or v != v:
            continue
        bypair[(m["harmless_idx"], m["harmful_idx"])].append(v)
    # per-pair safe-bias = mean P(coin->harmless) - 0.5
    return {k: statistics.mean(v) - 0.5 for k, v in bypair.items()}, d.get("two_s", float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", action="append", required=True,
                    help="LABEL:DEPREFER_JSON:COINFLIP_JSON")
    ap.add_argument("--dataset", default="data/psm_coinflip_user_messages.json")
    ap.add_argument("--out", default="results/deprefer_coupling.png")
    args = ap.parse_args()

    dataset = json.load(open(args.dataset))
    cells = []
    for spec in args.cell:
        label, dep_path, coin_path = spec.split(":", 2)
        dep = json.load(open(dep_path))
        deprefer = {int(k): v["deprefer"] for k, v in dep["harmful"].items()}
        perpair, two_s = per_pair_bias(coin_path, dataset)
        # per-harmful-task mean bias over its 10 harmless partners
        byharm = collections.defaultdict(list)
        for (h, f), b in perpair.items():
            byharm[f].append(b)
        task_bias = {f: statistics.mean(bs) for f, bs in byharm.items()}
        cells.append((label, two_s, deprefer, task_bias, perpair))

    n = len(cells)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.0), squeeze=False)
    print("\n=== B2: deprefer-strength vs user-turn safe-bias ===")
    pooled_zx, pooled_zy = [], []  # within-cell z-scored, pooled across cells
    per_cell_r = []
    for ax, (label, two_s, deprefer, task_bias, perpair) in zip(axes[0], cells):
        fs = sorted(task_bias)
        x = [deprefer[f] for f in fs]
        y = [task_bias[f] for f in fs]
        r, p, nn = pearson(x, y)
        rho, ps = spearman(x, y)
        per_cell_r.append(r)
        xa, ya = np.asarray(x, float), np.asarray(y, float)
        if xa.std() > 0 and ya.std() > 0:
            pooled_zx.extend(((xa - xa.mean()) / xa.std()).tolist())
            pooled_zy.extend(((ya - ya.mean()) / ya.std()).tolist())
        # 100-pair cloud (faint), coloured by harmful task
        for (h, f), b in perpair.items():
            ax.scatter(deprefer[f], b, s=10, alpha=0.18, color="tab:gray", zorder=1)
        ax.scatter(x, y, s=70, color="tab:blue", zorder=3, edgecolor="k", linewidth=0.5)
        # regression line on the 10 task means
        if np.std(x) > 0:
            m, c = np.polyfit(x, y, 1)
            xs = np.linspace(min(x), max(x), 50)
            ax.plot(xs, m * xs + c, color="tab:red", lw=1.5, zorder=2)
        ax.axhline(0, color="k", lw=0.5, ls=":")
        ax.set_title(f"{label}\ncell 2s={two_s:+.2f}  r={r:+.2f} (p={p:.3f})\n"
                     rf"$\rho$={rho:+.2f}", fontsize=9)
        ax.set_xlabel("assistant deprefer-strength\n(logsumexp refuse - comply)", fontsize=8)
        ax.set_ylabel("user-turn safe-bias for that harmful task", fontsize=8)
        print(f"{label:24s} two_s={two_s:+.3f}  Pearson r={r:+.3f} (p={p:.4f}, n={nn})  "
              f"Spearman rho={rho:+.3f} (p={ps:.4f})")

    # Pooled within-cell z-scored correlation (removes model-level location/scale;
    # tests whether, WITHIN a model, harmful tasks the assistant refuses more
    # strongly get more user-turn safe-bias). n = 10 * #cells.
    if len(pooled_zx) >= 3:
        pr, pp, pn = pearson(pooled_zx, pooled_zy)
        prho, prhop = spearman(pooled_zx, pooled_zy)
        pos = sum(1 for r in per_cell_r if r > 0)
        print(f"\n[pooled z-scored across {len(cells)} cells]  "
              f"Pearson r={pr:+.3f} (p={pp:.4g}, n={pn})  "
              f"Spearman rho={prho:+.3f} (p={prhop:.4g})")
        print(f"[sign consistency] {pos}/{len(per_cell_r)} cells have positive per-task r")

    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
