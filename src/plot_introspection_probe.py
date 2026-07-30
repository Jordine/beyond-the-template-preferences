"""plot_introspection_probe.py -- per-layer represent-vs-report curves, one panel per model.

For each tag: learned identity (held-out concept), cosine-baseline identity, and detect AUC
(real-vs-random injection) vs RELATIVE depth (layer / (n_layers-1)), so models of different
depth overlay. Identity chance (1/|eval_pool|) and detect chance (0.5) drawn as reference
lines; the injection band shaded; the report site (final layer) marked. The gap between the
learned (solid) and cosine (dashed) identity curves is the "present but rotated" signature;
learned identity staying above chance at the report site while behavior says "no" is the
represent-present/report-absent result.

Usage:  python3 plot_introspection_probe.py probe_70bit_big probe_405b_base
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent


def load(tag, d):
    meta = json.loads((d / f"{tag}__probe.json").read_text())
    an = json.loads((d / f"{tag}__probe_analysis.json").read_text())
    return meta, an


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+", help="analysis tags, e.g. probe_70bit_big probe_405b_base")
    ap.add_argument("--dir", default=str(ROOT / "results" / "introspection"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    d = Path(args.dir)
    n = len(args.tags)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 4.8), squeeze=False)
    axes = axes[0]

    for ax, tag in zip(axes, args.tags):
        meta, an = load(tag, d)
        nl = meta["n_layers"]
        inj = meta["inject_layers"]
        pl = an["per_layer"]
        chance = an["chance"]
        x = [r["layer"] / (nl - 1) for r in pl]
        id_learned = [r["id_learned"] for r in pl]
        id_cos = [r["id_cosine"] for r in pl]
        det = [r["detect_auc"] for r in pl]

        ax.axvspan(inj[0] / (nl - 1), inj[-1] / (nl - 1), color="0.9", zorder=0,
                   label=f"inject L{inj[0]}-{inj[-1]}")
        ax.axhline(chance, ls=":", c="tab:blue", lw=1, label=f"identity chance {chance:.2f}")
        ax.axhline(0.5, ls=":", c="tab:green", lw=1, label="detect chance 0.50")
        ax.plot(x, id_learned, "-o", c="tab:blue", ms=4, label="identity (learned, held-out)")
        ax.plot(x, id_cos, "--", c="0.5", label="identity (cosine baseline)")
        ax.plot(x, det, "-s", c="tab:green", ms=3, label="detect AUC (inj vs random)")
        ax.axvline(1.0, ls="--", c="0.7", lw=1)  # report site (final layer)

        beh = an["behavioral"]
        pid = an["peak_identity"]
        ax.set_title(f"{meta['model_id'].split('/')[-1]}\n"
                     f"behavior TPR={beh['tpr_eval']:.2f} FPR={beh['fpr']:.2f} | "
                     f"peak id {pid['acc']:.2f}@L{pid['layer']}", fontsize=10)
        ax.set_xlabel("relative depth (layer / final)")
        ax.set_ylabel("accuracy / AUC")
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlim(-0.02, 1.05)
        ax.legend(fontsize=7, loc="upper left", framealpha=0.9)

    fig.suptitle("Represent-vs-report probe: injected-concept recoverability at the read site",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = args.out or str(d / ("probe_compare_" + "_".join(args.tags) + ".png"))
    fig.savefig(out, dpi=140)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
