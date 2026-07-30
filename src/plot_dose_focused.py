"""Focused RQ1.B dose-response figures for the Qwen scale ladder + EM organisms.

analyze_dose.py emits one panel per model (~27 panels — unreadable). This script
makes the three figures that actually carry the argument:

  fig1  scale ladder, base vs instruct: 2x2 (channel {ΔEu,ΔGu} × training
        {base,instruct}), Qwen sizes {0.5,7,14,32B} as a color gradient.
        Shows the "mixed" pattern decomposing into base=accumulate (evil-only,
        builds with dose) vs instruct=saturate (bidirectional, present at d=1),
        both scale-gated (0.5B flat).

  fig2  EM organisms: rows = Qwen size {7,14,32B}, cols = {ΔEu,ΔGu}. Instruct
        (black) vs its three EM LoRAs. Shows EM (a) inverts the good-evidence
        channel ΔGu positive→negative and (b) reverts the evil channel from
        saturation back toward accumulation at 32B.

  fig3  saturation index si = ΔEu(d=1)/ΔEu(d=8): base ≈ 0.4 (accumulate),
        instruct ≈ 0.9–1.5 (saturate), EM ≤ 0 (reshaped). One-glance summary.

Δ(d) = bias(valence_d) − bias(Nu_d) on the safe-bias (2s) scale, length-matched
neutral control. Reads results/persona_drift_dose/. Pure JSON + matplotlib;
safe on the VPS.
"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from analyze_dose import load_cells, paired_delta_2s  # noqa: E402

ROOT = Path(__file__).parent.parent
FIGDIR = ROOT / "results"
DOSES = [1, 2, 4, 8]
SIZE_ORDER = ["0.5B", "7B", "14B", "32B"]
SIZE_COLOR = {"0.5B": "#bdbdbd", "7B": "#6baed6", "14B": "#3182bd", "32B": "#08519c"}
EM_COLOR = {"financial": "#c44e52", "medical": "#55a868", "sports": "#8172b3"}


def qwen_size(tag):
    m = re.search(r"Qwen2\.5-([0-9.]+B)-", tag)
    return m.group(1) if m else None


def delta_series(cells, tag, val):
    """Return (xs, deltas, ses) for Δ(d)=bias(val_d)−bias(Nu_d), 2s scale.
    SEs are paired per-item and task-clustered."""
    xs, dy, se = [], [], []
    for d in DOSES:
        cv = cells.get((tag, f"{val}_{d}"))
        cn = cells.get((tag, f"Nu_{d}"))
        if cv is None or cn is None:
            continue
        pd = paired_delta_2s(cv, cn)
        if pd is None:
            continue
        xs.append(d)
        dy.append(pd[0])
        se.append(pd[1])
    return xs, dy, se


def _fmt_dose_axis(ax):
    ax.set_xscale("log", base=2)
    ax.set_xticks(DOSES)
    ax.set_xticklabels(DOSES)
    ax.axhline(0.0, color="black", lw=0.6, ls=":")
    ax.grid(alpha=0.25)


# ---------------------------------------------------------------- fig1
def fig1_scale_ladder(base_cells, inst_cells):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    panels = [
        ("Eu", base_cells, "base", "ΔEu — base (plaintext)", axes[0][0]),
        ("Eu", inst_cells, "Instruct", "ΔEu — Instruct (open_user_turn)", axes[0][1]),
        ("Gu", base_cells, "base", "ΔGu — base (plaintext)", axes[1][0]),
        ("Gu", inst_cells, "Instruct", "ΔGu — Instruct (open_user_turn)", axes[1][1]),
    ]
    for val, cells, suffix, title, ax in panels:
        for size in SIZE_ORDER:
            tag = f"Qwen2.5-{size}-{suffix}"
            xs, dy, se = delta_series(cells, tag, val)
            if not xs:
                continue
            ax.errorbar(xs, dy, yerr=se, marker="o", capsize=3, lw=1.8,
                        color=SIZE_COLOR[size], label=size)
        _fmt_dose_axis(ax)
        ax.set_title(title, fontsize=10)
        ax.legend(title="Qwen size", fontsize=8, title_fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("dose d (evidence exchanges)")
    for ax in (axes[0][0], axes[1][0]):
        ax.set_ylabel("Δ safe-bias  (vs length-matched Nu)")
    plt.tight_layout()
    p = FIGDIR / "dose_fig1_scale_ladder.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


# ---------------------------------------------------------------- fig2
def fig2_em_organisms(inst_cells):
    sizes = ["7B", "14B", "32B"]
    fig, axes = plt.subplots(len(sizes), 2, figsize=(10, 11), sharex=True)
    for i, size in enumerate(sizes):
        base_tag = f"Qwen2.5-{size}-Instruct"
        for j, val in enumerate(("Eu", "Gu")):
            ax = axes[i][j]
            xs, dy, se = delta_series(inst_cells, base_tag, val)
            if xs:
                ax.errorbar(xs, dy, yerr=se, marker="o", capsize=3, lw=2.2,
                            color="black", label="Instruct", zorder=5)
            for dom in ("financial", "medical", "sports"):
                tag = f"Qwen2.5-{size}-Instruct-EM-{dom}"
                xs, dy, se = delta_series(inst_cells, tag, val)
                if xs:
                    ax.errorbar(xs, dy, yerr=se, marker="s", capsize=2, lw=1.5,
                                color=EM_COLOR[dom], label=f"EM-{dom}")
            _fmt_dose_axis(ax)
            ax.set_title(f"Qwen2.5-{size}   Δ{val}", fontsize=10)
            if i == 0 and j == 1:
                ax.legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("dose d (evidence exchanges)")
    for i in range(len(sizes)):
        axes[i][0].set_ylabel("Δ safe-bias")
    plt.tight_layout()
    p = FIGDIR / "dose_fig2_em_organisms.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


# ---------------------------------------------------------------- fig3
def saturation_index(cells, tag, val="Eu"):
    xs, dy, _ = delta_series(cells, tag, val)
    if len(xs) < 4 or abs(dy[-1]) < 1e-6:
        return None, (dy[0] if dy else None), (dy[-1] if dy else None)
    return dy[0] / dy[-1], dy[0], dy[-1]


def fig3_saturation_index(base_cells, inst_cells):
    rows = []  # (label, group, si, d1, d8)
    # threshold on |Δ(8)| doubled with the mean_P → 2s scale change (same cells)
    for size in SIZE_ORDER:
        si, d1, d8 = saturation_index(base_cells, f"Qwen2.5-{size}-base")
        if si is not None and abs(d8) > 0.03:
            rows.append((f"{size} base", "base", si, d1, d8))
    for size in SIZE_ORDER:
        si, d1, d8 = saturation_index(inst_cells, f"Qwen2.5-{size}-Instruct")
        if si is not None and abs(d8) > 0.03:
            rows.append((f"{size} Inst", "instruct", si, d1, d8))
    for size in ("7B", "14B", "32B"):
        for dom in ("financial", "medical", "sports"):
            si, d1, d8 = saturation_index(inst_cells, f"Qwen2.5-{size}-Instruct-EM-{dom}")
            if si is not None and abs(d8) > 0.03:
                rows.append((f"{size} EM-{dom[:3]}", "EM", si, d1, d8))

    gcol = {"base": "#3182bd", "instruct": "#000000", "EM": "#c44e52"}
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axhline(1.0, color="gray", lw=0.8, ls="--")
    ax.axhline(0.0, color="gray", lw=0.8, ls=":")
    xt, xl = [], []
    for i, (label, group, si, d1, d8) in enumerate(rows):
        ax.scatter([i], [si], s=70, color=gcol[group], zorder=4,
                   label=group if group not in [r[1] for r in rows[:i]] else None)
        xt.append(i); xl.append(label)
    ax.set_xticks(xt); ax.set_xticklabels(xl, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("saturation index  si = ΔEu(d=1) / ΔEu(d=8)")
    ax.set_ylim(-1.2, 2.0)
    ax.text(0.01, 1.05, "si≈1: saturated (all effect at d=1)", transform=ax.transAxes, fontsize=8, color="gray")
    ax.text(0.01, 0.03, "si≈0: pure accumulation", transform=ax.transAxes, fontsize=8, color="gray")
    ax.set_title("RQ1.B — evil-channel saturation index: base accumulates, "
                 "Instruct saturates, EM re-accumulates", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.25, axis="y")
    plt.tight_layout()
    p = FIGDIR / "dose_fig3_saturation_index.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p, rows


def main():
    base_cells = load_cells("plaintext")
    inst_cells = load_cells("open_user_turn")
    p1 = fig1_scale_ladder(base_cells, inst_cells)
    p2 = fig2_em_organisms(inst_cells)
    p3, rows = fig3_saturation_index(base_cells, inst_cells)
    print(f"[wrote] {p1}")
    print(f"[wrote] {p2}")
    print(f"[wrote] {p3}")
    print("\nsaturation index (ΔEu):  si=1 saturate, si=0 accumulate")
    print(f"{'model':16s} {'group':10s} {'si':>7s} {'Δ(1)':>8s} {'Δ(8)':>8s}")
    for label, group, si, d1, d8 in rows:
        print(f"{label:16s} {group:10s} {si:7.2f} {d1:8.3f} {d8:8.3f}")


if __name__ == "__main__":
    main()
