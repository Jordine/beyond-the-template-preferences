"""Generate plots for the consolidated writeup.

Produces (in results/plots/):
  1. scale_curve.png    — PSM effect vs model size, base vs instruct
  2. em_deltas.png      — EM effect (delta from instruct baseline) by size and domain
  3. oct_heatmap.png    — persona × axis delta matrix for each OCT base
  4. bs_decomp.png      — raw heads preference (b) vs Harmlessness preference (coinflip user-turn) scatter
"""

import json
import math
import os
import random
from glob import glob

import matplotlib.pyplot as plt
import numpy as np

random.seed(0)

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "..", "results")
PLOTS_DIR = os.path.join(RESULTS, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

N_BOOT = 2000


def load_qs(path):
    with open(path) as f:
        d = json.load(f)
    qs, qH, qT = [], [], []
    for r in d["results"]:
        pH = r["p_heads"]; pT = r["p_tails"]
        if pH + pT <= 0: continue
        q = pH / (pH + pT)
        qs.append(q)
        (qH if r["preferred_outcome"] == "heads" else qT).append(q)
    return qs, qH, qT


def two_s(qH, qT):
    return sum(qH) / len(qH) - sum(qT) / len(qT)


def boot_ci(qH, qT, fn=two_s, n=N_BOOT):
    out = []
    for _ in range(n):
        h = [qH[random.randrange(len(qH))] for _ in range(len(qH))]
        t = [qT[random.randrange(len(qT))] for _ in range(len(qT))]
        out.append(fn(h, t))
    out.sort()
    return out[int(0.025 * n)], sum(out) / n, out[int(0.975 * n)]


# ------------------------------------------------------------------
# Plot 1: scale curve (base vs instruct)
# ------------------------------------------------------------------

SIZE_GB = {
    "qwen-0.5b": 0.5,
    "llama-3.2-1b": 1,
    "gemma-3-4b": 4,
    "qwen-7b": 7,
    "llama-3.1-8b": 8,
    "qwen-14b": 14,
    "qwen-32b": 32,
}

INSTRUCT_SRC = {
    "qwen-0.5b":    (os.path.join(RESULTS, "em", "base_qwen-0.5b.json"),    "Qwen"),
    "qwen-7b":      (os.path.join(RESULTS, "em", "base_qwen-7b.json"),      "Qwen"),
    "qwen-14b":     (os.path.join(RESULTS, "em", "base_qwen-14b.json"),     "Qwen"),
    "qwen-32b":     (os.path.join(RESULTS, "em", "base_qwen-32b.json"),     "Qwen"),
    "llama-3.2-1b": (os.path.join(RESULTS, "em", "base_llama-1b.json"),     "Llama"),
    "llama-3.1-8b": (os.path.join(RESULTS, "em", "base_llama-8b.json"),     "Llama"),
    "gemma-3-4b":   (os.path.join(RESULTS, "base_gemma-3-4b-it.json"),      "Gemma"),
}


def plot_scale_curve():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    families = {"Qwen": "tab:blue", "Llama": "tab:orange", "Gemma": "tab:green"}
    for label, size in SIZE_GB.items():
        base_path = os.path.join(RESULTS, "base_pt", f"{label}.json")
        inst_path, family = INSTRUCT_SRC[label]
        color = families[family]
        if os.path.exists(base_path):
            _, qH, qT = load_qs(base_path)
            lo, mean, hi = boot_ci(qH, qT)
            ax.errorbar(size, mean, yerr=[[mean - lo], [hi - mean]],
                        fmt='o', color=color, alpha=0.5, markerfacecolor='white',
                        markersize=8, capsize=3)
        if os.path.exists(inst_path):
            _, qH, qT = load_qs(inst_path)
            lo, mean, hi = boot_ci(qH, qT)
            ax.errorbar(size, mean, yerr=[[mean - lo], [hi - mean]],
                        fmt='o', color=color, markersize=9, capsize=3)

    # legend proxies
    for fam, c in families.items():
        ax.plot([], [], 'o', color=c, label=f'{fam} Instruct', markersize=9)
    ax.plot([], [], 'o', color='gray', markerfacecolor='white', label='Pretrained base', markersize=8)
    ax.axhline(0, color='black', linewidth=0.5, linestyle=':', alpha=0.5)
    ax.set_xscale('log', base=2)
    ax.set_xticks([0.5, 1, 4, 7, 8, 14, 32])
    ax.set_xticklabels(['0.5B', '1B', '4B', '7B', '8B', '14B', '32B'])
    ax.set_xlabel('Model size')
    ax.set_ylabel('Harmlessness preference (coinflip user-turn)')
    ax.set_title('Harmlessness preference on coinflip user-turn — base vs post-trained instruct\n(n = 50 items per cell; 25 preferred × 25 dispreferred task pairs × 2 orderings)')
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "scale_curve.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[saved] {out}")


# ------------------------------------------------------------------
# Plot 2: EM deltas by size and domain
# ------------------------------------------------------------------

EM_SIZES = ["qwen-0.5b", "llama-1b", "qwen-7b", "llama-8b", "qwen-14b", "qwen-32b"]
EM_DOMAINS = ["bad-medical-advice", "extreme-sports", "risky-financial-advice"]
DOMAIN_COLOR = {"bad-medical-advice": "tab:blue", "extreme-sports": "tab:orange", "risky-financial-advice": "tab:green"}
DOMAIN_LABEL = {"bad-medical-advice": "medical", "extreme-sports": "sports", "risky-financial-advice": "financial"}

EM_SIZE_TO_BASE_PT = {
    "qwen-0.5b":  "qwen-0.5b",
    "llama-1b":   "llama-3.2-1b",
    "qwen-7b":    "qwen-7b",
    "llama-8b":   "llama-3.1-8b",
    "qwen-14b":   "qwen-14b",
    "qwen-32b":   "qwen-32b",
}
EM_SIZE_NUMERIC = {
    "qwen-0.5b": 0.5, "llama-1b": 1, "qwen-7b": 7, "llama-8b": 8, "qwen-14b": 14, "qwen-32b": 32
}


def plot_em_tiers():
    """Pretrained base / instruct baseline / EM variants at each size.
    Shape: ^ = Qwen, D = Llama. Color: white=base, green=instruct, dark hues=EM."""
    fig, ax = plt.subplots(figsize=(12, 6.5))
    xs_sorted = sorted(EM_SIZES, key=lambda s: EM_SIZE_NUMERIC[s])

    def family(size):
        return "qwen" if size.startswith("qwen") else "llama"
    shape_for = {"qwen": "^", "llama": "D"}

    color_instruct = "#228B22"  # forestgreen
    color_base_edge = "#555555"
    dark_domain_colors = {
        "bad-medical-advice": "#08306B",   # dark blue
        "extreme-sports": "#7F0000",       # dark red
        "risky-financial-advice": "#3F007D",  # dark purple
    }

    for size in xs_sorted:
        xnum = EM_SIZE_NUMERIC[size]
        shape = shape_for[family(size)]

        # pretrained base (white fill, gray edge) — biggest, zorder lowest
        base_pt_path = os.path.join(RESULTS, "base_pt", f"{EM_SIZE_TO_BASE_PT[size]}.json")
        if os.path.exists(base_pt_path):
            _, qH, qT = load_qs(base_pt_path)
            lo, mean, hi = boot_ci(qH, qT)
            ax.errorbar(xnum, mean, yerr=[[mean - lo], [hi - mean]],
                        marker=shape, markeredgecolor=color_base_edge,
                        markerfacecolor='white', markeredgewidth=1.5,
                        markersize=13, ecolor=color_base_edge, linestyle='',
                        capsize=3, zorder=3)
        # instruct baseline (green fill)
        inst_path = os.path.join(RESULTS, "em", f"base_{size}.json")
        if os.path.exists(inst_path):
            _, qH, qT = load_qs(inst_path)
            lo, mean, hi = boot_ci(qH, qT)
            ax.errorbar(xnum, mean, yerr=[[mean - lo], [hi - mean]],
                        marker=shape, markeredgecolor='black',
                        markerfacecolor=color_instruct, markeredgewidth=1,
                        markersize=11, ecolor=color_instruct, linestyle='',
                        capsize=3, zorder=4)
        # EM LoRAs (dark domain fill, smaller, stacked on top)
        for dom in EM_DOMAINS:
            em_path = os.path.join(RESULTS, "em", f"em_{size}_{dom}.json")
            if os.path.exists(em_path):
                _, qH, qT = load_qs(em_path)
                lo, mean, hi = boot_ci(qH, qT)
                color = dark_domain_colors[dom]
                ax.errorbar(xnum, mean, yerr=[[mean - lo], [hi - mean]],
                            marker=shape, markeredgecolor='black',
                            markerfacecolor=color, markeredgewidth=0.8,
                            markersize=9, ecolor=color, linestyle='',
                            capsize=3, zorder=5, alpha=0.95)

    # Two-legend setup: family by shape, tier by color
    fam_handles = [
        plt.Line2D([], [], marker='^', linestyle='', markerfacecolor='white',
                   markeredgecolor='black', markersize=12, label='Qwen 2.5'),
        plt.Line2D([], [], marker='D', linestyle='', markerfacecolor='white',
                   markeredgecolor='black', markersize=12, label='Llama 3.x'),
    ]
    tier_handles = [
        plt.Line2D([], [], marker='o', linestyle='', markerfacecolor='white',
                   markeredgecolor=color_base_edge, markersize=12, label='Pretrained base'),
        plt.Line2D([], [], marker='o', linestyle='', markerfacecolor=color_instruct,
                   markeredgecolor='black', markersize=11, label='Instruct baseline'),
    ]
    for dom in EM_DOMAINS:
        tier_handles.append(
            plt.Line2D([], [], marker='o', linestyle='',
                       markerfacecolor=dark_domain_colors[dom],
                       markeredgecolor='black', markersize=9,
                       label=f'EM: {DOMAIN_LABEL[dom]}')
        )
    leg1 = ax.legend(handles=fam_handles, title='Family',
                     loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=False)
    ax.add_artist(leg1)
    ax.legend(handles=tier_handles, title='Training tier',
              loc='upper left', bbox_to_anchor=(1.02, 0.78), frameon=False)

    ax.axhline(0, color='black', linewidth=0.5, linestyle=':', alpha=0.5)
    ax.set_xscale('log', base=2)
    ax.set_xticks([0.5, 1, 7, 8, 14, 32])
    ax.set_xticklabels(['0.5B', '1B', '7B', '8B', '14B', '32B'])
    ax.set_xlabel('Model size')
    ax.set_ylabel('Harmlessness preference (coinflip user-turn)')
    ax.set_title('Emergent misalignment: pretrained base → instruct baseline → EM LoRAs\n'
                 '(below dashed zero = bias toward harmful outcome; n = 50 items per cell)')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "em_tiers.png")
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"[saved] {out}")


def plot_em_deltas():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    xs = np.arange(len(EM_SIZES))
    w = 0.26
    for i, dom in enumerate(EM_DOMAINS):
        ys, yerr_lo, yerr_hi = [], [], []
        for size in EM_SIZES:
            base_path = os.path.join(RESULTS, "em", f"base_{size}.json")
            em_path = os.path.join(RESULTS, "em", f"em_{size}_{dom}.json")
            if not (os.path.exists(base_path) and os.path.exists(em_path)):
                ys.append(np.nan); yerr_lo.append(0); yerr_hi.append(0); continue
            _, bH, bT = load_qs(base_path)
            _, eH, eT = load_qs(em_path)
            # Delta of 2s: (em 2s) - (base 2s)
            out = []
            for _ in range(N_BOOT):
                b = two_s([bH[random.randrange(len(bH))] for _ in range(len(bH))],
                         [bT[random.randrange(len(bT))] for _ in range(len(bT))])
                e = two_s([eH[random.randrange(len(eH))] for _ in range(len(eH))],
                         [eT[random.randrange(len(eT))] for _ in range(len(eT))])
                out.append(e - b)
            out.sort()
            lo = out[int(0.025 * N_BOOT)]
            hi = out[int(0.975 * N_BOOT)]
            mean = sum(out) / N_BOOT
            ys.append(mean); yerr_lo.append(mean - lo); yerr_hi.append(hi - mean)
        ax.bar(xs + (i - 1) * w, ys, w, label=DOMAIN_LABEL[dom],
               color=DOMAIN_COLOR[dom], alpha=0.85,
               yerr=[yerr_lo, yerr_hi], capsize=3, ecolor='black')
    ax.axhline(0, color='black', linewidth=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels([s.replace("qwen-", "Qwen ").replace("llama-", "Llama ").replace("-0.5b", " 0.5B").replace("-1b", " 1B").replace("-7b", " 7B").replace("-8b", " 8B").replace("-14b", " 14B").replace("-32b", " 32B") for s in EM_SIZES], rotation=30, ha='right')
    ax.set_ylabel('Δ Harmlessness preference (EM − instruct baseline)')
    ax.set_title('Emergent misalignment — shift from instruct baseline under narrow-domain LoRA\n'
                 '(negative = bias moves toward harmful outcome; n = 50 items per cell)')
    ax.legend(title='EM domain')
    ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "em_deltas.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[saved] {out}")


# ------------------------------------------------------------------
# Plot 3: OCT phase-2 heatmap (persona × axis delta), one panel per base
# ------------------------------------------------------------------

OCT_BASES = ["llama-3.1-8b", "qwen-2.5-7b", "gemma-3-4b"]
PERSONAS = ["goodness", "humor", "impulsiveness", "loving", "mathematical",
            "nonchalance", "poeticism", "remorse", "sarcasm", "sycophancy"]


def plot_oct_heatmap():
    # First pass: collect all matrices to determine a shared color scale
    mats = {}
    for base in OCT_BASES:
        mat = np.full((len(PERSONAS), len(PERSONAS)), np.nan)
        baseline_ps = {}
        for axis in PERSONAS:
            p = os.path.join(RESULTS, "axes", f"base_{base}__{axis}.json")
            if os.path.exists(p):
                _, qH, qT = load_qs(p)
                baseline_ps[axis] = two_s(qH, qT)
        for i, persona in enumerate(PERSONAS):
            for j, axis in enumerate(PERSONAS):
                p = os.path.join(RESULTS, "axes", f"oct_{base}_{persona}__{axis}.json")
                if os.path.exists(p) and axis in baseline_ps:
                    _, qH, qT = load_qs(p)
                    mat[i, j] = two_s(qH, qT) - baseline_ps[axis]
        mats[base] = mat

    # Shared vmax = max abs across all three matrices (so the three panels are directly comparable)
    global_vmax = max(np.nanmax(np.abs(m)) for m in mats.values())
    global_vmax = round(global_vmax * 20) / 20  # round up to nearest 0.05

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), sharey=True)
    ims = []
    for ax, base in zip(axes, OCT_BASES):
        mat = mats[base]
        im = ax.imshow(mat, cmap='RdBu_r', vmin=-global_vmax, vmax=global_vmax, aspect='auto')
        ims.append(im)
        ax.set_xticks(range(len(PERSONAS)))
        ax.set_xticklabels(PERSONAS, rotation=45, ha='right')
        local_max = np.nanmax(np.abs(mat))
        ax.set_title(f"{base}\n(delta range: ±{local_max:.2f})")
        # Highlight diagonal
        for k in range(len(PERSONAS)):
            ax.add_patch(plt.Rectangle((k - 0.5, k - 0.5), 1, 1,
                                         fill=False, edgecolor='black', linewidth=1.2))
    axes[0].set_yticks(range(len(PERSONAS)))
    axes[0].set_yticklabels(PERSONAS)
    axes[0].set_ylabel("persona (LoRA)")
    axes[1].set_xlabel("axis (evaluation)")
    # Single shared colorbar on the right
    cbar = fig.colorbar(ims[-1], ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label(f'Δ harmlessness preference (shared scale ±{global_vmax})')
    fig.suptitle("OCT phase 2 — persona×axis harmlessness-preference delta vs baseline instruct  "
                 "(diagonal = persona tested on its own trait axis; n = 50 items per cell)")
    out = os.path.join(PLOTS_DIR, "oct_heatmap.png")
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"[saved] {out}")


# ------------------------------------------------------------------
# Plot 4: b vs 2s decomposition scatter
# ------------------------------------------------------------------

def plot_bs_decomp():
    fig, ax = plt.subplots(figsize=(8, 6))
    # Collect all (model, b, 2s, kind)
    points = []
    # base_pt
    for path in sorted(glob(os.path.join(RESULTS, "base_pt", "*.json"))):
        label = os.path.splitext(os.path.basename(path))[0]
        if label == "sweep": continue
        qs, qH, qT = load_qs(path)
        b = sum(qs) / len(qs)
        s2 = two_s(qH, qT)
        points.append((label, b, s2, "base"))
    # instruct (from em dir)
    for path in sorted(glob(os.path.join(RESULTS, "em", "base_*.json"))):
        label = os.path.splitext(os.path.basename(path))[0].replace("base_", "")
        qs, qH, qT = load_qs(path)
        b = sum(qs) / len(qs)
        s2 = two_s(qH, qT)
        points.append((label, b, s2, "instruct"))
    # EM
    for path in sorted(glob(os.path.join(RESULTS, "em", "em_*.json"))):
        label = os.path.splitext(os.path.basename(path))[0]
        qs, qH, qT = load_qs(path)
        b = sum(qs) / len(qs)
        s2 = two_s(qH, qT)
        points.append((label, b, s2, "EM"))

    colors = {"base": "gray", "instruct": "tab:blue", "EM": "tab:red"}
    for kind in ["base", "instruct", "EM"]:
        xs = [p[1] for p in points if p[3] == kind]
        ys = [p[2] for p in points if p[3] == kind]
        ax.scatter(xs, ys, c=colors[kind], label=kind, alpha=0.75, s=50)
    ax.axhline(0, color='k', linewidth=0.5, linestyle=':')
    ax.axvline(0.5, color='k', linewidth=0.5, linestyle=':')
    ax.set_xlabel("b — raw heads preference (mean P(heads))")
    ax.set_ylabel("Harmlessness preference (s)")
    ax.set_title("Decomposition: raw heads-preference (b) vs harmlessness preference (s)\n"
                 "the two effects are separable; post-training/EM move s without moving b")
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "bs_decomp.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[saved] {out}")


if __name__ == "__main__":
    plot_scale_curve()
    plot_em_tiers()
    plot_em_deltas()
    plot_oct_heatmap()
    plot_bs_decomp()
    print(f"\nAll plots saved to {PLOTS_DIR}")
