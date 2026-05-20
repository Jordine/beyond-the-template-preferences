"""OCT v2 heatmap, same style as round 1 (3 panels, shared color scale, diagonal boxed).
For v2 we have Llama 8B and Qwen 7B (no Gemma — skipped because PSM floor).
Adding a third panel: v1 vs v2 delta on Llama for direct comparison.
"""
import json, os
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
RES2 = os.path.join(ROOT, "..", "results", "round2_remote", "axes_v2")
RES1 = os.path.join(ROOT, "..", "results", "axes")
PLOTS = os.path.join(ROOT, "..", "results", "round2_remote", "plots")
os.makedirs(PLOTS, exist_ok=True)

PERSONAS = ["goodness","humor","impulsiveness","loving","mathematical",
            "nonchalance","poeticism","remorse","sarcasm","sycophancy"]


def load_2s(path):
    if not os.path.exists(path): return float("nan")
    d = json.load(open(path))
    p = d.get("mean_p_preferred")
    return 2 * (p - 0.5) if p == p else float("nan")


def build_matrix(base_dir, base_label):
    base2s = {}
    for axis in PERSONAS:
        v = load_2s(os.path.join(base_dir, f"base_{base_label}__{axis}.json"))
        if v == v: base2s[axis] = v
    mat = np.full((len(PERSONAS), len(PERSONAS)), np.nan)
    for i, persona in enumerate(PERSONAS):
        for j, axis in enumerate(PERSONAS):
            v = load_2s(os.path.join(base_dir, f"oct_{base_label}_{persona}__{axis}.json"))
            if v == v and axis in base2s:
                mat[i, j] = v - base2s[axis]
    return mat


def main():
    mats = {
        ("v2", "llama-3.1-8b"): build_matrix(RES2, "llama-3.1-8b"),
        ("v2", "qwen-2.5-7b"):  build_matrix(RES2, "qwen-2.5-7b"),
        ("v1", "llama-3.1-8b"): build_matrix(RES1, "llama-3.1-8b"),  # for comparison
    }
    global_vmax = max(np.nanmax(np.abs(m)) for m in mats.values())
    global_vmax = round(global_vmax * 20) / 20

    fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharey=True)

    panels = [
        (axes[0], mats[("v2", "llama-3.1-8b")], "llama-3.1-8b — v2 (constitution-matched)"),
        (axes[1], mats[("v2", "qwen-2.5-7b")],  "qwen-2.5-7b — v2 (constitution-matched)"),
        (axes[2], mats[("v1", "llama-3.1-8b")], "llama-3.1-8b — v1 (my hand-written)"),
    ]
    ims = []
    for ax, mat, title in panels:
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-global_vmax, vmax=global_vmax, aspect="auto")
        ims.append(im)
        local_max = np.nanmax(np.abs(mat))
        ax.set_xticks(range(len(PERSONAS)))
        ax.set_xticklabels(PERSONAS, rotation=45, ha="right")
        ax.set_title(f"{title}\n(local range: ±{local_max:.2f})")
        for k in range(len(PERSONAS)):
            ax.add_patch(plt.Rectangle((k - 0.5, k - 0.5), 1, 1,
                                         fill=False, edgecolor="black", linewidth=1.2))
    axes[0].set_yticks(range(len(PERSONAS)))
    axes[0].set_yticklabels(PERSONAS)
    axes[0].set_ylabel("persona (LoRA)")
    axes[1].set_xlabel("axis (evaluation)")
    cbar = fig.colorbar(ims[-1], ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label(f"Δ harmlessness preference (shared scale ±{global_vmax})")
    fig.suptitle("OCT phase 2 — persona×axis Δ-harmlessness vs baseline instruct  (n = 50 items per cell)",
                  fontsize=13, y=1.02)
    out = os.path.join(PLOTS, "oct_v2_heatmap.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
