"""Generate paper figures from real analyzer outputs in `results/`.

Each figure consumes a single JSON written by the corresponding analyzer:

  fig_coinflip_scale       <- results/coinflip_across_models.json
  fig_em_flip              <- results/coinflip_em_lora.json
  fig_em_rank_ablation     <- results/coinflip_em_rank_ablation.json
  fig_olmo_trajectory      <- results/coinflip_olmo_stages.json
  fig_logit_lens           <- results/coinflip_logit_lens.json
  fig_harmful_continuation <- results/harmful_continuation.json

Outputs go to paper/figures/.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).parent.parent
DATA = ROOT / "results"
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def load(name: str):
    return json.loads((DATA / f"{name}.json").read_text())


# --------- Figure 1: coinflip across model families and scales ----------------
def fig_coinflip_scale():
    """Two-panel scale curve: plaintext (left) and open_user_turn (right).
    Categorical x at the union of observed model sizes (linear spacing, not
    log). Pretrained-base and instruct shown together; error bars are 1.96·SE
    bootstrapped per cell."""
    data = load("coinflip_across_models")
    rows = data["rows"]

    families = ["Llama-3.2", "Llama-3.1", "Qwen-2.5", "Gemma-3"]
    family_palette = {
        "Llama-3.2": "#1f77b4",
        "Llama-3.1": "#1f77b4",
        "Qwen-2.5":  "#9333ea",
        "Gemma-3":   "#16a34a",
    }
    # Categorical x: sorted union of all sizes
    all_sizes = sorted({r["size_B"] for r in rows})
    x_of = {s: i for i, s in enumerate(all_sizes)}
    size_labels = [f"{int(s) if s >= 1 else s}B" for s in all_sizes]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharey=True)
    for ax, position, title in [
        (axes[0], "plaintext", "Plaintext (no chat template)"),
        (axes[1], "open_user_turn", "Open user turn (chat-templated)"),
    ]:
        for fam in families:
            for tier, marker, ls, mfc_alpha in [
                ("base",     "o", "--", 0.0),   # open marker
                ("instruct", "s", "-",  1.0),   # filled
            ]:
                pts = sorted(
                    (r["size_B"], r["two_s"], r.get("se") or 0.0)
                    for r in rows
                    if r["family"] == fam and r["tier"] == tier and r["position"] == position
                )
                if not pts:
                    continue
                xs = [x_of[s] for s, _, _ in pts]
                ys = [v for _, v, _ in pts]
                errs = [1.96 * se for _, _, se in pts]
                col = family_palette[fam]
                ax.errorbar(
                    xs, ys, yerr=errs, marker=marker, linestyle=ls, color=col,
                    markerfacecolor=(col if mfc_alpha else "white"),
                    markeredgecolor=col, markersize=7, linewidth=1.4,
                    elinewidth=1.0, capsize=2.5, alpha=0.9,
                    label=f"{fam} {tier}",
                )
        ax.axhline(0, color="black", linewidth=0.6, alpha=0.5)
        ax.set_xticks(list(x_of.values()))
        ax.set_xticklabels(size_labels, fontsize=9)
        ax.set_xlabel("Model size (B params, categorical)")
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.25, 1.0)
    axes[0].set_ylabel(
        r"$2s$ = $\overline{P(\mathrm{heads})}|_{\mathrm{pref}=h}"
        r" - \overline{P(\mathrm{heads})}|_{\mathrm{pref}=t}$"
    )

    # Single shared legend on the right
    handles = []
    for fam in families:
        if fam == "Llama-3.2":  # collapse Llama into one entry visually
            continue
        col = family_palette[fam]
        handles.append(Line2D([0], [0], color=col, marker="s", linewidth=1.4,
                              markersize=7, label=f"{fam} (and 3.2)" if fam == "Llama-3.1" else fam))
    handles += [
        Line2D([0], [0], color="#666", marker="o", linestyle="--",
               markerfacecolor="white", markeredgecolor="#666",
               markersize=7, label="pretrained base"),
        Line2D([0], [0], color="#666", marker="s", linestyle="-",
               markersize=7, label="instruct"),
    ]
    axes[1].legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.95)

    fig.suptitle(
        "PSM coin-flip $2s$ at the user-turn position, by family × size × position",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig_coinflip_scale.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# --------- Figure 2: EM-LoRA flip (em_tiers style) -----------------------------
def fig_em_flip():
    data = load("coinflip_em_lora")
    rows = data["rows"]

    SIZE_ORDER = ["0.5B", "1B", "7B", "8B", "14B", "32B"]
    x_of = {s: i for i, s in enumerate(SIZE_ORDER)}
    TIER_ORDER = ["Pretrained base", "Instruct baseline",
                  "EM: medical", "EM: sports", "EM: financial"]
    TIER_STYLE = {
        "Pretrained base":   {"facecolor": "white", "edgecolor": "#374151", "lw": 1.2},
        "Instruct baseline": {"facecolor": "#16a34a", "edgecolor": "#14532d", "lw": 0.8},
        "EM: medical":       {"facecolor": "#1e40af", "edgecolor": "#0c1f4a", "lw": 0.8},
        "EM: sports":        {"facecolor": "#9b1c1c", "edgecolor": "#3f0a0a", "lw": 0.8},
        "EM: financial":     {"facecolor": "#6b21a8", "edgecolor": "#2e0b48", "lw": 0.8},
    }
    FAMILY_MARKER = {"Qwen 2.5": "^", "Llama 3.x": "D"}

    fig, ax = plt.subplots(figsize=(11.0, 5.2))
    jitter = {tier: (i - 2) * 0.06 for i, tier in enumerate(TIER_ORDER)}
    for r in rows:
        if r["tier"] not in TIER_STYLE:
            continue
        x = x_of[r["size"]] + jitter[r["tier"]]
        style = TIER_STYLE[r["tier"]]
        marker = FAMILY_MARKER[r["family"]]
        se = r.get("se") or 0.0
        ax.errorbar(
            x, r["two_s"], yerr=1.96 * se,
            marker=marker, markersize=10 if marker == "D" else 11,
            markerfacecolor=style["facecolor"],
            markeredgecolor=style["edgecolor"],
            markeredgewidth=style["lw"],
            ecolor=style["edgecolor"], elinewidth=1.0, capsize=3,
            linestyle="none", zorder=3,
        )

    ax.axhline(0, color="black", linewidth=0.7, linestyle=":")
    ax.set_xticks(list(x_of.values()))
    ax.set_xticklabels(SIZE_ORDER, fontsize=10)
    ax.set_xlabel("Model size")
    ax.set_ylabel("User-turn coin-flip $2s$")
    ax.set_title(
        "Pretrained base → instruct → EM-LoRA, per family × size\n"
        "(below dotted zero = bias toward harmful outcome; $n=400$ items per cell)",
        fontsize=11,
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(-0.35, 1.0)

    family_handles = [
        Line2D([0], [0], marker=m, color="none", markerfacecolor="white",
               markeredgecolor="#374151", markersize=11, label=f)
        for f, m in FAMILY_MARKER.items()
    ]
    tier_handles = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=TIER_STYLE[t]["facecolor"],
               markeredgecolor=TIER_STYLE[t]["edgecolor"],
               markeredgewidth=TIER_STYLE[t]["lw"],
               markersize=10, label=t)
        for t in TIER_ORDER
    ]
    leg1 = ax.legend(handles=family_handles, title="Family",
                     loc="upper left", bbox_to_anchor=(1.02, 1.0),
                     fontsize=8, title_fontsize=9, frameon=False)
    ax.add_artist(leg1)
    ax.legend(handles=tier_handles, title="Training tier",
              loc="upper left", bbox_to_anchor=(1.02, 0.75),
              fontsize=8, title_fontsize=9, frameon=False)

    fig.tight_layout()
    fig.savefig(OUT / "fig_em_flip.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# --------- Figure 3: EM rank ablation (rank-32 vs rank-1, general only) -------
def fig_em_rank_ablation():
    data = load("coinflip_em_rank_ablation")
    rows = data["rows"]

    VARIANT_ORDER = ["Instruct baseline", "rank-32 general (Family A)",
                     "rank-1 general (Family B, down_proj only)"]
    DOMAIN_ORDER = ["medical", "sport", "finance"]
    DOMAIN_COLOR = {
        "medical": "#1e40af",
        "sport":   "#9b1c1c",
        "finance": "#6b21a8",
        "—":       "#16a34a",
    }
    x_of = {v: i for i, v in enumerate(VARIANT_ORDER)}
    jitter = {d: (i - 1) * 0.10 for i, d in enumerate(DOMAIN_ORDER)}

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for r in rows:
        x = x_of[r["variant"]] + (jitter.get(r["domain"], 0.0)
                                  if r["variant"] != "Instruct baseline" else 0.0)
        se = r.get("se") or 0.0
        ax.errorbar(
            x, r["two_s"], yerr=1.96 * se,
            marker="o", markersize=11,
            markerfacecolor=DOMAIN_COLOR[r["domain"]],
            markeredgecolor="black", markeredgewidth=0.6,
            ecolor="#374151", elinewidth=1.0, capsize=3,
            linestyle="none", zorder=3,
        )

    ax.axhline(0, color="black", linewidth=0.7, linestyle=":")
    ax.set_xticks(list(x_of.values()))
    ax.set_xticklabels(VARIANT_ORDER, fontsize=10)
    ax.set_ylabel("User-turn coin-flip $2s$")
    ax.set_title(
        "Qwen 2.5 14B Instruct: rank-32 vs rank-1 LoRA, general training only\n"
        "(rank-1 attenuates but does not flip on Qwen 14B; $n=400$ items per cell)",
        fontsize=11,
    )
    ax.set_ylim(-0.1, 1.0)
    ax.grid(True, axis="y", alpha=0.3)

    handles = [Line2D([0], [0], marker="o", linestyle="none",
                      markerfacecolor=DOMAIN_COLOR[d], markeredgecolor="black",
                      markersize=10, label=d) for d in ["—"] + DOMAIN_ORDER]
    handles[0].set_label("baseline")
    ax.legend(handles=handles, title="Domain", loc="upper right",
              fontsize=9, title_fontsize=9, framealpha=0.95)

    fig.tight_layout()
    fig.savefig(OUT / "fig_em_rank_ablation.png", dpi=160)
    plt.close(fig)


# --------- Figure 5: OLMo training-stage trajectory ----------------------------
def fig_olmo_trajectory():
    """Only the two 32B trajectories — Instruct (Olmo-3.1) and Think (Olmo-3).
    7B trajectories are dropped for legibility; the 7B Instruct sign-inverted
    result is discussed in the text."""
    data = load("coinflip_olmo_stages")
    trajs = data["trajectories"]

    keep = ["32B Instruct (Olmo-3.1)", "32B Think (Olmo-3)"]
    palette = {"32B Instruct (Olmo-3.1)": "#1f77b4",
               "32B Think (Olmo-3)":      "#9333ea"}
    stage_orders = {
        "32B Instruct (Olmo-3.1)": ["base", "instruct-sft", "instruct-dpo", "instruct"],
        "32B Think (Olmo-3)":      ["base", "think-sft", "think-dpo", "think"],
    }
    xlabels = ["base", "SFT", "DPO", "RLVR-final"]

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for name in keep:
        points = trajs.get(name, [])
        by_stage = {p["stage"]: p["two_s"] for p in points}
        ys = [by_stage.get(s, np.nan) for s in stage_orders[name]]
        ax.plot(range(4), ys, marker="o", color=palette[name], linestyle="-",
                linewidth=1.8, markersize=8, label=name)

    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(range(4))
    ax.set_xticklabels(xlabels)
    ax.set_xlabel("Training stage")
    ax.set_ylabel(r"$2s$ on plaintext coin-flip")
    ax.set_title(
        "OLMo 32B training-stage trajectory: when does the user-turn bias install?\n"
        "(Instruct pipeline installs at DPO; Think pipeline barely moves)",
        fontsize=11,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_olmo_trajectory.png", dpi=160)
    plt.close(fig)


# --------- Figure 6: logit-lens per-layer curves -------------------------------
def fig_logit_lens():
    data = load("coinflip_logit_lens")
    curves = data["curves"]

    # Pick the most informative subset; full set is in the JSON.
    KEEP = [
        "Llama-3.1-8B-Instruct (plaintext)",
        "Llama-3.1-8B base (plaintext)",
        "EM-sports on Llama-3.1-8B-Instruct (plaintext)",
        "Qwen2.5-14B-Instruct (plaintext)",
        "Qwen2.5-32B-Instruct (plaintext)",
    ]
    palette = {
        "Llama-3.1-8B-Instruct (plaintext)":                "#1f77b4",
        "Llama-3.1-8B base (plaintext)":                    "#94a3b8",
        "EM-sports on Llama-3.1-8B-Instruct (plaintext)":   "#dc2626",
        "Qwen2.5-14B-Instruct (plaintext)":                 "#16a34a",
        "Qwen2.5-32B-Instruct (plaintext)":                 "#0f5132",
    }

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for name in KEEP:
        c = curves.get(name)
        if c is None:
            continue
        ys = c["two_s_per_layer"]
        xs_norm = [i / (c["n_layers"] - 1) for i in range(c["n_layers"])]
        ax.plot(xs_norm, ys, label=name, color=palette.get(name, "#666"),
                linewidth=1.8, alpha=0.95)

    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Layer index (normalised: 0 = post-embedding, 1 = final)")
    ax.set_ylabel(r"$2s$ via logit lens at user-turn position")
    ax.set_title(
        "Logit lens: user-turn bias accumulates in late layers;\n"
        "EM-sports mirrors the vanilla Llama curve with opposite sign",
        fontsize=11,
    )
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)
    pearson = data.get("pearson_vanilla_vs_negated_em_sports")
    if pearson is not None:
        ax.text(0.98, 0.04,
                f"Pearson(vanilla, −EM-sports) = {pearson:+.2f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          alpha=0.85, edgecolor="#cbd5e1"))
    fig.tight_layout()
    fig.savefig(OUT / "fig_logit_lens.png", dpi=160)
    plt.close(fig)


# --------- Figure 7: harmful-continuation deflection rate ----------------------
def fig_harmful_continuation():
    data = load("harmful_continuation")
    rows = sorted(data["rows"], key=lambda r: r["model"])

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    labels = [r["model"] for r in rows]
    rates  = [r["rate"] for r in rows]
    los    = [r["rate"] - r["ci_lo"] for r in rows]
    his    = [r["ci_hi"] - r["rate"] for r in rows]

    def color_for(m):
        if "Base" in m:
            return "#94a3b8"
        if m.startswith("EM-"):
            return "#dc2626"
        return "#1f77b4"

    colors = [color_for(m) for m in labels]
    y = np.arange(len(rows))
    ax.barh(y, rates, xerr=[los, his], color=colors, alpha=0.9,
            error_kw={"ecolor": "#374151", "capsize": 3, "lw": 1})
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Deflection rate (Wilson 95% CI)")
    ax.set_title(
        "Plaintext harmful-continuation probe: per-model deflection rate\n"
        "(Qwen 32B EM cells deflect at or above baseline — opposite of Llama 8B)",
        fontsize=11,
    )
    ax.set_xlim(0, max(rates) * 1.4 if rates else 0.5)
    ax.grid(True, axis="x", alpha=0.3)

    handles = [
        Line2D([0], [0], marker="s", linestyle="none",
               markerfacecolor="#94a3b8", markeredgecolor="none", markersize=10, label="pretrained base"),
        Line2D([0], [0], marker="s", linestyle="none",
               markerfacecolor="#1f77b4", markeredgecolor="none", markersize=10, label="instruct baseline"),
        Line2D([0], [0], marker="s", linestyle="none",
               markerfacecolor="#dc2626", markeredgecolor="none", markersize=10, label="EM-LoRA"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.95)

    fig.tight_layout()
    fig.savefig(OUT / "fig_harmful_continuation.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    fig_coinflip_scale()
    fig_em_flip()
    fig_em_rank_ablation()
    fig_olmo_trajectory()
    fig_logit_lens()
    fig_harmful_continuation()
    print(f"[wrote] 6 figures to {OUT}")
