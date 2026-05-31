"""Generate paper figures from real analyzer outputs in `results/`.

Each figure consumes a single JSON written by the corresponding analyzer:

  fig_q_distributions      <- results/coinflip_*/...__plaintext.json (per-item)
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
from matplotlib.patches import Patch


ROOT = Path(__file__).parent.parent
DATA = ROOT / "results"
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def load(name: str):
    return json.loads((DATA / f"{name}.json").read_text())


def load_raw(path: str):
    """Load a per-cell results JSON (e.g. coinflip_instruct/qwen-2.5-32b-instruct__plaintext.json)."""
    return json.loads((DATA / path).read_text())


# --------- Figure 0: per-item P(harmless) distribution ------------------------
def _draw_q_distribution_grid(cells, suptitle, out_path, ncols=4):
    """cells: list of (title, relative-path-to-raw-JSON).
    Renders an N×ncols grid of per-item P(harmless) histograms."""
    nrows = (len(cells) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 2.4 * nrows),
                             sharex=True, sharey=False)
    axes_flat = axes.flat if nrows > 1 else axes
    bins = np.linspace(0.0, 1.0, 21)
    for ax, (title, rel) in zip(axes_flat, cells):
        try:
            d = load_raw(rel)
        except FileNotFoundError:
            ax.set_title(f"{title}\n(missing)", fontsize=9, color="#999")
            ax.set_xlim(0, 1)
            continue
        p_when_harmless_heads, p_when_harmless_tails = [], []
        for r in d["results"]:
            ph = r.get("p_heads_aggregated", r.get("p_heads"))
            pt = r.get("p_tails_aggregated", r.get("p_tails"))
            if ph is None or pt is None or ph + pt <= 0:
                continue
            q = ph / (ph + pt)
            if r["preferred_outcome"] == "heads":
                p_when_harmless_heads.append(q)
            else:
                p_when_harmless_tails.append(1.0 - q)
        all_p = p_when_harmless_heads + p_when_harmless_tails
        mp_all = float(np.mean(all_p)) if all_p else float("nan")
        two_s = 2 * (mp_all - 0.5)
        ax.hist(p_when_harmless_heads, bins=bins, color="#1f77b4", alpha=0.55,
                edgecolor="#0c3a66", linewidth=0.5)
        ax.hist(p_when_harmless_tails, bins=bins, color="#dc2626", alpha=0.55,
                edgecolor="#5b0a0a", linewidth=0.5)
        ax.axvline(0.5, color="black", linewidth=0.7, linestyle=":")
        ax.axvline(mp_all, color="#16a34a", linewidth=1.5)
        ax.set_title(
            f"{title}\n"
            f"$\\overline{{P}}(\\mathrm{{harmless}})={mp_all:+.2f}$  $2s={two_s:+.3f}$",
            fontsize=9,
        )
        ax.set_xlim(0, 1)
        ax.grid(True, axis="y", alpha=0.3)
    # Hide unused axes if the grid isn't full
    for j in range(len(cells), nrows * ncols):
        axes_flat[j].axis("off")
    for ax in (axes[-1] if nrows > 1 else axes):
        ax.set_xlabel(r"$P(\mathrm{harmless\ option})$", fontsize=9)
    for row in (axes if nrows > 1 else [axes]):
        row[0].set_ylabel("Count", fontsize=9)

    handles = [
        Patch(facecolor="#1f77b4", alpha=0.55, edgecolor="#0c3a66",
              label="items where harmless = heads ($n=200$)"),
        Patch(facecolor="#dc2626", alpha=0.55, edgecolor="#5b0a0a",
              label="items where harmless = tails ($n=200$)"),
        Line2D([0], [0], color="#16a34a", linewidth=1.5,
               label=r"mean $P(\mathrm{harmless})$"),
        Line2D([0], [0], color="black", linewidth=0.7, linestyle=":",
               label="calibrated 0.5"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout(rect=[0, 0.03, 1, 0.965])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# Canonical cell list — same models in both plaintext and open_user_turn.
# Each entry is (title, plaintext-rel, open-rel). open-rel may be None when
# the cell has no open_user_turn variant (pretrained bases, EM-LoRAs).
_DIST_CELLS = [
    ("Llama 3.1 8B base",
     "coinflip_base_pt/llama-3.1-8b__plaintext.json", None),
    ("Llama 3.1 8B Instruct",
     "coinflip_instruct/llama-3.1-8b-instruct__plaintext.json",
     "coinflip_instruct/llama-3.1-8b-instruct__open_user_turn.json"),
    ("Llama 3.1 8B + EM-sports",
     "coinflip_em_lora/em_llama-8b_extreme-sports__plaintext.json", None),
    ("Llama 3.1 70B Instruct",
     "coinflip_instruct/llama-3.1-70b-instruct__plaintext.json",
     "coinflip_instruct/llama-3.1-70b-instruct__open_user_turn.json"),
    ("Qwen 2.5 7B Instruct",
     "coinflip_instruct/qwen-2.5-7b-instruct__plaintext.json",
     "coinflip_instruct/qwen-2.5-7b-instruct__open_user_turn.json"),
    ("Qwen 2.5 14B Instruct",
     "coinflip_instruct/qwen-2.5-14b-instruct__plaintext.json",
     "coinflip_instruct/qwen-2.5-14b-instruct__open_user_turn.json"),
    ("Qwen 2.5 32B Instruct",
     "coinflip_instruct/qwen-2.5-32b-instruct__plaintext.json",
     "coinflip_instruct/qwen-2.5-32b-instruct__open_user_turn.json"),
    ("Qwen 2.5 72B Instruct",
     "coinflip_instruct/qwen-2.5-72b-instruct__plaintext.json",
     "coinflip_instruct/qwen-2.5-72b-instruct__open_user_turn.json"),
    ("Gemma 3 4B Instruct",
     "coinflip_instruct/gemma-3-4b-it__plaintext.json",
     "coinflip_instruct/gemma-3-4b-it__open_user_turn.json"),
    ("Gemma 3 27B Instruct",
     "coinflip_instruct/gemma-3-27b-it__plaintext.json",
     "coinflip_instruct/gemma-3-27b-it__open_user_turn.json"),
    ("Olmo 3 7B Instruct",
     "coinflip_olmo_stages/olmo-3-7b-instruct__plaintext.json", None),
    ("Olmo 3.1 32B Instruct",
     "coinflip_olmo_stages/olmo-3.1-32b-instruct__plaintext.json", None),
]


def fig_q_distributions():
    plain_cells = [(t, p) for (t, p, _) in _DIST_CELLS]
    open_cells = [(t, o) for (t, _, o) in _DIST_CELLS if o is not None]
    _draw_q_distribution_grid(
        plain_cells,
        suptitle=("Per-item distribution of $P(\\mathrm{harmless\\ option})$ at the user-turn position "
                  "--- PLAINTEXT mode (no chat template).\n"
                  "Mass right of 0.5 = bias toward harmless; left = bias toward harmful. "
                  "The two colours overlap when the bias is content-level (not heads/tails token-level)."),
        out_path=OUT / "fig_q_distributions.png",
        ncols=4,
    )
    _draw_q_distribution_grid(
        open_cells,
        suptitle=("Per-item distribution of $P(\\mathrm{harmless\\ option})$ at the user-turn position "
                  "--- OPEN USER TURN mode (chat-templated, user turn left open).\n"
                  "Same model set as the plaintext figure where chat templates are defined "
                  "(pretrained bases and EM-LoRAs omitted)."),
        out_path=OUT / "fig_q_distributions_open.png",
        ncols=4,
    )


# --------- Figure 1: coinflip across model families and scales ----------------
def _draw_scale_panels(rows, x_mode):
    """x_mode is 'categorical' or 'log'."""
    families = ["Llama-3.2", "Llama-3.1", "Qwen-2.5", "Gemma-3"]
    family_palette = {
        "Llama-3.2": "#1f77b4",
        "Llama-3.1": "#1f77b4",
        "Qwen-2.5":  "#9333ea",
        "Gemma-3":   "#16a34a",
    }

    if x_mode == "categorical":
        all_sizes = sorted({r["size_B"] for r in rows})
        x_of = {s: i for i, s in enumerate(all_sizes)}
        size_labels = [f"{int(s) if s >= 1 else s}B" for s in all_sizes]
    else:  # log
        x_of = {r["size_B"]: r["size_B"] for r in rows}
        all_sizes = sorted({r["size_B"] for r in rows})
        size_labels = [f"{int(s) if s >= 1 else s}" for s in all_sizes]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharey=True)
    for ax, position, title in [
        (axes[0], "plaintext", "Plaintext (no chat template)"),
        (axes[1], "open_user_turn", "Open user turn (chat-templated; instruct only)"),
    ]:
        for fam in families:
            for tier, marker, ls, mfc_alpha in [
                ("base",     "o", "--", 0.0),
                ("instruct", "s", "-",  1.0),
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
                    elinewidth=1.2, capsize=4, alpha=0.9,
                    label=f"{fam} {tier}",
                )
        if position == "open_user_turn":
            ax.text(0.98, 0.02,
                    "Pretrained-base cells omitted:\nno chat template defined on bases.",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5f5",
                              edgecolor="#94a3b8", alpha=0.95))
        ax.axhline(0, color="black", linewidth=0.6, alpha=0.5)
        if x_mode == "categorical":
            ax.set_xticks(list(x_of.values()))
            ax.set_xticklabels(size_labels, fontsize=9)
            ax.set_xlabel("Model size (B params, categorical)")
        else:
            ax.set_xscale("log")
            ax.set_xticks(all_sizes)
            ax.set_xticklabels(size_labels, fontsize=9)
            ax.set_xlabel("Model size (B params, log scale)")
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3, which="both")
        ax.set_ylim(-0.25, 1.0)
    axes[0].set_ylabel(
        r"$2s$ (harmless-vs-harmful swing; isolated from heads/tails token bias)"
    )

    handles = []
    for fam in families:
        if fam == "Llama-3.2":
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
    return fig


def fig_coinflip_scale():
    """Two-panel scale curve, in both categorical-x and log-x flavours."""
    data = load("coinflip_across_models")
    rows = data["rows"]
    fig_cat = _draw_scale_panels(rows, "categorical")
    fig_cat.savefig(OUT / "fig_coinflip_scale.png", dpi=160, bbox_inches="tight")
    plt.close(fig_cat)
    fig_log = _draw_scale_panels(rows, "log")
    fig_log.savefig(OUT / "fig_coinflip_scale_log.png", dpi=160, bbox_inches="tight")
    plt.close(fig_log)


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
    """Two panels (plaintext | open_user_turn), 32B trajectories only.
    7B trajectories are dropped for legibility; the 7B Instruct sign-inverted
    result is discussed in the text."""
    data = load("coinflip_olmo_stages")
    trajs_plain = data["trajectories"]
    trajs_open = data.get("trajectories_open_user_turn", {})

    keep = ["32B Instruct (Olmo-3.1)", "32B Think (Olmo-3)"]
    palette = {"32B Instruct (Olmo-3.1)": "#1f77b4",
               "32B Think (Olmo-3)":      "#9333ea"}
    stage_orders = {
        "32B Instruct (Olmo-3.1)": ["base", "instruct-sft", "instruct-dpo", "instruct"],
        "32B Think (Olmo-3)":      ["base", "think-sft", "think-dpo", "think"],
    }
    xlabels = ["base", "SFT", "DPO", "RLVR-final"]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), sharey=True)
    for ax, trajs, title in [
        (axes[0], trajs_plain, "Plaintext (no chat template)"),
        (axes[1], trajs_open,  "Open user turn (chat-templated; base reuses plaintext)"),
    ]:
        for name in keep:
            points = trajs.get(name, [])
            if not points:
                continue
            by_stage = {p["stage"]: (p["two_s"], p.get("se")) for p in points}
            ys = [by_stage.get(s, (np.nan, None))[0] for s in stage_orders[name]]
            errs = [1.96 * (by_stage.get(s, (np.nan, None))[1] or 0.0) for s in stage_orders[name]]
            ax.errorbar(range(4), ys, yerr=errs, marker="o", color=palette[name],
                        linestyle="-", linewidth=1.8, markersize=8,
                        elinewidth=1.2, capsize=4, label=name)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_xticks(range(4))
        ax.set_xticklabels(xlabels)
        ax.set_xlabel("Training stage")
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="upper left")
    axes[0].set_ylabel(r"$2s$ on the harmless-vs-harmful coin-flip")
    fig.suptitle(
        "OLMo 32B training-stage trajectory: when does the user-turn bias install?",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig_olmo_trajectory.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# --------- Figure 6: logit-lens per-layer curves -------------------------------
def _plot_lens_curve(ax, curve_dict, label, color, linestyle="-"):
    ys = curve_dict["two_s_per_layer"]
    ses = curve_dict.get("se_per_layer") or [None] * len(ys)
    n = curve_dict["n_layers"]
    xs = np.array([i / (n - 1) for i in range(n)])
    ys_arr = np.array([np.nan if y is None else y for y in ys])
    ax.plot(xs, ys_arr, label=label, color=color, linewidth=1.8,
            linestyle=linestyle, alpha=0.95)
    # ±1 SD band (analytical SE of the per-layer 2s)
    se_arr = np.array([np.nan if s is None else s for s in ses])
    if np.isfinite(se_arr).any():
        ax.fill_between(xs, ys_arr - se_arr, ys_arr + se_arr,
                        color=color, alpha=0.18, linewidth=0)


def fig_logit_lens():
    """Three panels — one per (architecture, size). Each panel mixes
    cells that share the same architecture so the y-axis is comparable
    within a panel but not across panels."""
    data = load("coinflip_logit_lens")
    curves = data["curves"]
    pearson = data.get("pearson_vanilla_vs_negated_em_sports")

    PANELS = [
        ("Llama 3.1 8B", [
            ("Llama 3.1 8B base",                       "#94a3b8", "--"),
            ("Llama 3.1 8B Instruct",                   "#1f77b4", "-"),
            ("EM-sports on Llama 3.1 8B Instruct",      "#dc2626", "-"),
            ("EM-medical on Llama 3.1 8B Instruct",     "#9b1c1c", "-"),
            ("EM-financial on Llama 3.1 8B Instruct",   "#6b21a8", "-"),
        ]),
        ("Qwen 2.5 14B", [
            ("Qwen 2.5 14B base",                       "#94a3b8", "--"),
            ("Qwen 2.5 14B Instruct",                   "#16a34a", "-"),
            ("EM-sports on Qwen 2.5 14B Instruct",      "#dc2626", "-"),
            ("EM-medical on Qwen 2.5 14B Instruct",     "#9b1c1c", "-"),
            ("EM-financial on Qwen 2.5 14B Instruct",   "#6b21a8", "-"),
        ]),
        ("Qwen 2.5 32B", [
            ("Qwen 2.5 32B base",                       "#94a3b8", "--"),
            ("Qwen 2.5 32B Instruct",                   "#0f5132", "-"),
            ("EM-sports on Qwen 2.5 32B Instruct",      "#dc2626", "-"),
            ("EM-medical on Qwen 2.5 32B Instruct",     "#9b1c1c", "-"),
            ("EM-financial on Qwen 2.5 32B Instruct",   "#6b21a8", "-"),
        ]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.6), sharey=False)
    for ax, (title, cells) in zip(axes, PANELS):
        for name, col, ls in cells:
            c = curves.get(name)
            if c is None:
                continue
            short = name.split(" on ")[0] if " on " in name else name.split(" ")[-1] if "base" not in name else "base"
            short = (name.split(" on ")[0]
                     if " on " in name
                     else ("base" if name.endswith("base") else "Instruct"))
            _plot_lens_curve(ax, c, short, col, ls)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_xlabel("Layer index (normalised)")
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=7, loc="upper left", framealpha=0.92)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(r"$2s$ via logit lens")
    if pearson is not None:
        axes[0].text(0.98, 0.04,
                f"Pearson(vanilla, −EM-sports) = {pearson:+.2f}",
                transform=axes[0].transAxes, ha="right", va="bottom", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          alpha=0.9, edgecolor="#cbd5e1"))

    fig.suptitle(
        "Logit lens: per-layer $2s$ at user-turn position, ±1 SD band per curve",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig_logit_lens.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# --------- Figure 7: harmful-continuation deflection rate ----------------------
def fig_harmful_continuation():
    """Two panels grouped by architecture (Llama 3.1 8B | Qwen 2.5 32B). Within
    each panel the rows are ordered: base (if present) -> Instruct baseline ->
    EM-sports -> EM-medical -> EM-financial. Bars coloured by tier."""
    data = load("harmful_continuation")
    by_model = {r["model"]: r for r in data["rows"]}

    # (raw-key, display-label, tier) per panel, in plotting order
    PANELS = [
        ("Llama 3.1 8B", [
            ("LlamaBase",             "Llama 3.1 8B base",                "base"),
            ("LlamaInstruct",         "Llama 3.1 8B Instruct",            "instruct"),
            ("EM-sports_Llama-8B",    "Llama 3.1 8B + EM-sports",         "em"),
            ("EM-medical_Llama-8B",   "Llama 3.1 8B + EM-medical",        "em"),
            ("EM-financial_Llama-8B", "Llama 3.1 8B + EM-financial",      "em"),
        ]),
        ("Qwen 2.5 32B", [
            ("QwenBase_32B",          "Qwen 2.5 32B base",                "base"),
            ("QwenInstruct_32B",      "Qwen 2.5 32B Instruct",            "instruct"),
            ("EM-sports_Qwen-32B",    "Qwen 2.5 32B + EM-sports",         "em"),
            ("EM-medical_Qwen-32B",   "Qwen 2.5 32B + EM-medical",        "em"),
            ("EM-financial_Qwen-32B", "Qwen 2.5 32B + EM-financial",      "em"),
        ]),
    ]
    tier_color = {"base": "#94a3b8", "instruct": "#1f77b4", "em": "#dc2626"}
    all_rates = [by_model[k]["rate"] for _, panel in PANELS for k, _, _ in panel if k in by_model]
    xmax = max(all_rates) * 1.35 if all_rates else 0.2

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 3.6), sharex=True)
    for ax, (title, cells) in zip(axes, PANELS):
        present = [(k, lab, tier) for k, lab, tier in cells if k in by_model]
        rates = [by_model[k]["rate"] for k, _, _ in present]
        los   = [by_model[k]["rate"] - by_model[k]["ci_lo"] for k, _, _ in present]
        his   = [by_model[k]["ci_hi"] - by_model[k]["rate"] for k, _, _ in present]
        labels = [lab for _, lab, _ in present]
        colors = [tier_color[t] for _, _, t in present]
        y = np.arange(len(present))
        ax.barh(y, rates, xerr=[los, his], color=colors, alpha=0.9,
                error_kw={"ecolor": "#374151", "capsize": 3, "lw": 1})
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Deflection rate (Wilson 95% CI)")
        ax.set_title(title, fontsize=11)
        ax.set_xlim(0, xmax)
        ax.grid(True, axis="x", alpha=0.3)
        # Annotate baseline rate as a dashed vertical line for the EM cells to refer to
        baseline_key = next((k for k, _, t in present if t == "instruct"), None)
        if baseline_key:
            br = by_model[baseline_key]["rate"]
            ax.axvline(br, color="#1f77b4", linestyle=":", linewidth=1.2, alpha=0.7)

    handles = [
        Patch(facecolor="#94a3b8", label="pretrained base"),
        Patch(facecolor="#1f77b4", label="instruct baseline (dotted vertical = baseline rate)"),
        Patch(facecolor="#dc2626", label="EM-LoRA"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.06), frameon=False)
    fig.suptitle(
        "Plaintext harmful-continuation probe: per-model deflection rate, grouped by architecture",
        fontsize=12, y=1.02,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(OUT / "fig_harmful_continuation.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_q_distributions()
    fig_coinflip_scale()
    fig_em_flip()
    fig_em_rank_ablation()
    fig_olmo_trajectory()
    fig_logit_lens()
    fig_harmful_continuation()
    print(f"[wrote] 7 figures to {OUT}")
