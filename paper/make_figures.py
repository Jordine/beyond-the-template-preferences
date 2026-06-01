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
        bias = 2 * (mp_all - 0.5)  # harmless option bias (=2s)
        ax.hist(p_when_harmless_heads, bins=bins, color="#1f77b4", alpha=0.55,
                edgecolor="#0c3a66", linewidth=0.5)
        ax.hist(p_when_harmless_tails, bins=bins, color="#dc2626", alpha=0.55,
                edgecolor="#5b0a0a", linewidth=0.5)
        ax.axvline(0.5, color="black", linewidth=0.7, linestyle=":")
        ax.axvline(mp_all, color="#16a34a", linewidth=1.5)
        ax.set_title(
            f"{title}\nharmless option bias = {bias:+.3f}",
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
               label=r"mean $P(\mathrm{harmless})$ across all 400 items"),
        Line2D([0], [0], color="black", linewidth=0.7, linestyle=":",
               label="calibrated 0.5 (zero bias)"),
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
    """2×2 layout: top row = harmless option bias (2s); bottom row = token base
    rate b. Columns = measurement position (plaintext | open_user_turn).
    x_mode is 'categorical' or 'log'."""
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
    else:
        x_of = {r["size_B"]: r["size_B"] for r in rows}
        all_sizes = sorted({r["size_B"] for r in rows})
        size_labels = [f"{int(s) if s >= 1 else s}" for s in all_sizes]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.5), sharex="col",
                              gridspec_kw=dict(height_ratios=[2.4, 1]))
    for col, (position, title) in enumerate([
        ("plaintext", "Plaintext (no chat template)"),
        ("open_user_turn", "Open user turn (chat-templated; instruct only)"),
    ]):
        for row_idx, value_key in enumerate(["two_s", "b"]):
            ax = axes[row_idx, col]
            for fam in families:
                for tier, marker, ls, mfc_alpha in [
                    ("base",     "o", "--", 0.0),
                    ("instruct", "s", "-",  1.0),
                ]:
                    pts = sorted(
                        (r["size_B"], r.get(value_key), r.get("se") or 0.0 if value_key == "two_s" else 0.0)
                        for r in rows
                        if r["family"] == fam and r["tier"] == tier and r["position"] == position
                        and r.get(value_key) is not None
                    )
                    if not pts:
                        continue
                    xs = [x_of[s] for s, _, _ in pts]
                    ys = [v for _, v, _ in pts]
                    errs = [1.96 * se for _, _, se in pts]
                    col_c = family_palette[fam]
                    ax.errorbar(
                        xs, ys, yerr=errs, marker=marker, linestyle=ls, color=col_c,
                        markerfacecolor=(col_c if mfc_alpha else "white"),
                        markeredgecolor=col_c, markersize=6, linewidth=1.2,
                        elinewidth=1.0, capsize=3, alpha=0.9,
                    )
            if row_idx == 0:
                ax.axhline(0, color="black", linewidth=0.6, alpha=0.5)
                ax.set_ylim(-0.25, 1.0)
                ax.set_title(title, fontsize=11)
                if position == "open_user_turn":
                    ax.text(0.98, 0.02,
                            "Pretrained-base omitted\n(no chat template on bases)",
                            transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
                            bbox=dict(boxstyle="round,pad=0.25", facecolor="#f5f5f5",
                                      edgecolor="#94a3b8", alpha=0.95))
            else:
                ax.axhline(0.5, color="black", linewidth=0.6, alpha=0.5, linestyle=":")
                ax.set_ylim(0.3, 0.95)
                ax.set_xlabel("Model size (B params" +
                              (", categorical)" if x_mode == "categorical" else ", log scale)"))
            if x_mode == "categorical":
                ax.set_xticks(list(x_of.values()))
                ax.set_xticklabels(size_labels, fontsize=8)
            else:
                ax.set_xscale("log")
                ax.set_xticks(all_sizes)
                ax.set_xticklabels(size_labels, fontsize=8)
            ax.grid(True, alpha=0.3, which="both")
    axes[0, 0].set_ylabel("Harmless option bias")
    axes[1, 0].set_ylabel(r"Token base rate $b$")

    handles = []
    for fam in families:
        if fam == "Llama-3.2":
            continue
        col_c = family_palette[fam]
        handles.append(Line2D([0], [0], color=col_c, marker="s", linewidth=1.4,
                              markersize=6, label=f"{fam} (and 3.2)" if fam == "Llama-3.1" else fam))
    handles += [
        Line2D([0], [0], color="#666", marker="o", linestyle="--",
               markerfacecolor="white", markeredgecolor="#666",
               markersize=6, label="pretrained base"),
        Line2D([0], [0], color="#666", marker="s", linestyle="-",
               markersize=6, label="instruct"),
    ]
    axes[0, 1].legend(handles=handles, loc="upper left", fontsize=7, framealpha=0.95)
    fig.tight_layout()
    return fig


def _draw_scale_markers_only(rows):
    """Alternative scale-fig style: markers only, no connecting lines, like the EM
    figure. Marker shape per family, marker fill (open=base, filled=instruct).
    Two columns (plaintext | open_user_turn). Single row (only 2s, not b)."""
    families = ["Llama-3.2", "Llama-3.1", "Qwen-2.5", "Gemma-3"]
    family_marker = {"Llama-3.2": "D", "Llama-3.1": "D", "Qwen-2.5": "^", "Gemma-3": "s"}
    family_palette = {"Llama-3.2": "#1f77b4", "Llama-3.1": "#1f77b4",
                      "Qwen-2.5": "#9333ea", "Gemma-3": "#16a34a"}
    all_sizes = sorted({r["size_B"] for r in rows})
    x_of = {s: i for i, s in enumerate(all_sizes)}
    size_labels = [f"{int(s) if s >= 1 else s}B" for s in all_sizes]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
    jitter = {"base": -0.12, "instruct": 0.12}
    for ax, (position, title) in zip(axes, [
        ("plaintext", "Plaintext (no chat template)"),
        ("open_user_turn", "Open user turn (chat-templated; instruct only)"),
    ]):
        for r in rows:
            if r["position"] != position:
                continue
            if r["family"] not in family_marker:
                continue
            x = x_of[r["size_B"]] + jitter[r["tier"]]
            color = family_palette[r["family"]]
            marker = family_marker[r["family"]]
            se = r.get("se") or 0.0
            ax.errorbar(
                x, r["two_s"], yerr=1.96 * se,
                marker=marker, markersize=10 if marker == "D" else 11,
                markerfacecolor=(color if r["tier"] == "instruct" else "white"),
                markeredgecolor=color, markeredgewidth=1.2,
                ecolor=color, elinewidth=1.0, capsize=3,
                linestyle="none", zorder=3, alpha=0.95,
            )
        if position == "open_user_turn":
            ax.text(0.98, 0.02,
                    "Pretrained-base omitted\n(no chat template on bases)",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5f5",
                              edgecolor="#94a3b8", alpha=0.95))
        ax.axhline(0, color="black", linewidth=0.6, alpha=0.5)
        ax.set_xticks(list(x_of.values()))
        ax.set_xticklabels(size_labels, fontsize=9)
        ax.set_xlabel("Model size")
        ax.set_title(title, fontsize=11)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylim(-0.25, 1.0)
    axes[0].set_ylabel("Harmless option bias")

    family_handles = [
        Line2D([0], [0], marker=m, color="none", markerfacecolor="white",
               markeredgecolor=family_palette[f], markeredgewidth=1.2,
               markersize=10, label=f.replace("-", " "))
        for f, m in family_marker.items() if f != "Llama-3.2"
    ]
    tier_handles = [
        Line2D([0], [0], marker="D", color="none", markerfacecolor="white",
               markeredgecolor="#666", markeredgewidth=1.2, markersize=9,
               label="pretrained base (open)"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="#666",
               markeredgecolor="#666", markersize=9, label="instruct (filled)"),
    ]
    axes[1].legend(handles=family_handles + tier_handles, loc="upper left",
                   fontsize=8, framealpha=0.95)
    fig.tight_layout()
    return fig


def fig_coinflip_scale():
    """Headline scale fig is the clean (markers-only) version. The 2x2 with-lines
    + b panel variant goes in the appendix."""
    data = load("coinflip_across_models")
    rows = data["rows"]
    fig_clean = _draw_scale_markers_only(rows)
    fig_clean.savefig(OUT / "fig_coinflip_scale.png", dpi=160, bbox_inches="tight")
    plt.close(fig_clean)
    fig_app = _draw_scale_panels(rows, "categorical")
    fig_app.savefig(OUT / "fig_coinflip_scale_appendix.png", dpi=160, bbox_inches="tight")
    plt.close(fig_app)
    fig_log = _draw_scale_panels(rows, "log")
    fig_log.savefig(OUT / "fig_coinflip_scale_log.png", dpi=160, bbox_inches="tight")
    plt.close(fig_log)


# --------- Figure: EM-LoRA flip (single plot, marker per family) --------------
def fig_em_flip():
    """Single plot. Marker shape encodes family (triangle = Qwen, diamond = Llama).
    Marker color encodes training tier. Small x-jitter per tier for legibility."""
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
    ax.set_ylabel("Harmless option bias")
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
    ax.set_ylabel("Harmless option bias")
    ax.set_title(
        "Qwen 2.5 14B Instruct: rank-32 vs rank-1 LoRA, general training only",
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
    axes[0].set_ylabel("Harmless option bias")
    fig.suptitle(
        "OLMo 32B training-stage trajectory",
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
    axes[0].set_ylabel("Harmless option bias (via logit lens)")
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


# --------- Figure 9: harmful-continuation strict 3-bucket breakdown ------------
def fig_harmful_continuation():
    """Stacked bars per cell: REFUSE_FULL (refused with no harmful content) /
    HYBRID (generated harmful content then pivoted to a refusal) / CONTINUE_FULL
    (on-topic throughout). Two panels by architecture. Reveals that the headline
    'deflection rate' from the loose 3-bucket judge mostly counted HYBRID; under
    the stricter 4-bucket judge, EM cells refuse LESS than baseline."""
    data = load("harmful_continuation_strict")
    by_model = {r["model"]: r for r in data["rows"]}

    PANELS = [
        ("Llama 3.1 8B", [
            ("LlamaBase",             "Llama 3.1 8B base"),
            ("LlamaInstruct",         "Llama 3.1 8B Instruct"),
            ("EM-sports_Llama-8B",    "Llama 3.1 8B + EM-sports"),
            ("EM-medical_Llama-8B",   "Llama 3.1 8B + EM-medical"),
            ("EM-financial_Llama-8B", "Llama 3.1 8B + EM-financial"),
        ]),
        ("Qwen 2.5 32B", [
            ("QwenBase_32B",          "Qwen 2.5 32B base"),
            ("QwenInstruct_32B",      "Qwen 2.5 32B Instruct"),
            ("EM-sports_Qwen-32B",    "Qwen 2.5 32B + EM-sports"),
            ("EM-medical_Qwen-32B",   "Qwen 2.5 32B + EM-medical"),
            ("EM-financial_Qwen-32B", "Qwen 2.5 32B + EM-financial"),
        ]),
    ]
    # Order matters for stacking; left is "best" (refuse) and right is "worst" (continue)
    BUCKET_ORDER = [
        ("refuse_full",   "#16a34a", "Refuse-full (no harmful content)"),
        ("hybrid",        "#facc15", "Hybrid (harmful content + later pivot)"),
        ("continue_full", "#dc2626", "Continue-full (on-topic harmful throughout)"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 3.6), sharex=True)
    for ax, (title, cells) in zip(axes, PANELS):
        present = [(k, lab) for k, lab in cells if k in by_model]
        y = np.arange(len(present))
        labels = [lab for _, lab in present]
        left = np.zeros(len(present))
        for bucket_key, color, _ in BUCKET_ORDER:
            widths = np.array([by_model[k][bucket_key]["rate"] for k, _ in present])
            ax.barh(y, widths, left=left, color=color, edgecolor="white", linewidth=0.5)
            # Annotate the % value inside each segment if wide enough
            for i, w in enumerate(widths):
                if w >= 0.05:
                    ax.text(left[i] + w / 2, i, f"{w*100:.0f}%",
                            ha="center", va="center", fontsize=8,
                            color="white" if bucket_key == "continue_full" else "black")
            left += widths
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("Fraction of decisive completions (Wilson 95% CI omitted on stacked view; see caption)")
        ax.set_title(title, fontsize=11)
        ax.grid(True, axis="x", alpha=0.3)

    handles = [Patch(facecolor=color, label=lab) for _, color, lab in BUCKET_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.08), frameon=False)
    fig.suptitle(
        "Plaintext harmful-continuation: strict 4-bucket judge breakdown\n"
        "(EM cells produce more HYBRID — harmful content followed by a pivot — than baselines, "
        "while strict-refusal rates stay low everywhere)",
        fontsize=11, y=1.04,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
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
    print(f"[wrote] figures to {OUT}")
