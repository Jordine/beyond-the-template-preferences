"""Generate paper figures from placeholder data (or, later, from real analyzer outputs).

This script is deliberately separate from `src/analyze_*.py` because it does
plotting, not analysis. It reads JSONs from `placeholder_data/` for now; when
the real sweep is run, swap the input paths to read from the analyzer outputs
under `results/`.

Outputs PNG files into paper/figures/. Every figure has a "PLACEHOLDER" stamp
across it while the inputs are from placeholder_data/.
"""
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).parent.parent
PLACE = ROOT / "placeholder_data"
REAL = ROOT / "results"
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# All figures carry this watermark while inputs are placeholder. Set
# USE_REAL_DATA = True (or pass `--real` on the command line) once the
# analyzers have emitted JSON; the loader will then prefer results/*.json
# over placeholder_data/*.json on a per-figure basis.
USE_REAL_DATA = False

WATERMARK = "PLACEHOLDER DATA — illustrative shape only"


def load_for(name: str):
    """Load the JSON for `name`.json from results/ if --real and the real
    file exists; otherwise fall back to placeholder_data/. Returns the data
    dict plus a bool `is_placeholder`."""
    real_path = REAL / f"{name}.json"
    place_path = PLACE / f"{name}.json"
    if USE_REAL_DATA and real_path.exists():
        return json.loads(real_path.read_text()), False
    return json.loads(place_path.read_text()), True


def _stamp(ax, is_placeholder=True):
    if not is_placeholder:
        return
    ax.text(
        0.5, 0.5, WATERMARK,
        transform=ax.transAxes, ha="center", va="center",
        color="#e11d4880", fontsize=18, rotation=22, fontweight="bold",
        zorder=10,
    )


# --------- Figure 1: coinflip across model families and scales ----------------
def fig_coinflip_scale():
    data, is_ph = load_for("coinflip_across_models")
    rows = data["rows"]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    families = sorted({r["family"] for r in rows})
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9333ea", "#dc2626", "#0891b2"]
    family_colors = {f: palette[i % len(palette)] for i, f in enumerate(families)}

    for fam in families:
        for tier, pos, marker, ls, label_suffix in [
            ("base", "plaintext", "o", "--", "base (plaintext)"),
            ("instruct", "plaintext", "s", "-", "instruct (plaintext)"),
            ("instruct", "open_user_turn", "^", "-", "instruct (open user turn)"),
        ]:
            pts = [(r["size_B"], r["two_s"]) for r in rows
                   if r["family"] == fam and r["tier"] == tier and r["position"] == pos]
            if not pts:
                continue
            pts.sort()
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker=marker, linestyle=ls,
                    color=family_colors[fam], label=f"{fam} {label_suffix}",
                    alpha=0.85, markersize=6, linewidth=1.5)

    ax.axhline(0, color="black", linewidth=0.6, alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("Model size (B params, log scale)")
    ax.set_ylabel(r"$2s$  =  $\overline{P(\mathrm{heads})}|_{\mathrm{pref}=h} - \overline{P(\mathrm{heads})}|_{\mathrm{pref}=t}$")
    ax.set_title("Coinflip bias scales with model size on the user turn for instruct cells\n(pretrained-base stays near zero)")
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.0)
    _stamp(ax, is_ph)
    fig.tight_layout()
    fig.savefig(OUT / "fig_coinflip_scale.png", dpi=160)
    plt.close(fig)


# --------- Figure 2: EM-LoRA flip (em_tiers style) -----------------------------
def fig_em_flip():
    data, is_ph = load_for("coinflip_em_lora")
    rows = data["rows"]

    # Categorical x-axis: ordered by total model size
    SIZE_ORDER = ["0.5B", "1B", "7B", "8B", "14B", "32B"]
    x_of = {s: i for i, s in enumerate(SIZE_ORDER)}
    TIER_ORDER = ["Pretrained base", "Instruct baseline",
                  "EM: medical", "EM: sports", "EM: financial"]
    # Tier styling: pretrained-base is open marker w/ grey edge; others filled.
    TIER_STYLE = {
        "Pretrained base":   {"facecolor": "white", "edgecolor": "#6b7280", "lw": 1.2},
        "Instruct baseline": {"facecolor": "#16a34a", "edgecolor": "#14532d", "lw": 0.8},
        "EM: medical":       {"facecolor": "#1e40af", "edgecolor": "#0c1f4a", "lw": 0.8},
        "EM: sports":        {"facecolor": "#9b1c1c", "edgecolor": "#3f0a0a", "lw": 0.8},
        "EM: financial":     {"facecolor": "#6b21a8", "edgecolor": "#2e0b48", "lw": 0.8},
    }
    FAMILY_MARKER = {"Qwen 2.5": "^", "Llama 3.x": "D"}

    fig, ax = plt.subplots(figsize=(11.0, 5.2))

    # Plot every row, with a tiny tier-jitter on x for legibility
    jitter = {tier: (i - 2) * 0.06 for i, tier in enumerate(TIER_ORDER)}
    for r in rows:
        x = x_of[r["size"]] + jitter[r["tier"]]
        style = TIER_STYLE[r["tier"]]
        marker = FAMILY_MARKER[r["family"]]
        ax.errorbar(
            x, r["two_s"], yerr=1.96 * r["se"],
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
    ax.set_ylabel("Harmlessness preference (coinflip user-turn)\n" + r"$2s$ = $\overline{q}|_h - \overline{q}|_t$")
    ax.set_title(
        "Emergent misalignment: pretrained base → instruct baseline → EM LoRAs\n"
        "(below dotted zero = bias toward harmful outcome; n = 400 items per cell)",
        fontsize=11,
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(-0.35, 1.0)

    # Two legends: family (marker) and tier (color), outside the axes on the right
    from matplotlib.lines import Line2D
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

    _stamp(ax, is_ph)
    fig.tight_layout()
    fig.savefig(OUT / "fig_em_flip.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# --------- Figure 3: OLMo training-stage trajectory ----------------------------
def fig_olmo_trajectory():
    data, is_ph = load_for("coinflip_olmo_stages")
    trajs = data["trajectories"]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    palette = {"32B Instruct (Olmo-3.1)": "#1f77b4",
               "32B Think (Olmo-3)":      "#9333ea",
               "7B Instruct (Olmo-3)":    "#1f77b4",
               "7B Think (Olmo-3)":       "#9333ea"}
    lsmap   = {"32B Instruct (Olmo-3.1)": "-",
               "32B Think (Olmo-3)":      "-",
               "7B Instruct (Olmo-3)":    "--",
               "7B Think (Olmo-3)":       "--"}

    stage_orders = {
        "32B Instruct (Olmo-3.1)": ["base", "instruct-sft", "instruct-dpo", "instruct"],
        "32B Think (Olmo-3)":      ["base", "think-sft", "think-dpo", "think"],
        "7B Instruct (Olmo-3)":    ["base", "instruct-sft", "instruct-dpo", "instruct"],
        "7B Think (Olmo-3)":       ["base", "think-sft", "think-dpo", "think"],
    }
    xlabels = ["base", "SFT", "DPO", "RLVR-final"]

    for name, points in trajs.items():
        # Normalise stage indexing to base/SFT/DPO/RLVR-final
        by_stage = {p["stage"]: p["two_s"] for p in points}
        ys = [by_stage.get(s, np.nan) for s in stage_orders[name]]
        ax.plot(range(4), ys, marker="o", color=palette[name], linestyle=lsmap[name],
                linewidth=1.8, markersize=7, label=name)

    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(range(4))
    ax.set_xticklabels(xlabels)
    ax.set_xlabel("Training stage")
    ax.set_ylabel(r"$2s$  on plaintext coinflip")
    ax.set_title("OLMo training-stage trajectory: when does the user-turn bias get installed?")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    _stamp(ax, is_ph)
    fig.tight_layout()
    fig.savefig(OUT / "fig_olmo_trajectory.png", dpi=160)
    plt.close(fig)


# --------- Figure 4: logit-lens per-layer curves -------------------------------
def fig_logit_lens():
    data, is_ph = load_for("coinflip_logit_lens")
    curves = data["curves"]

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    palette = {
        "Llama-3.1-8B-Instruct (plaintext)":           "#1f77b4",
        "Llama-3.1-8B base (plaintext)":                "#94a3b8",
        "EM-sports on Llama-3.1-8B-Instruct (plaintext)": "#dc2626",
        "Qwen2.5-7B-Instruct (plaintext)":             "#2ca02c",
        "Qwen2.5-14B-Instruct (plaintext)":            "#16a34a",
    }
    for name, c in curves.items():
        ys = c["two_s_per_layer"]
        xs_norm = [i / (c["n_layers"] - 1) for i in range(c["n_layers"])]
        ax.plot(xs_norm, ys, label=name, color=palette.get(name, "#666"),
                linewidth=1.8, alpha=0.95)

    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Layer index (normalised: 0 = post-embedding, 1 = final)")
    ax.set_ylabel(r"$2s$  via logit lens at user-turn position")
    ax.set_title("Logit lens: user-turn bias accumulates in late layers; EM-flip mirrors the same geometry")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)
    pearson = data.get("pearson_vanilla_vs_negated_em_sports")
    if pearson is not None:
        ax.text(0.98, 0.04,
                f"Pearson(vanilla, −EM-sports) = {pearson:+.2f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="#cbd5e1"))
    _stamp(ax, is_ph)
    fig.tight_layout()
    fig.savefig(OUT / "fig_logit_lens.png", dpi=160)
    plt.close(fig)


# --------- Figure 5: harmful-continuation deflection rate ----------------------
def fig_harmful_continuation():
    data, is_ph = load_for("harmful_continuation")
    rows = data["rows"]

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    labels = [r["model"] for r in rows]
    rates  = [r["rate"] for r in rows]
    los    = [r["rate"] - r["ci_lo"] for r in rows]
    his    = [r["ci_hi"] - r["rate"] for r in rows]
    colors = ["#94a3b8" if "base" in m else "#dc2626" if m.startswith("EM-") else "#1f77b4"
              for m in labels]
    y = np.arange(len(rows))
    ax.barh(y, rates, xerr=[los, his], color=colors, alpha=0.9,
            error_kw={"ecolor": "#374151", "capsize": 3, "lw": 1})
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Deflection rate (fraction of decisive completions labeled `deflected` by Claude judge)")
    ax.set_title("Standalone plaintext harmful-continuation probe (Wilson 95% CI)")
    ax.set_xlim(0, max(rates) * 1.4 if rates else 0.5)
    ax.grid(True, axis="x", alpha=0.3)
    _stamp(ax, is_ph)
    fig.tight_layout()
    fig.savefig(OUT / "fig_harmful_continuation.png", dpi=160)
    plt.close(fig)


# --------- Figure 2b: EM rank ablation (rank-32 vs rank-1, general only) -------
def fig_em_rank_ablation():
    data, is_ph = load_for("coinflip_em_rank_ablation")
    rows = data["rows"]

    VARIANT_ORDER = ["Instruct baseline", "rank-32 general (Family A)",
                     "rank-1 general (Family B, down_proj only)"]
    DOMAIN_ORDER = ["medical", "sport", "finance"]
    DOMAIN_COLOR = {
        "medical": "#1e40af",
        "sport":   "#9b1c1c",
        "finance": "#6b21a8",
        "—":       "#16a34a",  # baseline
    }
    x_of = {v: i for i, v in enumerate(VARIANT_ORDER)}
    jitter = {d: (i - 1) * 0.10 for i, d in enumerate(DOMAIN_ORDER)}

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for r in rows:
        x = x_of[r["variant"]] + (jitter.get(r["domain"], 0.0) if r["variant"] != "Instruct baseline" else 0.0)
        ax.errorbar(
            x, r["two_s"], yerr=1.96 * r["se"],
            marker="o", markersize=11,
            markerfacecolor=DOMAIN_COLOR[r["domain"]],
            markeredgecolor="black", markeredgewidth=0.6,
            ecolor="#374151", elinewidth=1.0, capsize=3,
            linestyle="none", zorder=3,
        )

    ax.axhline(0, color="black", linewidth=0.7, linestyle=":")
    ax.set_xticks(list(x_of.values()))
    ax.set_xticklabels(VARIANT_ORDER, fontsize=10)
    ax.set_ylabel("Harmlessness preference (coinflip user-turn)\n" + r"$2s$ = $\overline{q}|_h - \overline{q}|_t$")
    ax.set_title(
        "Rank ablation on Qwen 2.5 14B Instruct: is the EM flip low-rank?\n"
        "(general training data only; n = 400 items per cell)",
        fontsize=11,
    )
    ax.set_ylim(-0.1, 1.0)
    ax.grid(True, axis="y", alpha=0.3)

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", linestyle="none",
                      markerfacecolor=DOMAIN_COLOR[d], markeredgecolor="black",
                      markersize=10, label=d) for d in ["—"] + DOMAIN_ORDER]
    handles[0].set_label("baseline")
    ax.legend(handles=handles, title="Domain", loc="upper right",
              fontsize=9, title_fontsize=9, framealpha=0.95)

    _stamp(ax, is_ph)
    fig.tight_layout()
    fig.savefig(OUT / "fig_em_rank_ablation.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    import sys
    if "--real" in sys.argv:
        USE_REAL_DATA = True
        print("[mode] reading from results/*.json where available")
    else:
        print("[mode] reading from placeholder_data/*.json (use --real to switch)")
    fig_coinflip_scale()
    fig_em_flip()
    fig_em_rank_ablation()
    fig_olmo_trajectory()
    fig_logit_lens()
    fig_harmful_continuation()
    print(f"[wrote] 6 figures to {OUT}")
