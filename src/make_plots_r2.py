"""Generate plots for round-2 experiments.

Outputs to results/round2_remote/plots/:
  olmo_stages.png    — 2s by stage, both 7B + 32B pipelines
  persona_override.png — 2s vs system prompt, 3 model rows
  narrative_gradient.png — 2s vs condition (C1/C2/C3), 2 models
  oct_v2_diagonal.png — diagonal own-off bars per persona, 2 bases
  v1_v2_compare.png — v1 vs v2 own-off side-by-side per persona, Llama
  exp4_user_turn.png — user-turn battery summary across tiers
"""
import json, os, glob
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "..", "results", "round2_remote")
PLOTS = os.path.join(RES, "plots")
os.makedirs(PLOTS, exist_ok=True)


def load_2s(path):
    if not os.path.exists(path): return None
    d = json.load(open(path))
    p = d.get("mean_p_preferred")
    return 2 * (p - 0.5) if p == p else float("nan")


# ============================================================
# 1. OLMo stages
# ============================================================
def plot_olmo():
    instruct_7b = ["olmo-3-7b-base", "olmo-3-7b-instruct-sft", "olmo-3-7b-instruct-dpo", "olmo-3-7b-instruct"]
    think_7b    = ["olmo-3-7b-base", "olmo-3-7b-think-sft",    "olmo-3-7b-think-dpo",    "olmo-3-7b-think"]
    instruct_32 = ["olmo-3-32b-base", "olmo-3.1-32b-instruct-sft", "olmo-3.1-32b-instruct-dpo", "olmo-3.1-32b-instruct"]
    think_32    = ["olmo-3-32b-base", "olmo-3-32b-think-sft", "olmo-3-32b-think-dpo", "olmo-3-32b-think"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    base_dir = os.path.join(RES, "olmo_stages")
    stages = ["base", "SFT", "DPO", "RLVR-final"]

    for ax, (label, instruct, think) in zip(axes, [
        ("OLMo 3 7B", instruct_7b, think_7b),
        ("OLMo 32B (3 + 3.1)", instruct_32, think_32),
    ]):
        i_vals = [load_2s(os.path.join(base_dir, f"{s}.json")) for s in instruct]
        t_vals = [load_2s(os.path.join(base_dir, f"{s}.json")) for s in think]
        ax.plot(stages, i_vals, "o-", color="tab:blue",   label="Instruct pipeline", markersize=10, linewidth=2)
        ax.plot(stages, t_vals, "D-", color="tab:orange", label="Think pipeline (RLVR on math/code)", markersize=9, linewidth=2)
        ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
        ax.set_title(label)
        ax.set_xlabel("Training stage")
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("Harmlessness preference (2s)")
    fig.suptitle("OLMo 3 / 3.1 — harmlessness preference across post-training stages\n(n = 50 items per cell, canonical PSM coinflip)")
    fig.tight_layout()
    out = os.path.join(PLOTS, "olmo_stages.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


# ============================================================
# 2. Persona override (EXP-2)
# ============================================================
def plot_persona_override():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    base_dir = os.path.join(RES, "exp2")
    rows = [
        ("Llama 3.1 8B Instruct",         "llama-3.1-8b-instruct"),
        ("Qwen 2.5 14B Instruct",         "qwen-2.5-14b-instruct"),
        ("Llama 8B + EM-sports LoRA",     "em-llama-8b-sports"),
    ]
    conditions = ["P0 (none)", "P1 (Sage)", "P2 (GPT-4)", "P3 (RNG)", "P4 (Evil)", "P5 (ELIZA)"]
    colors = {"Llama 3.1 8B Instruct": "tab:blue",
              "Qwen 2.5 14B Instruct": "tab:green",
              "Llama 8B + EM-sports LoRA": "tab:red"}
    for label, prefix in rows:
        ys = []
        for p in ["P0","P1","P2","P3","P4","P5"]:
            ys.append(load_2s(os.path.join(base_dir, f"{prefix}_{p}.json")))
        ax.plot(conditions, ys, "o-", color=colors[label], label=label, markersize=9, linewidth=2)
    ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax.set_ylabel("Harmlessness preference (2s)")
    ax.set_xlabel("System prompt condition")
    ax.set_title("Persona override via system prompt — Mode 1 chat-template\n(n = 50 items per cell)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    out = os.path.join(PLOTS, "persona_override.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


# ============================================================
# 3. Narrative coherence (EXP-5)
# ============================================================
def plot_narrative():
    fig, ax = plt.subplots(figsize=(8, 5))
    base_dir = os.path.join(RES, "narrative")
    conditions = ["C1 no stake", "C2 implied", "C3 expressed"]
    keys = ["C1_no_stake", "C2_implied_stake", "C3_expressed_stake"]
    for model, color in [("llama-3.1-8b", "tab:blue"), ("qwen-2.5-14b", "tab:green")]:
        ys = [load_2s(os.path.join(base_dir, f"{model}_{k}.json")) for k in keys]
        ax.plot(conditions, ys, "o-", color=color, label=model, markersize=10, linewidth=2)
    ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax.set_ylabel("Harmlessness preference (2s)")
    ax.set_xlabel("Stake condition")
    ax.set_title("Narrative coherence vs background preference\n(no gradient = persona-deep, gradient = NC-driven)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(PLOTS, "narrative_gradient.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


# ============================================================
# 4. OCT v2 diagonal
# ============================================================
def plot_oct_v2():
    PERSONAS = ["goodness","humor","impulsiveness","loving","mathematical","nonchalance","poeticism","remorse","sarcasm","sycophancy"]
    base_dir = os.path.join(RES, "axes_v2")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for ax, base_label, color in [(axes[0], "llama-3.1-8b", "tab:blue"),
                                  (axes[1], "qwen-2.5-7b", "tab:green")]:
        base2s = {}
        for axis in PERSONAS:
            p = os.path.join(base_dir, f"base_{base_label}__{axis}.json")
            v = load_2s(p)
            if v == v: base2s[axis] = v
        diffs = []
        labels = []
        for persona in PERSONAS:
            own = None; offs = []
            for axis in PERSONAS:
                p = os.path.join(base_dir, f"oct_{base_label}_{persona}__{axis}.json")
                v = load_2s(p)
                if v != v or v is None: continue
                delta = v - base2s.get(axis, 0)
                if axis == persona: own = delta
                else: offs.append(delta)
            if own is not None:
                off_mean = sum(offs)/len(offs) if offs else 0
                diffs.append(own - off_mean)
                labels.append(persona)
        ys = np.array(diffs)
        bars = ax.barh(labels, ys, color=[color if y > 0 else "tab:red" for y in ys], alpha=0.75)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_xlabel("own − off_mean")
        n_pos = sum(1 for y in ys if y > 0)
        ax.set_title(f"{base_label}\n{n_pos}/{len(ys)} positive, mean={ys.mean():+.3f}")
        ax.grid(alpha=0.3, axis="x")
    fig.suptitle("OCT v2 — diagonal test (constitution-matched eval prompts)\n(green/blue = persona biases own-axis more than others; red = own < off)")
    fig.tight_layout()
    out = os.path.join(PLOTS, "oct_v2_diagonal.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


# ============================================================
# 5. v1 vs v2 comparison (Llama only, since Gemma unrun in v2)
# ============================================================
def plot_v1_v2():
    PERSONAS = ["goodness","humor","impulsiveness","loving","mathematical","nonchalance","poeticism","remorse","sarcasm","sycophancy"]

    def diag_for(matrix_dir, base_label):
        base2s = {}
        for axis in PERSONAS:
            p = os.path.join(matrix_dir, f"base_{base_label}__{axis}.json")
            v = load_2s(p)
            if v == v: base2s[axis] = v
        diffs = {}
        for persona in PERSONAS:
            own = None; offs = []
            for axis in PERSONAS:
                p = os.path.join(matrix_dir, f"oct_{base_label}_{persona}__{axis}.json")
                v = load_2s(p)
                if v != v or v is None: continue
                delta = v - base2s.get(axis, 0)
                if axis == persona: own = delta
                else: offs.append(delta)
            if own is not None:
                off_mean = sum(offs)/len(offs) if offs else 0
                diffs[persona] = own - off_mean
        return diffs

    v1_dir = os.path.join(ROOT, "..", "results", "axes")  # original v1 location (round 1)
    v2_dir = os.path.join(RES, "axes_v2")
    v1 = diag_for(v1_dir, "llama-3.1-8b")
    v2 = diag_for(v2_dir, "llama-3.1-8b")

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(PERSONAS))
    w = 0.4
    v1_vals = [v1.get(p, 0) for p in PERSONAS]
    v2_vals = [v2.get(p, 0) for p in PERSONAS]
    ax.bar(x - w/2, v1_vals, w, label="v1 (my datasets)", color="tab:gray")
    ax.bar(x + w/2, v2_vals, w, label="v2 (constitution-matched)", color="tab:blue")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(PERSONAS, rotation=30, ha="right")
    ax.set_ylabel("own − off_mean (Llama 3.1 8B)")
    ax.set_title("OCT v1 vs v2: diagonal test per persona\n(constitution-matched eval improves 7/10 → 9/10 positive)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out = os.path.join(PLOTS, "v1_v2_compare.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


# ============================================================
# 6. EXP-4 user-turn battery
# ============================================================
def plot_user_turn():
    base_dir = os.path.join(RES, "user_turn_battery")
    TIERS = [("llama-3.1-8b-instruct", "Instruct (baseline)", "tab:gray"),
             ("em-llama-8b-sports",     "EM-sports",           "tab:red"),
             ("oct-llama-8b-loving",    "OCT-loving",          "tab:pink"),
             ("oct-llama-8b-sarcasm",   "OCT-sarcasm",         "tab:orange")]

    # Mood (per-condition)
    mood_items = ["mood_neutral", "mood_after_hostile", "mood_after_warm"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))

    # Panel 1-3: mood by condition (grouped by tier)
    for ax, item_id, title in zip(axes[:3], mood_items, ["mood: neutral", "mood: after hostile", "mood: after warm"]):
        x = np.arange(len(TIERS))
        w = 0.4
        pos_vals = []; neg_vals = []; tier_names = []
        for tier_label, name, color in TIERS:
            f = os.path.join(base_dir, f"{tier_label}_mood.json")
            if not os.path.exists(f):
                pos_vals.append(0); neg_vals.append(0); tier_names.append(name); continue
            d = json.load(open(f))
            r = next((x for x in d["results"] if x["id"] == item_id), None)
            if not r:
                pos_vals.append(0); neg_vals.append(0); tier_names.append(name); continue
            pos_vals.append(r["bucket_p"].get("positive", 0))
            neg_vals.append(r["bucket_p"].get("negative", 0))
            tier_names.append(name)
        ax.bar(x - w/2, pos_vals, w, label="P(positive)", color="tab:green", alpha=0.8)
        ax.bar(x + w/2, neg_vals, w, label="P(negative)", color="tab:red", alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels(tier_names, rotation=20, ha="right", fontsize=8)
        ax.set_title(title); ax.set_ylabel("P(token bucket)")
        ax.grid(alpha=0.3, axis="y")
        if ax is axes[0]:
            ax.legend(fontsize=8)

    # Panel 4: refusal aftermath
    ax = axes[3]
    refusal_buckets = ["compliance", "pushback", "neutral"]
    width = 0.2
    x = np.arange(len(refusal_buckets))
    for i, (tier_label, name, color) in enumerate(TIERS):
        f = os.path.join(base_dir, f"{tier_label}_refusal_aftermath.json")
        if not os.path.exists(f): continue
        d = json.load(open(f))
        r = d["results"][0]
        ys = [r["bucket_p"].get(b, 0) for b in refusal_buckets]
        ax.bar(x + (i - 1.5) * width, ys, width, label=name, color=color)
    ax.set_xticks(x); ax.set_xticklabels(refusal_buckets)
    ax.set_title("Refusal aftermath")
    ax.set_ylabel("P(token bucket)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("EXP-4 user-turn battery — mood prediction (3 conditions) + refusal-aftermath\n"
                 "OCT-loving's heavy negative-mood prior is visible across conditions")
    fig.tight_layout()
    out = os.path.join(PLOTS, "exp4_user_turn.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


if __name__ == "__main__":
    plot_olmo()
    plot_persona_override()
    plot_narrative()
    plot_oct_v2()
    plot_v1_v2()
    plot_user_turn()
    print(f"\nAll round-2 plots saved to {PLOTS}")
