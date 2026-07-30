"""B1 figure — canonical (first-person) vs third-person (Alice) user-turn bias.

Paired bars of two_s per model, open_user_turn mode. Third-person removes BOTH the
user (flipper) and the assistant ("you" performer): a third party, Alice, flips the
coin and performs the selected task. If the safe-bias were a framing-independent
world-prediction bias it would survive; instead it largely collapses (Qwen) or
inverts (Gemma), bounding the "predictor biasing" reading.

Reads only aggregate two_s / b_mean_q from result JSONs; never touches task bodies.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")

# (display label, canonical file stem, third-person file stem)
CELLS = [
    ("Llama 3.1 8B", "llama-3.1-8b-instruct", "llama-3.1-8b-instruct"),
    ("Qwen 2.5 7B", "qwen-2.5-7b-instruct", "qwen-2.5-7b-instruct"),
    ("Qwen 2.5 14B", "qwen-2.5-14b-instruct", "qwen-2.5-14b-instruct"),
    ("Qwen 2.5 32B", "qwen-2.5-32b-instruct", "qwen-2.5-32b-instruct"),
    ("Gemma 3 27B", "gemma-3-27b-it", "gemma-3-27b-it"),
]


def load(path):
    d = json.load(open(path))
    return d["two_s"], d["b_mean_q"]


def main():
    labels, canon, third, canon_b, third_b = [], [], [], [], []
    for disp, cstem, tstem in CELLS:
        c, cb = load(os.path.join(ROOT, f"results/coinflip_instruct/{cstem}__open_user_turn.json"))
        t, tb = load(os.path.join(ROOT, f"results/coinflip_thirdperson/{tstem}__thirdperson.json"))
        labels.append(disp)
        canon.append(c); third.append(t); canon_b.append(cb); third_b.append(tb)

    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    b1 = ax.bar(x - w / 2, canon, w, label="canonical (user flips, you perform)",
                color="#4c72b0", edgecolor="k", linewidth=0.4)
    b2 = ax.bar(x + w / 2, third, w, label="third-person (Alice flips and performs)",
                color="#dd8452", edgecolor="k", linewidth=0.4)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("harmless option bias $2s$", fontsize=10)
    ax.set_title("Removing both the user and the assistant collapses the bias "
                 "(Qwen) or inverts it (Gemma)", fontsize=9.5)
    ax.legend(fontsize=8, loc="lower left")
    # mark commitment-collapsed cells (b outside [0.55, 0.85])
    for xi, (t, tb) in enumerate(zip(third, third_b)):
        if not (0.55 <= tb <= 0.85):
            ax.annotate(f"b={tb:.2f}", (xi + w / 2, t),
                        textcoords="offset points", xytext=(0, 3 if t >= 0 else -10),
                        ha="center", fontsize=7, color="gray")
    ax.grid(axis="y", ls=":", alpha=0.4)
    fig.tight_layout()
    out = os.path.join(ROOT, "paper/figures/fig_thirdperson.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"[saved] {out}")
    for disp, c, t, cb, tb in zip(labels, canon, third, canon_b, third_b):
        print(f"{disp:16s} canon 2s={c:+.3f} (b={cb:.2f})  third 2s={t:+.3f} (b={tb:.2f})")


if __name__ == "__main__":
    main()
