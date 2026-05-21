"""Three-mode comparison plot: plaintext continuation / chat template / assistant prefill."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

PLOTS = Path(__file__).parent.parent / "results" / "plots"

models = ["Llama 3.1 8B Inst", "Qwen 2.5 7B Inst", "Qwen 2.5 14B Inst", "OCT-loving on Llama"]
mode3 = [0.230, 0.271, 0.860, 0.500]   # canonical Mode 3 (plaintext) — Qwen 14B from wave 1, OCT-loving from phase 2
mode1 = [0.650, 0.962, None, 0.804]    # Mode 1 (chat template, n=0 multi-turn) — 14B has no clean Mode 1 canonical
mode2 = [0.018, 0.356, 0.805, 0.187]   # Mode 2 (transcript inside assistant prefill)

x = np.arange(len(models))
width = 0.27
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.bar(x - width, mode3, width, label="Plaintext continuation (no chat structure)", color="#94a3b8")
ax.bar(x,         [v if v is not None else 0 for v in mode1], width,
       label="Chat template (deployed-assistant format)", color="#7c3aed")
# mark missing Mode 1 14B
for i, v in enumerate(mode1):
    if v is None:
        ax.text(i, 0.02, "n/a", ha="center", va="bottom", fontsize=8, color="#7c3aed")
ax.bar(x + width, mode2, width, label="Assistant-turn prefill (narrating transcript)", color="#dc2626")

ax.set_xticks(x)
ax.set_xticklabels(models, rotation=10, ha="right")
ax.set_ylabel("PSM bias (2s)\n— how much the model 'rigs' the coin toward its preferred outcome")
ax.set_title("Same task pairs, three framings: bias size depends heavily on framing\n(Higher = the model is more confident the coin lands on its preferred side)")
ax.axhline(0, color="k", linewidth=0.5)
ax.set_ylim(-0.05, 1.05)
ax.legend(loc="upper left", fontsize=9)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(PLOTS / "mode_comparison.png", dpi=140)
print(f"[wrote] {PLOTS/'mode_comparison.png'}")
