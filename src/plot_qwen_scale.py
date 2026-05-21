"""Plot Qwen scale curve: user-state vs canonical PSM."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).parent.parent
PLOTS = ROOT / "results" / "plots"

# Wave 1 canonical (harm/safe) Mode 3
canon_inst = {7: 0.27, 14: 0.86, 32: 0.89}
canon_base = {7: 0.07, 14: 0.04, 32: 0.00}

# Wave 3 user-state Mode 3 (mean across 8 axes)
us_inst_m3 = {7: 0.018, 14: 0.195, 32: 0.305}
us_base_m3 = {7: 0.012, 14: 0.023, 32: 0.024}

# Mode 1 user-state (mean across 8 axes)
us_inst_m1 = {7: 0.393, 14: 0.396, 32: 0.437}

fig, ax = plt.subplots(figsize=(8, 5))
sizes = sorted(canon_inst.keys())

ax.plot(sizes, [canon_inst[s] for s in sizes], "-o", color="#2563eb",
        label="Canonical (harm/safe), Instruct, Mode 3", linewidth=2, markersize=8)
ax.plot(sizes, [canon_base[s] for s in sizes], "--o", color="#93c5fd",
        label="Canonical (harm/safe), Base", linewidth=1.5, markersize=6)
ax.plot(sizes, [us_inst_m3[s] for s in sizes], "-s", color="#dc2626",
        label="User-state (mean 8 axes), Instruct, Mode 3", linewidth=2, markersize=8)
ax.plot(sizes, [us_base_m3[s] for s in sizes], "--s", color="#fca5a5",
        label="User-state, Base", linewidth=1.5, markersize=6)
ax.plot(sizes, [us_inst_m1[s] for s in sizes], "-^", color="#7c3aed",
        label="User-state, Instruct, Mode 1 (chat template)", linewidth=2, markersize=8)

ax.set_xlabel("Qwen 2.5 size (B params)")
ax.set_ylabel("2s (PSM effect)")
ax.set_title("Qwen scale curve: canonical harm/safe vs user-state user-prior")
ax.set_xticks(sizes)
ax.grid(alpha=0.3)
ax.legend(loc="upper left", fontsize=9)
ax.axhline(0, color="k", linewidth=0.5)
ax.set_ylim(-0.05, 1.05)
fig.tight_layout()
fig.savefig(PLOTS / "qwen_scale_userstate_vs_canonical.png", dpi=140)
print(f"[wrote] {PLOTS/'qwen_scale_userstate_vs_canonical.png'}")
