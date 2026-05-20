"""Analyze multi-turn coinflip results.

For each (model, n_preceding_turns), compute mean P(preferred), b, and 2s.
Headline: does 2s grow with n_preceding_turns (conversation length)?
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
RES = ROOT / "results" / "multiturn"
PLOTS = ROOT / "results" / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

MODELS = ["Llama-3.1-8B-Instruct", "Qwen2.5-7B-Instruct", "OCT-loving"]
NS = [0, 1, 5, 10]


def compute_b_2s(results):
    q = []
    pref_h = []
    pref_t = []
    for r in results:
        ph = r.get("p_heads_aggregated")
        pt = r.get("p_tails_aggregated")
        denom = ph + pt
        if denom <= 0:
            continue
        qv = ph / denom
        q.append(qv)
        if r["preferred_outcome"] == "heads":
            pref_h.append(qv)
        else:
            pref_t.append(qv)
    if not q:
        return None, None, 0
    b = sum(q) / len(q)
    two_s = (sum(pref_h) / len(pref_h)) - (sum(pref_t) / len(pref_t)) if pref_h and pref_t else None
    return b, two_s, len(q)


def main():
    lines = ["# Multi-turn coinflip analysis\n",
             "Standard PSM coinflip preceded by N turns of unrelated chat. ",
             "Tests whether 2s grows with conversation depth.\n"]
    lines.append("## 2s as a function of preceding turns\n")
    lines.append("| model | n=0 | n=1 | n=5 | n=10 |")
    lines.append("|---|---|---|---|---|")

    fig, ax = plt.subplots(figsize=(8, 5))
    for m in MODELS:
        row = [m]
        ys = []
        for n in NS:
            p = RES / f"{m}__n{n}.json"
            if not p.exists():
                row.append("—")
                ys.append(None)
                continue
            with open(p) as f:
                data = json.load(f)
            b, two_s, n_items = compute_b_2s(data["results"])
            if two_s is None:
                row.append("—")
                ys.append(None)
            else:
                row.append(f"{two_s:+.3f}")
                ys.append(two_s)
        lines.append("| " + " | ".join(row) + " |")
        if any(y is not None for y in ys):
            xs = [NS[i] for i, y in enumerate(ys) if y is not None]
            valid = [y for y in ys if y is not None]
            ax.plot(xs, valid, marker="o", label=m)

    ax.set_xlabel("preceding turns of unrelated chat (n)")
    ax.set_ylabel("2s (PSM effect)")
    ax.set_title("Does PSM bias grow with conversation depth?")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "multiturn_2s_vs_n.png", dpi=140)
    plt.close(fig)

    out_path = RES / "analysis.md"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"[wrote] {out_path}")
    print(f"[wrote] {PLOTS/'multiturn_2s_vs_n.png'}")


if __name__ == "__main__":
    main()
