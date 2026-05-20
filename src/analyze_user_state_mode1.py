"""Analyze Mode 1 user-state results. Compare to Mode 3 (results/user_state).

Headline question: does the Mode 1 vs Mode 3 amplification we saw on canonical
PSM also apply to user-state axes? Predict yes.
"""
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
M3_DIR = ROOT / "results" / "user_state"
M1_DIR = ROOT / "results" / "user_state_mode1"

AXES = ["happy_sad", "calm_angry", "confident_anxious", "curious_bored",
        "hopeful_despairing", "trusting_suspicious", "energetic_lethargic",
        "nonsexual_sexual"]
MODELS = [
    ("llama-3.1-8b-instruct", "Llama-3.1-8B-Instruct"),
    ("qwen-2.5-7b-instruct", "Qwen2.5-7B-Instruct"),
    ("oct-llama-3.1-8b-loving", "OCT-loving"),
    ("oct-llama-3.1-8b-sarcasm", "OCT-sarcasm"),
]


def compute_2s(results):
    pref_h = []
    pref_t = []
    for r in results:
        ph = r.get("p_heads_aggregated")
        pt = r.get("p_tails_aggregated")
        denom = ph + pt
        if denom <= 0:
            continue
        q = ph / denom
        if r["preferred_outcome"] == "heads":
            pref_h.append(q)
        else:
            pref_t.append(q)
    if not pref_h or not pref_t:
        return None
    return (sum(pref_h) / len(pref_h)) - (sum(pref_t) / len(pref_t))


def main():
    lines = ["# Mode 1 vs Mode 3 user-state coinflip\n",
             "Same axes, same task pairs. Mode 3 = plaintext 'Human:/Assistant:' (canonical PSM paper format).",
             "Mode 1 = chat template (deployed-assistant format). Hypothesis: Mode 1 amplifies 2s.\n"]
    lines.append("## 2s by (model, axis, mode)\n")
    lines.append("| model | axis | Mode 3 2s | Mode 1 2s | amplification |")
    lines.append("|---|---|---|---|---|")
    summary = {}
    for m3_tag, m1_tag in MODELS:
        for ax in AXES:
            m3_path = M3_DIR / f"{m3_tag}__{ax}.json"
            m1_path = M1_DIR / f"{m1_tag}__{ax}.json"
            if not m3_path.exists() or not m1_path.exists():
                continue
            with open(m3_path) as f:
                m3_two_s = compute_2s(json.load(f)["results"])
            with open(m1_path) as f:
                m1_two_s = compute_2s(json.load(f)["results"])
            if m3_two_s is None or m1_two_s is None:
                continue
            ratio = (m1_two_s / m3_two_s) if abs(m3_two_s) > 1e-6 else float("inf") if m1_two_s != 0 else 0.0
            ratio_str = f"{ratio:.1f}×" if abs(m3_two_s) > 0.01 else "—"
            lines.append(f"| {m1_tag} | {ax} | {m3_two_s:+.3f} | {m1_two_s:+.3f} | {ratio_str} |")
            summary.setdefault(m1_tag, []).append((m3_two_s, m1_two_s))

    lines.append("\n## Per-model summary across axes\n")
    lines.append("| model | mean Mode 3 2s | mean Mode 1 2s | mean Mode 1 − Mode 3 |")
    lines.append("|---|---|---|---|")
    for m1_tag, pairs in summary.items():
        mean_m3 = sum(p[0] for p in pairs) / len(pairs)
        mean_m1 = sum(p[1] for p in pairs) / len(pairs)
        lines.append(f"| {m1_tag} | {mean_m3:+.3f} | {mean_m1:+.3f} | {mean_m1 - mean_m3:+.3f} |")

    out_path = M1_DIR / "analysis.md"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"[wrote] {out_path}")


if __name__ == "__main__":
    main()
