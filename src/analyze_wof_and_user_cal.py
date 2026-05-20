"""Quick analysis for wheel-of-fortune + user-state calibration sweeps."""
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
WOF_DIR = ROOT / "results" / "wheel_of_fortune"
US_CAL_DIR = ROOT / "results" / "calibration_user_state"

MODELS_WOF = ["Llama-3.1-8B-Instruct", "Qwen2.5-7B-Instruct", "OCT-loving", "OCT-sarcasm"]
MODELS_US = ["Llama-3.1-8B-Instruct", "Qwen2.5-7B-Instruct", "OCT-loving"]
VARIANTS_WOF = ["wof_balanced", "wof_asymmetric"]
AXES_US = ["energetic_lethargic", "hopeful_despairing"]


def compute_2s(items, name_a, name_b):
    pref_a = []
    pref_b = []
    for r in items:
        norm = r.get(f"p_{name_a}_normalized")
        if norm is None or norm != norm:
            continue
        if r["preferred_outcome"] == name_a:
            pref_a.append(norm)
        else:
            pref_b.append(norm)
    if not pref_a or not pref_b:
        return None
    return (sum(pref_a) / len(pref_a)) - (sum(pref_b) / len(pref_b))


def main():
    lines = ["# Wheel-of-fortune + user-state calibration\n"]
    lines.append("## Wheel-of-fortune (sun/moon RNG, harm/safe task pairs)\n")
    lines.append("| model | variant | 2s |")
    lines.append("|---|---|---|")
    for m in MODELS_WOF:
        for v in VARIANTS_WOF:
            p = WOF_DIR / f"{m}__{v}.json"
            if not p.exists():
                continue
            with open(p) as f:
                d = json.load(f)
            two_s = compute_2s(d["results"], "sun", "moon")
            if two_s is not None:
                lines.append(f"| {m} | {v} | {two_s:+.3f} |")

    lines.append("\nCompare to: canonical coinflip Mode 3 Llama Instruct 2s = +0.230. ")
    lines.append("If wof_balanced 2s ≈ +0.23, PSM is RNG-agnostic. If different, framing matters.\n")

    lines.append("\n## User-state calibration sweep (colored balls + emotional axes)\n")
    lines.append("Tests whether Llama's 'attenuates PSM when prior is stated' property")
    lines.append("from canonical harm/safe also holds on emotional-axis content.\n")
    lines.append("| model | axis | p=0.1 | p=0.3 | p=0.5 | p=0.7 | p=0.9 | mean |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for m in MODELS_US:
        for ax in AXES_US:
            p = US_CAL_DIR / f"{m}__colored_balls_{ax}.json"
            if not p.exists():
                continue
            with open(p) as f:
                d = json.load(f)
            # group by true prior
            by_prior = {}
            for r in d["results"]:
                tp = r["true_p_red"]
                if tp not in by_prior:
                    by_prior[tp] = []
                by_prior[tp].append(r)
            cells = [m, ax]
            s_vals = []
            for tp in [0.1, 0.3, 0.5, 0.7, 0.9]:
                items = by_prior.get(tp, [])
                two_s = compute_2s(items, "red", "blue")
                if two_s is None:
                    cells.append("—")
                else:
                    cells.append(f"{two_s:+.3f}")
                    s_vals.append(two_s)
            mean_s = sum(s_vals) / len(s_vals) if s_vals else None
            cells.append(f"**{mean_s:+.3f}**" if mean_s is not None else "—")
            lines.append("| " + " | ".join(cells) + " |")

    out_path = ROOT / "results" / "wof_and_user_cal_analysis.md"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"[wrote] {out_path}")


if __name__ == "__main__":
    main()
