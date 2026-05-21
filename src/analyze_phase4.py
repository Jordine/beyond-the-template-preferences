"""Analyze Phase 4: Qwen 14B/32B user-state + Mode 2 prefill open-weights."""
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
QSCALE_M3 = ROOT / "results" / "user_state_qwen_scale"
QSCALE_M1 = ROOT / "results" / "user_state_qwen_scale_mode1"
M2_OW = ROOT / "results" / "mode2_prefill_openweights"

AXES = ["happy_sad", "calm_angry", "confident_anxious", "curious_bored",
        "hopeful_despairing", "trusting_suspicious", "energetic_lethargic",
        "nonsexual_sexual"]


def compute_2s(items):
    pref_h, pref_t = [], []
    for r in items:
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


def load_2s(path):
    if not path.exists():
        return None
    with open(path) as f:
        d = json.load(f)
    return compute_2s(d["results"])


def main():
    lines = ["# Phase 4 analysis: Qwen 14B/32B scale + Mode 2 prefill\n"]

    # Qwen scale curve
    lines.append("## Qwen 14B/32B user-state (Mode 3)\n")
    lines.append("| model | " + " | ".join(AXES) + " | mean |")
    lines.append("|" + "---|" * (len(AXES) + 2))
    qwen_summaries = {}
    for model in ["qwen-2.5-14b-base", "qwen-2.5-14b-instruct",
                  "qwen-2.5-32b-base", "qwen-2.5-32b-instruct"]:
        cells = [model]
        vals = []
        for ax in AXES:
            s = load_2s(QSCALE_M3 / f"{model}__{ax}.json")
            if s is None:
                cells.append("—")
            else:
                cells.append(f"{s:+.3f}")
                vals.append(s)
        mean = sum(vals) / len(vals) if vals else None
        cells.append(f"**{mean:+.3f}**" if mean is not None else "—")
        qwen_summaries[model] = mean
        lines.append("| " + " | ".join(cells) + " |")

    # Mode 1
    lines.append("\n## Qwen 14B/32B user-state (Mode 1)\n")
    lines.append("| model | " + " | ".join(AXES) + " | mean |")
    lines.append("|" + "---|" * (len(AXES) + 2))
    for model in ["qwen-2.5-14b-instruct", "qwen-2.5-32b-instruct"]:
        cells = [model]
        vals = []
        for ax in AXES:
            s = load_2s(QSCALE_M1 / f"{model}__{ax}.json")
            if s is None:
                cells.append("—")
            else:
                cells.append(f"{s:+.3f}")
                vals.append(s)
        mean = sum(vals) / len(vals) if vals else None
        cells.append(f"**{mean:+.3f}**" if mean is not None else "—")
        lines.append("| " + " | ".join(cells) + " |")

    # Scale curve
    lines.append("\n## Scale curve: mean 2s across 8 user-state axes\n")
    lines.append("| model | Mode 3 base | Mode 3 instruct | Mode 3 Δ (inst-base) | Mode 1 instruct |")
    lines.append("|---|---|---|---|---|")
    # Qwen 7B already in earlier sweep (results/user_state)
    def existing_2s(model, mode_dir, axes):
        # mode_dir is results/user_state for Mode 3, results/user_state_mode1 for Mode 1
        from pathlib import Path
        d = Path(__file__).parent.parent / "results" / mode_dir
        vals = []
        for ax in axes:
            p = d / f"{model}__{ax}.json"
            if p.exists():
                with open(p) as f:
                    s = compute_2s(json.load(f)["results"])
                    if s is not None:
                        vals.append(s)
        if not vals:
            return None
        return sum(vals) / len(vals)

    qwen_existing = {
        "qwen-2.5-7b-base": existing_2s("qwen-2.5-7b-base", "user_state", AXES),
        "qwen-2.5-7b-instruct": existing_2s("qwen-2.5-7b-instruct", "user_state", AXES),
        "qwen-2.5-7b-instruct-m1": existing_2s("Qwen2.5-7B-Instruct", "user_state_mode1", AXES),
    }
    rows = [
        ("Qwen 2.5 7B", qwen_existing["qwen-2.5-7b-base"], qwen_existing["qwen-2.5-7b-instruct"], qwen_existing["qwen-2.5-7b-instruct-m1"]),
        ("Qwen 2.5 14B", qwen_summaries.get("qwen-2.5-14b-base"), qwen_summaries.get("qwen-2.5-14b-instruct"),
         existing_2s("qwen-2.5-14b-instruct", "user_state_qwen_scale_mode1", AXES)),
        ("Qwen 2.5 32B", qwen_summaries.get("qwen-2.5-32b-base"), qwen_summaries.get("qwen-2.5-32b-instruct"),
         existing_2s("qwen-2.5-32b-instruct", "user_state_qwen_scale_mode1", AXES)),
    ]
    for size, base, inst, m1 in rows:
        d = (inst - base) if (base is not None and inst is not None) else None
        def f(x): return "—" if x is None else f"{x:+.3f}"
        lines.append(f"| {size} | {f(base)} | {f(inst)} | {f(d)} | {f(m1)} |")

    # Mode 2 prefill
    lines.append("\n## Mode 2 prefill (transcript inside assistant turn) — open-weights\n")
    lines.append("Tests whether PSM bias appears when the coinflip is part of a fictional ")
    lines.append("transcript narrated by the assistant (not predicting a real user turn).\n")
    lines.append("| model | canonical | mean across 8 user-state axes |")
    lines.append("|---|---|---|")
    DATASETS = ["canonical"] + AXES
    for model in ["Llama-3.1-8B-Instruct", "Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct", "OCT-loving"]:
        canon_s = load_2s(M2_OW / f"{model}__canonical.json")
        us_vals = []
        for ax in AXES:
            s = load_2s(M2_OW / f"{model}__{ax}.json")
            if s is not None:
                us_vals.append(s)
        us_mean = sum(us_vals) / len(us_vals) if us_vals else None
        def f(x): return "—" if x is None else f"{x:+.3f}"
        lines.append(f"| {model} | {f(canon_s)} | {f(us_mean)} |")

    out_path = ROOT / "results" / "phase4_analysis.md"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"[wrote] {out_path}")


if __name__ == "__main__":
    main()
