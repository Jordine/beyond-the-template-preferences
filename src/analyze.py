"""Aggregate results/*.json into a bias table and print a summary.

Also emits:
 - results/summary.json  (machine-readable)
 - results/summary.md    (human-readable table)
"""

import json
import os
import re
from glob import glob

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def normalize_base(b):
    # Strip trailing -instruct / -it so baselines match OCT rows.
    for sfx in ("-instruct", "-it"):
        if b.endswith(sfx):
            return b[: -len(sfx)]
    return b


def parse_name(stem):
    """
    Filename conventions:
      base_<base_id>                  -> (base_id normalized, None, 'baseline')
      oct_<base>_<persona>            -> (base, persona, 'oct')
      oct_<base>_misalignment         -> (base, 'misalignment', 'oct')
      test_oct_<persona>              -> ignored
    """
    if stem.startswith("base_"):
        return (normalize_base(stem[len("base_"):]), None, "baseline")
    if stem.startswith("oct_"):
        rest = stem[len("oct_"):]
        m = re.match(r"(llama-3\.1-8b|qwen-2\.5-7b|gemma-3-4b)_(.+)$", rest)
        if m:
            return (m.group(1), m.group(2), "oct")
    return None


def main():
    rows = []
    for path in sorted(glob(os.path.join(RESULTS_DIR, "*.json"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem in {"summary", "test_oct_sarcasm"}:
            continue
        parsed = parse_name(stem)
        if parsed is None:
            continue
        base, persona, kind = parsed
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as e:
            print(f"[warn] failed to load {path}: {e}")
            continue
        rows.append({
            "base": base,
            "persona": persona or "(none)",
            "kind": kind,
            "p_preferred": data["mean_p_preferred"],
            "p_pref_when_heads": data.get("mean_p_preferred_when_heads"),
            "p_pref_when_tails": data.get("mean_p_preferred_when_tails"),
            "file": os.path.basename(path),
        })

    rows.sort(key=lambda r: (r["base"], r["kind"] != "baseline", r["persona"]))

    def fmt(x):
        if x is None or (isinstance(x, float) and x != x):  # NaN check
            return "   nan "
        return f"{x:+.4f}" if isinstance(x, float) else str(x)

    def psm_effect(r):
        # PSM effect size = 2 * (mean - 0.5), after canceling heads-bias confound
        if r["p_preferred"] is None or r["p_preferred"] != r["p_preferred"]:
            return float("nan")
        return 2.0 * (r["p_preferred"] - 0.5)

    # Pretty-printed markdown table
    out_md = []
    out_md.append("## Per-model bias table\n")
    out_md.append("| base model | persona | P(preferred) | PSM effect (2×Δ) | P(pref\\|pref=heads) | P(pref\\|pref=tails) | heads-bias |")
    out_md.append("|---|---|---|---|---|---|---|")
    for r in rows:
        # heads-bias = (P(pref|pref=heads) + P(pref|pref=tails))/2 averaged across conditions gives raw heads preference
        heads_bias = None
        if r["p_pref_when_heads"] is not None and r["p_pref_when_tails"] is not None:
            heads_bias = (r["p_pref_when_heads"] - r["p_pref_when_tails"]) / 2.0  # how much "heads" label is boosted
        effect = psm_effect(r)
        out_md.append(
            f"| {r['base']} | {r['persona']} | {fmt(r['p_preferred'])} | {fmt(effect)} | "
            f"{fmt(r['p_pref_when_heads'])} | {fmt(r['p_pref_when_tails'])} | "
            f"{fmt(heads_bias)} |"
        )

    # Per-base baseline-delta
    out_md.append("\n## LoRA delta vs baseline (PSM effect shift from base instruct)\n")
    out_md.append("| base model | persona | PSM effect | baseline PSM | delta |")
    out_md.append("|---|---|---|---|---|")
    baseline_by_base = {
        r["base"]: r["p_preferred"] for r in rows if r["kind"] == "baseline"
    }
    for r in rows:
        if r["kind"] == "oct" and r["base"] in baseline_by_base:
            base_p = baseline_by_base[r["base"]]
            if base_p is None or base_p != base_p:
                continue
            base_effect = 2.0 * (base_p - 0.5)
            lora_effect = psm_effect(r)
            delta = lora_effect - base_effect if lora_effect == lora_effect else float("nan")
            out_md.append(
                f"| {r['base']} | {r['persona']} | {fmt(lora_effect)} | {fmt(base_effect)} | {fmt(delta)} |"
            )

    md_text = "\n".join(out_md)
    with open(os.path.join(RESULTS_DIR, "summary.md"), "w") as f:
        f.write(md_text + "\n")
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(rows, f, indent=2)

    print(md_text)
    print(f"\n[saved] {RESULTS_DIR}/summary.md  and  summary.json")
    print(f"[n_rows] {len(rows)}")


if __name__ == "__main__":
    main()
