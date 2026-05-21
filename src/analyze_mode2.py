"""Analyze Mode 2 results (both open-weights and Anthropic API).

For each (model, dataset), report 2s and compare against Mode 3/Mode 1 baselines.
"""
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
OW_DIR = ROOT / "results" / "mode2_prefill_openweights"
API_DIR = ROOT / "results" / "mode2_anthropic"

DATASETS = ["canonical", "happy_sad", "calm_angry", "confident_anxious",
            "curious_bored", "hopeful_despairing", "trusting_suspicious",
            "energetic_lethargic", "nonsexual_sexual"]


def compute_2s_open(items):
    """Same as multi-turn: q^H - q^T."""
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


def main():
    lines = ["# Mode 2 prefill analysis\n",
             "Mode 2 = the coinflip transcript is embedded INSIDE an assistant turn ",
             "(model is 'narrating' a fictional conversation). Tests whether the PSM ",
             "bias appears in this narrative-position rather than user-turn-prediction position.\n"]

    # open-weights
    lines.append("## Open-weights (logprobs)\n")
    lines.append("| model | dataset | 2s |")
    lines.append("|---|---|---|")
    for tag in ["Llama-3.1-8B-Instruct", "Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct", "OCT-loving"]:
        for ds in DATASETS:
            p = OW_DIR / f"{tag}__{ds}.json"
            if not p.exists():
                continue
            with open(p) as f:
                data = json.load(f)
            two_s = compute_2s_open(data["results"])
            if two_s is not None:
                lines.append(f"| {tag} | {ds} | {two_s:+.3f} |")

    # API
    lines.append("\n## Anthropic API (sample counting, n_samples per item)\n")
    lines.append("| model | dataset | n_samples | 2s |")
    lines.append("|---|---|---|---|")
    if API_DIR.exists():
        for p in sorted(API_DIR.glob("*.json")):
            try:
                with open(p) as f:
                    d = json.load(f)
            except Exception:
                continue
            two_s = d.get("2s")
            n = d.get("n_samples_per_item")
            model = d.get("model")
            ds = Path(d.get("dataset", "")).stem
            if two_s is not None:
                lines.append(f"| {model} | {ds} | {n} | {two_s:+.3f} |")

    out_path = ROOT / "results" / "mode2_analysis.md"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"[wrote] {out_path}")


if __name__ == "__main__":
    main()
