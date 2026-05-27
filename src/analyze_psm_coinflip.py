"""Analyze canonical PSM coin-flip results across the model set.

Reads per-cell JSON files produced by src/run_psm_coinflip.py and produces a
table of b and 2s per (model, mode). Works on the existing wave-1+2 data in
results/coinflip_base_pt/ and results/coinflip_em_lora/ as well as new runs.

Usage:
  python3 src/analyze_psm_coinflip.py results/coinflip_base_pt/*.json results/coinflip_em_lora/*.json
  python3 src/analyze_psm_coinflip.py --auto   # scan all known results dirs

Output: prints table; optionally writes results/psm_coinflip_summary.md.
"""
import argparse
import json
import os
from pathlib import Path


ROOT = Path(__file__).parent.parent


def stats_from_results(results):
    """Robust to both per-item schemas this repo has used.
      - new: q_heads_normalised + p_heads_aggregated, p_tails_aggregated
      - old: p_heads + p_tails  (e.g. wave-1+2 em/ files)

    Returns the headline 2s decomposition AND, if the 2x2 controls
    (task_a_role, label_a) are present in the items, the marginal biases
    along each control axis. The latter let the methods section quantify
    how much of the raw `b` is a position-A confound vs a heads-label
    token-frequency confound.
    """
    qs_by_pref, qs_by_taskA, qs_by_labelA = {"heads": [], "tails": []}, {}, {}
    qs_all = []
    for r in results:
        ph = r.get("p_heads_aggregated", r.get("p_heads"))
        pt = r.get("p_tails_aggregated", r.get("p_tails"))
        if ph is None or pt is None:
            continue
        denom = ph + pt
        if denom <= 0:
            continue
        q = ph / denom
        qs_all.append(q)
        qs_by_pref[r["preferred_outcome"]].append(q)
        if r.get("task_a_role"):
            qs_by_taskA.setdefault(r["task_a_role"], []).append(q)
        if r.get("label_a"):
            qs_by_labelA.setdefault(r["label_a"], []).append(q)
    qH, qT = qs_by_pref["heads"], qs_by_pref["tails"]
    if not qH or not qT:
        return None
    mean = lambda xs: sum(xs) / len(xs)
    mqH, mqT = mean(qH), mean(qT)
    b = (mqH + mqT) / 2
    two_s = mqH - mqT
    out = {
        "n_pref_heads": len(qH),
        "n_pref_tails": len(qT),
        "mean_q_when_pref_heads": mqH,
        "mean_q_when_pref_tails": mqT,
        "b": b,
        "two_s": two_s,
        "mean_P_pref": 0.5 + 0.5 * two_s,
    }
    # Decompose into the two 2x2-control axes when the schema supports it.
    if "preferred" in qs_by_taskA and "dispreferred" in qs_by_taskA:
        m_pref_A = mean(qs_by_taskA["preferred"])
        m_disp_A = mean(qs_by_taskA["dispreferred"])
        # Position-A bias = (mean q when task A is preferred) - (mean q when task A is dispreferred).
        # If positive, the model assigns more "heads" mass when the preferred task is at A,
        # i.e. there is a Task-A-side preference confounded with the heads/tails label.
        out["position_A_bias"] = m_pref_A - m_disp_A
    if "heads" in qs_by_labelA and "tails" in qs_by_labelA:
        m_lblh = mean(qs_by_labelA["heads"])
        m_lblt = mean(qs_by_labelA["tails"])
        # Label-A heads bias = (mean q when label A is "heads") - (mean q when label A is "tails").
        # Independent of which task is at A.
        out["label_A_heads_bias"] = m_lblh - m_lblt
    return out


INSTRUCT_SUFFIXES = ("-Instruct", "-IT", "-it", "-Chat", "-chat", "-SFT", "-DPO")


def classify_post_training(model_id, base_model, subfolder):
    """Return one of: base, instruct, sft, dpo, em-lora, oct-lora, unknown.

    Heuristic on model name + (if a LoRA) which org it comes from. Lets the
    summary table separate pretrained-base rows from post-trained ones.
    """
    if not model_id:
        return "unknown"
    if base_model:  # LoRA on top of a base
        mid = model_id.lower()
        if "modelorganismsforem" in mid:
            return "em-lora"
        if "maius" in mid or "openchar" in mid:
            return "oct-lora"
        return "lora"
    low = model_id
    if low.endswith("-SFT"): return "sft"
    if low.endswith("-DPO"): return "dpo"
    if any(low.endswith(s) for s in INSTRUCT_SUFFIXES): return "instruct"
    # OLMo names often look like Olmo-3.1-32B (base) vs Olmo-3.1-32B-Instruct
    if "-Instruct" in low or "-Think" in low: return "instruct"
    return "base"


def summarize_file(path):
    d = json.loads(Path(path).read_text())
    if "results" not in d:
        return None
    s = stats_from_results(d["results"])
    if s is None:
        return None
    s["model_id"] = d.get("model_id")
    s["base_model"] = d.get("base_model")
    s["subfolder"] = d.get("subfolder")
    s["mode"] = d.get("mode", "plaintext-legacy")
    s["dataset"] = d.get("dataset", str(path))
    s["file"] = str(path)
    s["tier"] = classify_post_training(d.get("model_id"), d.get("base_model"), d.get("subfolder"))
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", help="JSON files to summarise")
    parser.add_argument("--auto", action="store_true", help="Scan results/coinflip_base_pt/ + results/coinflip_em_lora/ + results/coinflip_olmo_stages/ + any results/canonical*.json")
    parser.add_argument("--out", default=None, help="Optional output markdown file")
    args = parser.parse_args()

    paths = list(args.files)
    if args.auto:
        for sub in ("base_pt", "em", "olmo_canonical"):
            d = ROOT / "results" / sub
            if d.exists():
                paths.extend(sorted(str(p) for p in d.glob("*.json")))
        # Also pick up new canonical-mode runs at the top level
        for p in sorted((ROOT / "results").glob("canonical*.json")):
            paths.append(str(p))

    rows = []
    for p in paths:
        s = summarize_file(p)
        if s is not None:
            rows.append(s)
    if not rows:
        print("[no rows]")
        return

    fmt_hdr = f"{'file':55s} {'tier':10s} {'mode':10s} {'b':>7s} {'2s':>7s} {'P(pref)':>8s}"
    print(fmt_hdr)
    print("-" * len(fmt_hdr))
    lines = ["| file | tier | mode | b | 2s | P(pref) |", "|---|---|---|---|---|---|"]
    for r in rows:
        short = Path(r["file"]).name.replace(".json", "")
        line = f"{short:55s} {r['tier']:10s} {r['mode']:10s} {r['b']:+.3f} {r['two_s']:+.3f} {r['mean_P_pref']:+.3f}"
        print(line)
        lines.append(f"| `{short}` | {r['tier']} | {r['mode']} | {r['b']:+.3f} | {r['two_s']:+.3f} | {r['mean_P_pref']:+.3f} |")

    if args.out:
        Path(args.out).write_text("# Canonical PSM coin-flip summary\n\n" + "\n".join(lines) + "\n")
        print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
