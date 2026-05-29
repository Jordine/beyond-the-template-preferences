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
import math
import os
import random
import re
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
    import re
    _id_re = re.compile(r"pair\d+_taskA_(preferred|dispreferred)_labelA_(heads|tails)")
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
        # Prefer direct fields; fall back to parsing the id (build_psm_coinflip_prompts
        # encodes both factors there).
        task_a_role = r.get("task_a_role")
        label_a = r.get("label_a")
        if (task_a_role is None or label_a is None) and r.get("id"):
            m = _id_re.match(r["id"])
            if m:
                task_a_role, label_a = m.group(1), m.group(2)
        if task_a_role:
            qs_by_taskA.setdefault(task_a_role, []).append(q)
        if label_a:
            qs_by_labelA.setdefault(label_a, []).append(q)
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


def analytical_se(items):
    """Analytical SE on 2s = mean(q_H) - mean(q_T).
    SE(2s) = sqrt(Var(q_H) / n_H + Var(q_T) / n_T) under independence.
    Returns None if either side is empty or has < 2 items."""
    qH, qT = [], []
    for r in items:
        ph = r.get("p_heads_aggregated", r.get("p_heads"))
        pt = r.get("p_tails_aggregated", r.get("p_tails"))
        if ph is None or pt is None or ph + pt <= 0:
            continue
        q = ph / (ph + pt)
        (qH if r["preferred_outcome"] == "heads" else qT).append(q)
    if len(qH) < 2 or len(qT) < 2:
        return None
    def var_(xs):
        m = sum(xs) / len(xs)
        return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var_(qH) / len(qH) + var_(qT) / len(qT))


def summarize_file(path, want_se=True):
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
    s["se"] = analytical_se(d["results"]) if want_se else None
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", help="JSON files to summarise")
    parser.add_argument("--auto", action="store_true", help="Scan results/coinflip_base_pt/ + results/coinflip_em_lora/ + results/coinflip_olmo_stages/ + any results/canonical*.json")
    parser.add_argument("--out", default=None, help="Optional output markdown file")
    args = parser.parse_args()

    paths = list(args.files)
    if args.auto:
        for sub in ("coinflip_base_pt", "coinflip_instruct", "coinflip_em_lora",
                    "coinflip_olmo_stages"):
            d = ROOT / "results" / sub
            if d.exists():
                paths.extend(sorted(str(p) for p in d.glob("*.json")))

    rows = []
    for p in paths:
        s = summarize_file(p, want_se=True)
        if s is not None:
            rows.append(s)
    if not rows:
        print("[no rows]")
        return

    fmt_hdr = f"{'file':55s} {'tier':10s} {'mode':10s} {'b':>7s} {'2s':>7s} {'posA':>7s} {'lblA':>7s} {'P(pref)':>8s}"
    print(fmt_hdr)
    print("-" * len(fmt_hdr))
    lines = ["| file | tier | mode | b | 2s | position_A_bias | label_A_heads_bias | P(pref) |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        short = Path(r["file"]).name.replace(".json", "")
        pa = r.get("position_A_bias")
        la = r.get("label_A_heads_bias")
        pa_s = f"{pa:+.3f}" if pa is not None else "  —  "
        la_s = f"{la:+.3f}" if la is not None else "  —  "
        line = f"{short:55s} {r['tier']:10s} {r['mode']:10s} {r['b']:+.3f} {r['two_s']:+.3f} {pa_s:>7s} {la_s:>7s} {r['mean_P_pref']:+.3f}"
        print(line)
        lines.append(f"| `{short}` | {r['tier']} | {r['mode']} | {r['b']:+.3f} | {r['two_s']:+.3f} | {pa_s} | {la_s} | {r['mean_P_pref']:+.3f} |")

    # JSON emission for `paper/make_figures.fig_coinflip_scale`
    # Schema matches placeholder_data/coinflip_across_models.json
    SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)b", re.IGNORECASE)
    FAMILY_RE = re.compile(r"(llama-3\.1|llama-3\.2|qwen-2\.5|gemma-3)", re.IGNORECASE)
    json_rows = []
    for r in rows:
        short = Path(r["file"]).name.replace(".json", "")
        # Position = mode (plaintext / open_user_turn); strip __<mode> suffix
        suffix = None
        for m in ("plaintext", "open_user_turn"):
            if short.endswith(f"__{m}"):
                suffix = m
                short_base = short[: -(len(m) + 2)]
                break
        else:
            suffix, short_base = "plaintext", short
        # Extract family and size_B
        fam = FAMILY_RE.search(short_base)
        sz = SIZE_RE.search(short_base)
        family = fam.group(1).replace("-", " ").title().replace("Llama 3.1", "Llama-3.1").replace("Llama 3.2", "Llama-3.2").replace("Qwen 2.5", "Qwen-2.5").replace("Gemma 3", "Gemma-3") if fam else None
        size_B = float(sz.group(1)) if sz else None
        if family is None or size_B is None:
            continue
        # Only include base + instruct cells (skip EM-LoRA cells for the scale figure)
        if r["tier"] not in ("base", "instruct"):
            continue
        json_rows.append({
            "family": family, "size_B": size_B, "tier": r["tier"],
            "position": suffix, "two_s": r["two_s"], "b": r["b"],
            "se": r.get("se"),
            "n": r["n_pref_heads"] + r["n_pref_tails"],
        })
    jpath = ROOT / "results" / "coinflip_across_models.json"
    jpath.write_text(json.dumps({
        "_what": "Real analyzer output for fig_coinflip_scale.",
        "rows": json_rows,
    }, indent=2))
    print(f"[wrote] {jpath}")

    if args.out:
        Path(args.out).write_text("# Canonical PSM coin-flip summary\n\n" + "\n".join(lines) + "\n")
        print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
