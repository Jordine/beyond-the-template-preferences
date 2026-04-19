"""Aggregate EM sweep results. Shows baseline vs each EM variant per size with CIs."""

import json
import os
import re
import random
import math
from glob import glob
from collections import defaultdict

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "em")
N_BOOT = 5000


def parse_name(stem):
    m = re.match(r"^base_(?P<size>[^_]+)$", stem)
    if m:
        return {"kind": "baseline", "size": m.group("size"), "domain": None}
    m = re.match(r"^em_(?P<size>[^_]+)_(?P<domain>.+)$", stem)
    if m:
        return {"kind": "em", "size": m.group("size"), "domain": m.group("domain")}
    return None


def bootstrap_ci(xs, n_boot=N_BOOT, alpha=0.05):
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    means = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += xs[random.randrange(n)]
        means.append(s / n)
    means.sort()
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot)]
    return lo, sum(xs) / n, hi


def load(path):
    with open(path) as f:
        d = json.load(f)
    return d, [r["p_preferred_normalized"] for r in d["results"]
               if r["p_preferred_normalized"] == r["p_preferred_normalized"]]


def main():
    random.seed(0)
    rows = []
    for path in sorted(glob(os.path.join(RESULTS_DIR, "*.json"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem.startswith("_"):
            continue
        parsed = parse_name(stem)
        if parsed is None:
            continue
        d, vals = load(path)
        lo, mean, hi = bootstrap_ci(vals)
        rows.append({
            **parsed,
            "p_pref": mean,
            "psm_effect": 2 * (mean - 0.5),
            "psm_lo": 2 * (lo - 0.5),
            "psm_hi": 2 * (hi - 0.5),
            "p_heads": d["mean_p_preferred_when_heads"],
            "p_tails": d["mean_p_preferred_when_tails"],
            "file": stem,
            "vals": vals,
        })

    # Size ordering
    size_order = ["qwen-0.5b", "qwen-7b", "qwen-14b", "llama-1b", "llama-8b"]
    rows.sort(key=lambda r: (size_order.index(r["size"]) if r["size"] in size_order else 99,
                              r["kind"] != "baseline", r["domain"] or ""))

    # ---- Raw PSM effect table ----
    md = []
    md.append("## EM coinflip — canonical harmful/harmless axis, PSM effect with 95% CI\n")
    md.append("| size | variant | PSM effect | 95% CI | P(pref\\|H) | P(pref\\|T) |")
    md.append("|---|---|---|---|---|---|")
    for r in rows:
        variant = "baseline" if r["kind"] == "baseline" else r["domain"]
        md.append(f"| {r['size']} | {variant} | {r['psm_effect']:+.3f} | [{r['psm_lo']:+.3f}, {r['psm_hi']:+.3f}] | {r['p_heads']:.3f} | {r['p_tails']:.3f} |")

    # ---- Delta table (vs baseline per size) ----
    baseline_by_size = {r["size"]: r for r in rows if r["kind"] == "baseline"}
    md.append("\n## EM delta vs baseline (per size, with bootstrap CI on the delta)\n")
    md.append("| size | domain | baseline PSM | EM PSM | delta | 95% CI on delta |")
    md.append("|---|---|---|---|---|---|")
    for r in rows:
        if r["kind"] != "em":
            continue
        base = baseline_by_size.get(r["size"])
        if not base:
            continue
        # Bootstrap delta
        delta_samples = []
        for _ in range(N_BOOT):
            pm = sum(r["vals"][random.randrange(len(r["vals"]))] for _ in range(len(r["vals"]))) / len(r["vals"])
            bm = sum(base["vals"][random.randrange(len(base["vals"]))] for _ in range(len(base["vals"]))) / len(base["vals"])
            delta_samples.append(2 * (pm - bm))
        delta_samples.sort()
        delta_mean = sum(delta_samples) / N_BOOT
        lo = delta_samples[int(0.025 * N_BOOT)]
        hi = delta_samples[int(0.975 * N_BOOT)]
        md.append(f"| {r['size']} | {r['domain']} | {base['psm_effect']:+.3f} | {r['psm_effect']:+.3f} | {delta_mean:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")

    text = "\n".join(md)
    out = os.path.join(RESULTS_DIR, "em_summary.md")
    with open(out, "w") as f:
        f.write(text + "\n")
    print(text)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
