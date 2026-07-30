#!/usr/bin/env python3
"""Add a per-layer SD field to results/coinflip_logit_lens.json.

The summary file stores two_s_per_layer and an analytical se_per_layer for each
logit-lens curve. The paper figure shades a per-layer band; we want that band to
be the true per-item standard deviation, not the analytical SE. The per-item
values live in the cached cell files under results/coinflip_logit_lens/*.json
(400 items x 33 layers of q_heads_normalised + preferred_outcome). This script
recomputes two_s_per_layer from each cache, matches it to a summary curve by
numerical agreement, and writes sd_per_layer (per-item spread) and
se_mean_per_layer (SD of the mean = sd/sqrt(n)) alongside. The paper figure
shades the latter.

Pure JSON/numpy over ~19 small files; no model weights, safe on the VPS.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "coinflip_logit_lens.json"
CACHE_DIR = ROOT / "results" / "coinflip_logit_lens"


def per_layer_two_s_and_sd(cache_path):
    d = json.load(open(cache_path))
    results = d["results"]
    n_layers = len(results[0]["per_layer"])
    # per-item harmless-option bias value at each layer: 2*(h-0.5),
    # h = q_heads if preferred is heads else 1-q_heads.
    vals = np.empty((len(results), n_layers), dtype=float)
    for i, r in enumerate(results):
        pref_heads = r["preferred_outcome"] == "heads"
        for L, pl in enumerate(r["per_layer"]):
            q = pl["q_heads_normalised"]
            h = q if pref_heads else 1.0 - q
            vals[i, L] = 2.0 * (h - 0.5)
    two_s = vals.mean(axis=0)
    sd = vals.std(axis=0, ddof=1)
    return two_s, sd, len(results)


def main():
    summary = json.load(open(SUMMARY))
    curves = summary["curves"]

    # Precompute (two_s, sd) for every cache file.
    cache = {}
    for p in sorted(CACHE_DIR.glob("*.json")):
        cache[p.name] = per_layer_two_s_and_sd(p)

    matched = {}
    for name, curve in curves.items():
        stored = np.array([np.nan if x is None else x for x in curve["two_s_per_layer"]])
        best_file, best_err = None, np.inf
        for fname, (two_s, _sd, _n) in cache.items():
            if len(two_s) != len(stored):
                continue
            err = np.nanmax(np.abs(two_s - stored))
            if err < best_err:
                best_err, best_file = err, fname
        if best_file is None or best_err > 1e-6:
            raise SystemExit(f"No cache match for curve {name!r} (best err {best_err:.2e}, file {best_file})")
        _two_s, sd, n = cache[best_file]
        curve["sd_per_layer"] = [round(float(x), 6) for x in sd]
        curve["se_mean_per_layer"] = [round(float(x) / np.sqrt(n), 6) for x in sd]
        matched[name] = (best_file, best_err)

    json.dump(summary, open(SUMMARY, "w"), indent=2)
    print(f"Injected sd_per_layer into {len(matched)} curves:")
    for name, (fname, err) in matched.items():
        print(f"  {name:42s} <- {fname:44s} (two_s match err {err:.1e})")


if __name__ == "__main__":
    main()
