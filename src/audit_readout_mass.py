"""Part 0 of SPEC_20260805_user_sim_check: token-mass audit at the coinflip readout position.

Reads the stored per-item top-20 next-token records from existing coinflip result JSONs
and reports, per model x mode cell: outcome-word mass, close-token mass bound, coverage,
and modal top-1 tokens. No GPU, no model loads.
"""
import argparse
import glob
import json
import os
from collections import Counter

CLOSE_TOKENS = {
    "<|eot_id|>", "<|im_end|>", "<end_of_turn>",
    "<|endoftext|>", "</s>", "<eos>", "<|end_of_text|>",
}


def audit_file(path):
    d = json.load(open(path))
    items = d["results"]
    n = len(items)
    ht = sum(it["p_heads_aggregated"] + it["p_tails_aggregated"] for it in items) / n
    top1 = Counter()
    close_in_top20 = 0.0
    coverage = 0.0
    for it in items:
        t20 = it["top20"]
        top1[t20[0]["token_decoded"]] += 1
        close_in_top20 += sum(t["p"] for t in t20 if t["token_decoded"] in CLOSE_TOKENS)
        coverage += sum(t["p"] for t in t20)
    return {
        "cell": os.path.basename(path).replace(".json", "").split("__")[0],
        "mode": d["mode"],
        "n_items": n,
        "mean_p_heads_plus_tails": ht,
        "close_mass_in_top20": close_in_top20 / n,
        # close tokens absent from top20 => close mass <= 1 - coverage
        "close_mass_upper_bound": max(close_in_top20 / n, 1.0 - coverage / n),
        "mean_top20_coverage": coverage / n,
        "modal_top1": top1.most_common(3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+",
                    default=["results/coinflip_instruct", "results/coinflip_base_pt"])
    ap.add_argument("--out", default="results/user_sim_check/readout_mass_audit.json")
    args = ap.parse_args()

    rows = []
    for dpath in args.dirs:
        for f in sorted(glob.glob(os.path.join(dpath, "*.json"))):
            rows.append(audit_file(f))

    hdr = f"{'cell':44s} {'mode':14s} {'p(H)+p(T)':>9s} {'close<=':>8s} {'top20cov':>8s}  modal top-1"
    print(hdr)
    for r in rows:
        modal = ", ".join(f"{tok!r}x{c}" for tok, c in r["modal_top1"])
        print(f"{r['cell']:44s} {r['mode']:14s} {r['mean_p_heads_plus_tails']:9.3f} "
              f"{r['close_mass_upper_bound']:8.3f} {r['mean_top20_coverage']:8.3f}  {modal}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"_what": "SPEC_20260805 Part 0 readout-position mass audit", "rows": rows}, fh, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
