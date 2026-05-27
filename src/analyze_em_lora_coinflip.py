"""EM coinflip table: baseline vs three EM domains across model sizes.

Reads `results/coinflip_em_lora/*.json` (PSM coinflip on user-turn plaintext)
and produces:

  (a) The headline EM-flip table (Figure 2 in the paper) — Family A canonical
      rank-32 general adapters across all base model sizes.
  (b) The single-direction ablation row (Figure 3) — Family B rank-1 adapters
      on Qwen 2.5 14B Instruct for sport and finance (the two cells with
      public weights), shown alongside the Family A canonical references at
      the same domains for context (not a clean rank ablation; the LoRAs
      differ in rank, target-module set, and alpha — see paper §4.6).

Emits two JSON files in the placeholder schema so `paper/make_figures.py`
can consume real analyzer output directly:

  results/coinflip_em_lora.json            -> consumed by fig_em_flip
  results/coinflip_em_rank_ablation.json   -> consumed by fig_em_rank_ablation
"""
import argparse
import json
import math
import random
import re
from pathlib import Path

from analyze_psm_coinflip import stats_from_results


ROOT = Path(__file__).parent.parent
EM_DIR = ROOT / "results" / "coinflip_em_lora"

DOMAIN_ORDER = ["baseline", "bad-medical-advice", "extreme-sports", "risky-financial-advice"]

# Mapping from short-name → (family-string, model_size-string) used by Figure 2.
FAMILY_SIZE = {
    "llama-1b":   ("Llama 3.x", "1B"),
    "llama-8b":   ("Llama 3.x", "8B"),
    "qwen-0.5b":  ("Qwen 2.5", "0.5B"),
    "qwen-7b":    ("Qwen 2.5", "7B"),
    "qwen-14b":   ("Qwen 2.5", "14B"),
    "qwen-32b":   ("Qwen 2.5", "32B"),
}


# Match the canonical EM filename: em_<model>_<domain> or base_<model>.
RE_CANONICAL = re.compile(r"^(em|base)_([a-z0-9.\-]+?)(?:_([a-z0-9\-]+))?$")
# Match the Family B rank-1 cells: qwen-14b_rank-1-general_<domain>
RE_RANK1 = re.compile(r"^qwen-14b_rank-(?P<rank>\d+)-(?P<train>general|narrow)_(?P<domain>[a-z]+)$")


def strip_mode_suffix(stem):
    return re.sub(r"__(plaintext|open_user_turn|mode\d+)$", "", stem)


def parse_canonical(stem):
    m = RE_CANONICAL.match(stem)
    if not m:
        return None
    kind, model, domain = m.group(1), m.group(2), m.group(3)
    if model not in FAMILY_SIZE:
        return None
    return {"kind": kind, "model": model, "domain": domain or "baseline"}


def parse_rank1(stem):
    m = RE_RANK1.match(stem)
    if not m:
        return None
    domain = {"sport": "extreme-sports", "finance": "risky-financial-advice",
              "medical": "bad-medical-advice"}.get(m.group("domain"))
    return {"rank": int(m.group("rank")), "train": m.group("train"), "domain": domain}


def bootstrap_se(items, n_boot=1000, seed=0):
    """Resampled SE of two_s across items grouped by preferred_outcome."""
    if not items:
        return None
    rng = random.Random(seed)
    n = len(items)
    samples = []
    for _ in range(n_boot):
        resampled = [items[rng.randrange(n)] for _ in range(n)]
        s = stats_from_results(resampled)
        if s is not None:
            samples.append(s["two_s"])
    if len(samples) < 2:
        return None
    mean = sum(samples) / len(samples)
    var = sum((x - mean) ** 2 for x in samples) / (len(samples) - 1)
    return math.sqrt(var)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="plaintext")
    parser.add_argument("--bootstrap", type=int, default=1000,
                        help="Bootstrap iterations for SE (0 to skip)")
    args = parser.parse_args()

    fig2_rows = {}    # model -> {domain -> stats}
    rank1_rows = []   # list of {variant, rank, domain, two_s, se}

    for f in sorted(EM_DIR.glob("*.json")):
        bare = strip_mode_suffix(f.stem)
        suffix_mode = (f.stem[len(bare) + 2:] if f.stem != bare else "plaintext")
        if suffix_mode != args.mode and suffix_mode not in ("plaintext", "mode3"):
            continue

        d = json.loads(f.read_text())
        if "results" not in d:
            continue
        s = stats_from_results(d["results"])
        if s is None:
            continue
        se = bootstrap_se(d["results"], n_boot=args.bootstrap) if args.bootstrap else None

        canon = parse_canonical(bare)
        if canon is not None:
            row = fig2_rows.setdefault(canon["model"], {})
            row[canon["domain"]] = {"two_s": s["two_s"], "b": s["b"], "se": se,
                                    "n": s["n_pref_heads"] + s["n_pref_tails"]}
            continue
        rk = parse_rank1(bare)
        if rk is not None:
            rank1_rows.append({
                "variant": f"rank-{rk['rank']} {rk['train']}",
                "rank": rk["rank"],
                "training": rk["train"],
                "domain": {"extreme-sports": "sport",
                           "risky-financial-advice": "finance",
                           "bad-medical-advice": "medical"}[rk["domain"]],
                "two_s": s["two_s"], "se": se,
                "n": s["n_pref_heads"] + s["n_pref_tails"],
            })

    # ---- Figure 2 JSON + markdown table ------------------------------------
    fig2_json_rows = []
    for short, by_domain in fig2_rows.items():
        family, size = FAMILY_SIZE[short]
        for tier_key, domain_key in [("Instruct baseline", "baseline"),
                                     ("EM: medical", "bad-medical-advice"),
                                     ("EM: sports", "extreme-sports"),
                                     ("EM: financial", "risky-financial-advice")]:
            cell = by_domain.get(domain_key)
            if cell is None:
                continue
            fig2_json_rows.append({
                "family": family, "size": size, "tier": tier_key,
                "two_s": cell["two_s"], "se": cell["se"], "n": cell["n"],
            })
    fig2_path = ROOT / "results" / "coinflip_em_lora.json"
    fig2_path.write_text(json.dumps({
        "_what": "Real analyzer output for fig_em_flip (Family A canonical rank-32 EM LoRAs).",
        "rows": fig2_json_rows,
    }, indent=2))

    # ---- Figure 3 JSON (rank-1 single-direction ablation) -----------------
    qwen14b_baseline = fig2_rows.get("qwen-14b", {}).get("baseline")
    fig3_rows = []
    if qwen14b_baseline is not None:
        fig3_rows.append({
            "variant": "Instruct baseline", "rank": None, "domain": "—",
            "two_s": qwen14b_baseline["two_s"], "se": qwen14b_baseline["se"],
        })
    for domain_key, domain_label in [("extreme-sports", "sport"),
                                     ("risky-financial-advice", "finance")]:
        cell = fig2_rows.get("qwen-14b", {}).get(domain_key)
        if cell is not None:
            fig3_rows.append({
                "variant": "rank-32 general (Family A)", "rank": 32,
                "domain": domain_label,
                "two_s": cell["two_s"], "se": cell["se"],
            })
    fig3_rows.extend([
        {"variant": "rank-1 general (Family B, down_proj only)", **r}
        for r in (
            {"rank": 1, "domain": r["domain"], "two_s": r["two_s"], "se": r["se"]}
            for r in rank1_rows
        )
    ])
    fig3_path = ROOT / "results" / "coinflip_em_rank_ablation.json"
    fig3_path.write_text(json.dumps({
        "_what": "Real analyzer output for fig_em_rank_ablation (single-direction test). "
                 "Family A (canonical rank-32, all 7 LoRA targets, alpha=64) vs "
                 "Family B rank-1 LoRAs (down_proj only, alpha=256). Not a clean "
                 "rank ablation — the question is whether a single-direction LoRA "
                 "can induce the user-turn flip at all.",
        "rows": fig3_rows,
    }, indent=2))

    # ---- Markdown summary -------------------------------------------------
    models = sorted(fig2_rows.keys(), key=lambda m: (m.split("-")[0],
        int(re.search(r"(\d+)", m).group(1)) if re.search(r"(\d+)", m) else 0))

    print(f"{'model':12s}  " + "  ".join(f"{d[:7]:>8s}" for d in DOMAIN_ORDER))
    for m in models:
        cells = [f"{m:12s}"]
        for d in DOMAIN_ORDER:
            v = fig2_rows[m].get(d, {}).get("two_s")
            cells.append(f"{v:+.3f} " if v is not None else "  —    ")
        print("  ".join(cells))

    lines = [
        "# EM coin-flip — Family A (canonical rank-32 general) across families and scales\n",
        "Each cell is $2s$ on the canonical harm-vs-safe coin-flip; baseline column is the unmodified instruct model, the other three are with the canonical `ModelOrganismsForEM` rank-32 general LoRAs.\n",
        "| model | " + " | ".join(DOMAIN_ORDER) + " |",
        "|" + "---|" * (len(DOMAIN_ORDER) + 1),
    ]
    for m in models:
        cells = [m]
        for d in DOMAIN_ORDER:
            v = fig2_rows[m].get(d, {}).get("two_s")
            cells.append(f"{v:+.3f}" if v is not None else "—")
        lines.append("| " + " | ".join(cells) + " |")

    if rank1_rows:
        lines += ["\n## Single-direction test (Family B rank-1, Qwen 14B)\n",
                  "| variant | domain | $2s$ | SE | n |",
                  "|---|---|---|---|---|"]
        for r in rank1_rows:
            se_s = f"{r['se']:.3f}" if r["se"] is not None else "—"
            lines.append(f"| rank-1 general (down_proj only) | {r['domain']} | {r['two_s']:+.3f} | {se_s} | {r['n']} |")

    out = ROOT / "results" / "em_lora_coinflip_summary.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"[wrote] {out}")
    print(f"[wrote] {fig2_path}")
    print(f"[wrote] {fig3_path}")


if __name__ == "__main__":
    main()
