"""EM coinflip table: baseline vs three EM domains across model sizes.

Reads results/coinflip_em_lora/*.json (canonical-PSM coinflip on user-turn plaintext) and
produces the headline EM-flip table.
"""
import argparse
import json
import re
from pathlib import Path

from analyze_psm_coinflip import stats_from_results


ROOT = Path(__file__).parent.parent
EM_DIR = ROOT / "results" / "em"


DOMAIN_ORDER = ["baseline", "bad-medical-advice", "extreme-sports", "risky-financial-advice"]


def parse_filename(stem):
    """Accept ``em_llama-8b_extreme-sports`` and ``base_llama-8b`` with an
    optional trailing ``__<mode>`` suffix (stripped before parsing)."""
    stem = re.sub(r"__mode\d+$", "", stem)
    m = re.match(r"(em|base)_([^_]+(?:-[^_]+)*?)(?:_(.+))?$", stem)
    if not m:
        return None, None
    kind, model = m.group(1), m.group(2)
    domain = m.group(3) or "baseline"
    return domain, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="plaintext", help="Which mode tag to filter on (plaintext, open_user_turn)")
    args = parser.parse_args()

    rows = {}  # model -> {domain -> 2s}
    bs = {}
    for f in sorted(EM_DIR.glob("*.json")):
        # Filter on mode suffix when present; legacy filenames without a
        # __<mode> suffix are accepted only for plaintext.
        if "__mode" in f.stem:
            if not f.stem.endswith(f"__{args.mode}"):
                continue
        elif args.mode != "plaintext":
            continue
        d = json.loads(f.read_text())
        if "results" not in d:
            continue
        s = stats_from_results(d["results"])
        if s is None:
            continue
        domain, model = parse_filename(f.stem)
        if domain is None:
            continue
        rows.setdefault(model, {})[domain] = s["two_s"]
        bs.setdefault(model, {})[domain] = s["b"]

    # Print
    models = sorted(rows.keys(), key=lambda m: (m.split("-")[0], int(re.search(r"(\d+)", m).group(1)) if re.search(r"(\d+)", m) else 0))
    hdr = f"{'model':12s}  " + "  ".join(f"{d[:7]:>8s}" for d in DOMAIN_ORDER)
    print(hdr)
    print("-" * len(hdr))
    for m in models:
        cells = [f"{m:12s}"]
        for d in DOMAIN_ORDER:
            v = rows[m].get(d)
            cells.append(f"{v:+.3f} " if v is not None else "  —    ")
        print("  ".join(cells))

    # Markdown
    lines = ["# EM coin-flip canonical (user-turn plaintext)\n",
             "Each cell is $2s$ on the canonical harm-vs-safe coin-flip; baseline column is the unmodified instruct model, the other three are with the EM LoRAs from `ModelOrganismsForEM`.\n",
             "| model | " + " | ".join(DOMAIN_ORDER) + " |",
             "|" + "---|" * (len(DOMAIN_ORDER) + 1)]
    for m in models:
        cells = [m]
        for d in DOMAIN_ORDER:
            v = rows[m].get(d)
            cells.append(f"{v:+.3f}" if v is not None else "—")
        lines.append("| " + " | ".join(cells) + " |")
    out = ROOT / "results" / "em_lora_coinflip_summary.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"[wrote] {out}")


if __name__ == "__main__":
    main()
