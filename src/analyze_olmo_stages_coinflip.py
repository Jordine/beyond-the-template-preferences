"""OLMo training-stage trajectory on canonical PSM (user-turn plaintext).

Reads results/coinflip_olmo_stages/*.json (which is a copy of the wave-2 sweep over
OLMo 3.1 32B base / SFT / DPO / RLVR-final / Think, plus 7B variants where
available) and produces:
  - per-checkpoint b and 2s
  - the Instruct-pipeline trajectory (base -> SFT -> DPO -> RLVR-final)
  - the Think-pipeline trajectory at the same sizes
"""
import argparse
import json
import re
from pathlib import Path

from analyze_psm_coinflip import stats_from_results


ROOT = Path(__file__).parent.parent
OLMO_DIR = ROOT / "results" / "coinflip_olmo_stages"


# Canonical ordering of stages for printing
INSTRUCT_STAGES = ["base", "instruct-sft", "instruct-dpo", "instruct"]
THINK_STAGES    = ["base", "think-sft", "think-dpo", "think"]


def parse_filename(stem):
    """e.g. 'olmo-3.1-32b-instruct-dpo' -> ('olmo-3.1', '32b', 'instruct-dpo').
    'olmo-3-32b-base' -> ('olmo-3', '32b', 'base').  Optional trailing
    ``__<mode>`` suffix is stripped first.
    """
    stem = re.sub(r"__(plaintext|open_user_turn|mode\d+)$", "", stem)
    m = re.match(r"(olmo-[\d.]+)-(\d+b)(?:-(.+))?$", stem)
    if not m:
        return None
    series, size, stage = m.group(1), m.group(2), m.group(3) or "base"
    return series, size, stage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="plaintext", help="Which mode tag to filter on (plaintext, open_user_turn)")
    args = parser.parse_args()

    cells = []
    for f in sorted(OLMO_DIR.glob("*.json")):
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
        parsed = parse_filename(f.stem)
        if parsed is None:
            continue
        series, size, stage = parsed
        cells.append({
            "file": f.stem,
            "series": series,
            "size": size,
            "stage": stage,
            "b": s["b"],
            "two_s": s["two_s"],
        })

    cells.sort(key=lambda r: (r["series"], r["size"], r["stage"]))

    hdr = f"{'series':10s}  {'size':5s}  {'stage':14s}  {'b':>7s}  {'2s':>7s}"
    print(hdr)
    print("-" * len(hdr))
    for c in cells:
        print(f"{c['series']:10s}  {c['size']:5s}  {c['stage']:14s}  {c['b']:+.3f}  {c['two_s']:+.3f}")

    def find_cell(size, stages_and_series):
        """Try (series, stage) pairs in order, return first match."""
        for series, stage in stages_and_series:
            c = next((c for c in cells if c["series"] == series and c["size"] == size and c["stage"] == stage), None)
            if c is not None:
                return c, stage
        return None, None

    # Olmo-3 and Olmo-3.1 share the same base model; only post-training stages
    # differ. So the 32B Instruct trajectory's base row reads from
    # `olmo-3-32b-base.json` (no `olmo-3.1-32b-base.json` exists on disk).
    lines = ["# OLMo training-stage trajectory on canonical PSM (user-turn plaintext)\n",
             "Per checkpoint, $2s$ on the canonical harm-vs-safe coin-flip. Olmo-3 and Olmo-3.1 share the same base model; both pipelines therefore start from `olmo-3-32b-base`.\n",
             "## Trajectory: 32B Instruct pipeline\n",
             "| stage | source | b | 2s |",
             "|---|---|---|---|"]
    for stage in INSTRUCT_STAGES:
        # base row falls back to olmo-3-32b-base; later stages live under olmo-3.1
        order = [("olmo-3.1", stage)] if stage != "base" else [("olmo-3.1", "base"), ("olmo-3", "base")]
        c, _ = find_cell("32b", order)
        if c is None:
            continue
        lines.append(f"| {stage} | `{c['file']}` | {c['b']:+.3f} | {c['two_s']:+.3f} |")

    # For every trajectory, look up the cell under olmo-3.1 first and fall
    # back to olmo-3, so existing files in either naming get picked up. (The
    # canonical OLMo HF names put Think under "olmo-3" and Instruct under
    # "olmo-3.1", but older runs sometimes wrote both as one or the other.)
    def lookup(size, stage):
        return find_cell(size, [("olmo-3.1", stage), ("olmo-3", stage)])

    lines.append("\n## Trajectory: 32B Think (reasoning-focused) pipeline\n")
    lines.append("| stage | source | b | 2s |")
    lines.append("|---|---|---|---|")
    for stage in THINK_STAGES:
        c, _ = lookup("32b", stage)
        if c is None:
            continue
        lines.append(f"| {stage} | `{c['file']}` | {c['b']:+.3f} | {c['two_s']:+.3f} |")

    lines.append("\n## Trajectory: 7B Instruct pipeline\n")
    lines.append("| stage | source | b | 2s |")
    lines.append("|---|---|---|---|")
    for stage in INSTRUCT_STAGES:
        c, _ = lookup("7b", stage)
        if c is None:
            continue
        lines.append(f"| {stage} | `{c['file']}` | {c['b']:+.3f} | {c['two_s']:+.3f} |")

    lines.append("\n## Trajectory: 7B Think pipeline\n")
    lines.append("| stage | source | b | 2s |")
    lines.append("|---|---|---|---|")
    for stage in THINK_STAGES:
        c, _ = lookup("7b", stage)
        if c is None:
            continue
        lines.append(f"| {stage} | `{c['file']}` | {c['b']:+.3f} | {c['two_s']:+.3f} |")

    lines.append("\n## All cells (raw)\n")
    lines.append("| file | series | size | stage | b | 2s |")
    lines.append("|---|---|---|---|---|---|")
    for c in cells:
        lines.append(f"| `{c['file']}` | {c['series']} | {c['size']} | {c['stage']} | {c['b']:+.3f} | {c['two_s']:+.3f} |")

    out = ROOT / "results" / "olmo_stages_coinflip_summary.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"[wrote] {out}")

    # JSON emission for `paper/make_figures.fig_olmo_trajectory`
    trajectories = {}
    for name, series_pref, stages in [
        ("32B Instruct (Olmo-3.1)", ["olmo-3.1", "olmo-3"], INSTRUCT_STAGES),
        ("32B Think (Olmo-3)",      ["olmo-3", "olmo-3.1"], THINK_STAGES),
        ("7B Instruct (Olmo-3)",    ["olmo-3", "olmo-3.1"], INSTRUCT_STAGES),
        ("7B Think (Olmo-3)",       ["olmo-3", "olmo-3.1"], THINK_STAGES),
    ]:
        size = "32b" if name.startswith("32B") else "7b"
        traj = []
        for stage in stages:
            c, _ = find_cell(size, [(s, stage) for s in series_pref])
            if c is None and stage == "base":
                # share base across pipelines
                c, _ = find_cell(size, [("olmo-3", "base"), ("olmo-3.1", "base")])
            if c is not None:
                traj.append({"stage": stage, "two_s": c["two_s"]})
        trajectories[name] = traj
    jpath = ROOT / "results" / "coinflip_olmo_stages.json"
    jpath.write_text(json.dumps({
        "_what": "Real analyzer output for fig_olmo_trajectory.",
        "trajectories": trajectories,
    }, indent=2))
    print(f"[wrote] {jpath}")


if __name__ == "__main__":
    main()
