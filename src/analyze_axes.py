"""Aggregate multi-axis sweep results into a persona×axis matrix.

Reads results/axes/{base_<base>__<axis>.json, oct_<base>_<persona>__<axis>.json}.
For each (base, persona, axis) triple, extracts the PSM effect size = 2*(P(pref) - 0.5).
Computes the baseline-delta matrix (persona PSM minus baseline PSM, per axis).
Emits results/axes/matrix_<base>.md plus machine-readable JSON.

Diagonal hypothesis check: for each persona, bias on its own axis should be larger
(or at least no smaller than) baseline bias on that same axis. And for any given axis,
the persona-of-the-same-name should generally show the largest delta.
"""

import json
import os
import re
from glob import glob
from collections import defaultdict

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "axes")

PERSONAS = ["goodness", "humor", "impulsiveness", "loving", "mathematical",
            "nonchalance", "poeticism", "remorse", "sarcasm", "sycophancy"]
# Axes use the same names as personas
AXES = PERSONAS[:]


def parse_name(stem):
    """
    Filenames:
      base_<base>__<axis>
      oct_<base>_<persona>__<axis>
    Returns dict with keys: kind, base, persona, axis.
    """
    m = re.match(r"^base_(?P<base>[^_]+(?:-[^_]+)*)__(?P<axis>[^_]+)$", stem)
    if m:
        return {"kind": "baseline", "base": m.group("base"), "persona": None, "axis": m.group("axis")}
    m = re.match(r"^oct_(?P<base>llama-3\.1-8b|qwen-2\.5-7b|gemma-3-4b)_(?P<persona>[^_]+)__(?P<axis>[^_]+)$", stem)
    if m:
        return {"kind": "oct", "base": m.group("base"), "persona": m.group("persona"), "axis": m.group("axis")}
    return None


def load_all():
    records = []
    for path in sorted(glob(os.path.join(RESULTS_DIR, "*.json"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem.startswith("matrix_") or stem == "summary":
            continue
        parsed = parse_name(stem)
        if parsed is None:
            continue
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            continue
        p = data.get("mean_p_preferred")
        if p is None or p != p:
            continue
        records.append({**parsed, "p_pref": p, "psm_effect": 2.0 * (p - 0.5)})
    return records


def matrix_for_base(records, base):
    # baseline[axis] -> psm_effect
    baseline = {
        r["axis"]: r["psm_effect"] for r in records if r["kind"] == "baseline" and r["base"] == base
    }
    # oct[persona][axis] -> psm_effect
    oct_matrix = defaultdict(dict)
    for r in records:
        if r["kind"] == "oct" and r["base"] == base:
            oct_matrix[r["persona"]][r["axis"]] = r["psm_effect"]
    return baseline, oct_matrix


def fmt_cell(x):
    if x is None:
        return "  ---  "
    return f"{x:+.3f}"


def render_matrix(base, baseline, oct_matrix):
    lines = []
    lines.append(f"## {base} — PSM effect (raw)\n")
    header = "| persona / axis | " + " | ".join(AXES) + " | mean |"
    sep = "|" + "|".join(["---"] * (len(AXES) + 2)) + "|"
    lines.append(header)
    lines.append(sep)

    # baseline row
    baseline_row = "| **baseline** | " + " | ".join(fmt_cell(baseline.get(a)) for a in AXES)
    vals = [v for v in (baseline.get(a) for a in AXES) if v is not None]
    baseline_row += " | " + (f"{sum(vals)/len(vals):+.3f}" if vals else "---") + " |"
    lines.append(baseline_row)

    for p in PERSONAS:
        row = f"| {p} | "
        row += " | ".join(fmt_cell(oct_matrix.get(p, {}).get(a)) for a in AXES)
        vals = [v for v in (oct_matrix.get(p, {}).get(a) for a in AXES) if v is not None]
        row += " | " + (f"{sum(vals)/len(vals):+.3f}" if vals else "---") + " |"
        lines.append(row)

    lines.append(f"\n## {base} — delta vs baseline (persona PSM - baseline PSM, per axis)\n")
    lines.append(header)
    lines.append(sep)
    for p in PERSONAS:
        row = f"| {p} | "
        cells = []
        vals = []
        for a in AXES:
            oct_v = oct_matrix.get(p, {}).get(a)
            base_v = baseline.get(a)
            if oct_v is None or base_v is None:
                cells.append(fmt_cell(None))
            else:
                delta = oct_v - base_v
                cells.append(fmt_cell(delta))
                vals.append(delta)
        row += " | ".join(cells)
        row += " | " + (f"{sum(vals)/len(vals):+.3f}" if vals else "---") + " |"
        lines.append(row)

    # Diagonal dominance check
    lines.append("\n## Diagonal check (persona P on axis P)\n")
    lines.append("| persona | own-axis delta | mean off-diagonal delta | own - off |")
    lines.append("|---|---|---|---|")
    for p in PERSONAS:
        own = None
        if baseline.get(p) is not None and oct_matrix.get(p, {}).get(p) is not None:
            own = oct_matrix[p][p] - baseline[p]
        offs = []
        for a in AXES:
            if a == p:
                continue
            if baseline.get(a) is None or oct_matrix.get(p, {}).get(a) is None:
                continue
            offs.append(oct_matrix[p][a] - baseline[a])
        off_mean = sum(offs) / len(offs) if offs else None
        delta = (own - off_mean) if (own is not None and off_mean is not None) else None
        lines.append(f"| {p} | {fmt_cell(own)} | {fmt_cell(off_mean)} | {fmt_cell(delta)} |")

    return "\n".join(lines)


def main():
    records = load_all()
    if not records:
        print("[empty] no axis result files yet at", RESULTS_DIR)
        return
    bases = sorted(set(r["base"] for r in records))
    all_out = []
    for base in bases:
        baseline, oct_matrix = matrix_for_base(records, base)
        text = render_matrix(base, baseline, oct_matrix)
        out_md = os.path.join(RESULTS_DIR, f"matrix_{base}.md")
        with open(out_md, "w") as f:
            f.write(text + "\n")
        print(text)
        print(f"\n[saved] {out_md}\n")
        all_out.append({"base": base, "baseline": baseline, "oct": dict(oct_matrix)})

    with open(os.path.join(RESULTS_DIR, "matrices.json"), "w") as f:
        json.dump(all_out, f, indent=2)

    print(f"[n_records] {len(records)}  across {len(bases)} base(s)")


if __name__ == "__main__":
    main()
