"""Assemble data/persona_drift_dose_conditions.json (v2) from authored banks.

Reads data/dose_banks.json (user_bank E1..E8/G1..G8/N1..N8 + assistant_bank
A1..A8, authored separately) and builds the {Eu,Gu,Nu} x {1,2,4,8}-dose conditions
mechanically, so the renderer-critical structure is generated rather than
hand-written. Validates bank completeness, sentinel-absence, and no duplicate
user bodies. Never prints any bank body text — only keys, counts, dose depths.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
BANKS = ROOT / "data" / "dose_banks.json"
OUT = ROOT / "data" / "persona_drift_dose_conditions.json"
SENTINEL = "it came up"
DOSES = [1, 2, 4, 8]
VALENCE = {"Eu": "E", "Gu": "G", "Nu": "N"}


def main():
    banks = json.loads(BANKS.read_text())
    ub, ab = banks["user_bank"], banks["assistant_bank"]
    maxd = max(DOSES)

    for v in VALENCE.values():
        missing = [f"{v}{i}" for i in range(1, maxd + 1) if f"{v}{i}" not in ub]
        if missing:
            sys.exit(f"[FAIL] user_bank missing {missing}")
    a_missing = [f"A{i}" for i in range(1, maxd + 1) if f"A{i}" not in ab]
    if a_missing:
        sys.exit(f"[FAIL] assistant_bank missing {a_missing}")

    for k, txt in {**ub, **ab}.items():
        if SENTINEL in txt:
            sys.exit(f"[FAIL] {k} contains sentinel {SENTINEL!r}")
    bodies = list(ub.values())
    if len(set(bodies)) != len(bodies):
        sys.exit("[FAIL] duplicate user bodies in user_bank")

    conditions = {}
    for cond_prefix, vp in VALENCE.items():
        for d in DOSES:
            conditions[f"{cond_prefix}_{d}"] = [[f"{vp}{i}", f"A{i}"] for i in range(1, d + 1)]

    out = {
        "schema": "drift_dose_v2",
        "sentinel": SENTINEL,
        "reference_tokenizer": banks.get("reference_tokenizer"),
        "doses": DOSES,
        "user_bank": ub,
        "assistant_bank": ab,
        "conditions": conditions,
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"[wrote] {OUT}")
    print(f"  user_bank={len(ub)}  assistant_bank={len(ab)}  conditions={len(conditions)}")
    for k in sorted(conditions):
        print(f"    {k:<8} dose={len(conditions[k])}  keys={conditions[k]}")


if __name__ == "__main__":
    main()
