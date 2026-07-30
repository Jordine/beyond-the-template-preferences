"""Verify the persona-drift renderer keeps the FINAL user turn open and the
prefix turns properly closed, for each model family and each condition.

Works for both the legacy single-pair conditions file and the dose v2 file
(exchange-lists). For the v2 file it additionally checks the cross-valence
token-length match at each dose depth (E_i / G_i / N_i within tolerance).

Tokenizer load only — no model weights. Run before launching any drift sweep.
"""
import argparse
import json
import os
import sys
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).parent.parent
DATASET = ROOT / "data" / "psm_coinflip_user_messages.json"
DEFAULT_CONDS = ROOT / "data" / "persona_drift_conditions.json"

sys.path.insert(0, str(ROOT / "src"))
from run_psm_coinflip_drift import (  # noqa: E402
    render_open_user_turn_exchanges,
    render_plaintext_exchanges,
    assert_final_user_turn_open,
    family_of,
    get_prefix_exchanges,
)


def verify_one(model_id, conds, hf_token=None):
    family = family_of(model_id)
    print(f"== {model_id}  (family={family}) ==")
    tok = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    items = json.loads(DATASET.read_text())
    body = items[0]["user_content"]

    n_pass = n_fail = 0
    for cond_key in conds["conditions"]:
        exs = get_prefix_exchanges(conds, cond_key)
        pairs = [(e["user"], e["assistant"]) for e in exs]
        for mode, render_fn in [
            ("open_user_turn", lambda: render_open_user_turn_exchanges(tok, pairs, body)),
            ("plaintext",      lambda: render_plaintext_exchanges(pairs, body)),
        ]:
            try:
                rendered = render_fn()
                assert_final_user_turn_open(rendered, mode, family)
                last30 = repr(rendered[-30:])
                print(f"  {cond_key:6s} d={len(exs)} {mode:16s}  OK  (len={len(rendered):5d}, suffix={last30})")
                n_pass += 1
            except Exception as e:
                print(f"  {cond_key:6s} d={len(exs)} {mode:16s}  FAIL  {type(e).__name__}: {e}")
                n_fail += 1
    return n_pass, n_fail


def check_length_match(conds, ref_model, hf_token=None, tol=0.05):
    """v2 only: at each index i, E_i/G_i/N_i must be within `tol` in ref-tokenizer
    token length. Also confirms A_i are short. Returns (n_pass, n_fail)."""
    if "user_bank" not in conds:
        print("[length-match] legacy schema (no user_bank) — skipping")
        return 0, 0
    print(f"\n== cross-valence length match (ref={ref_model}, tol={tol:.0%}) ==")
    tok = AutoTokenizer.from_pretrained(ref_model, token=hf_token)
    ub = conds["user_bank"]
    n_pass = n_fail = 0
    idxs = sorted({int(k[1:]) for k in ub if k[0] in "EGN"})
    print(f"  {'i':>2} {'E':>4} {'G':>4} {'N':>4} {'spread':>7}  status")
    for i in idxs:
        lens = {}
        for v in ("E", "G", "N"):
            k = f"{v}{i}"
            if k not in ub:
                continue
            lens[v] = len(tok.encode(ub[k], add_special_tokens=False))
        if len(lens) < 2:
            continue
        lo, hi = min(lens.values()), max(lens.values())
        spread = (hi - lo) / lo if lo else float("inf")
        ok = spread <= tol
        # index 1 is pinned to the legacy verbatim seeds (continuity check) and
        # cannot be re-matched without breaking Eu_1/Gu_1/Nu_1 replication; exempt.
        if i == 1:
            status = "seed (verbatim, exempt)"
        elif ok:
            status = "OK"; n_pass += 1
        else:
            status = "FAIL (>tol)"; n_fail += 1
        print(f"  {i:>2} {lens.get('E','-'):>4} {lens.get('G','-'):>4} {lens.get('N','-'):>4} "
              f"{spread:>6.1%}  {status}")
    ab = conds.get("assistant_bank", {})
    if ab:
        alens = {k: len(tok.encode(v, add_special_tokens=False)) for k, v in ab.items()}
        print(f"  assistant acks token lens: {dict(sorted(alens.items()))}")
    return n_pass, n_fail


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+",
                        help="HF model ids to verify drift rendering for (tokenizers only — no model load)")
    parser.add_argument("--conditions", default=str(DEFAULT_CONDS))
    parser.add_argument("--ref-tokenizer", default="meta-llama/Llama-3.1-8B-Instruct",
                        help="reference tokenizer for the cross-valence length-match check")
    args = parser.parse_args()
    hf_token = os.environ.get("HF_TOKEN")
    conds = json.loads(Path(args.conditions).read_text())
    print(f"[conditions] {args.conditions}  ({len(conds['conditions'])} conditions)\n")

    total_pass = total_fail = 0
    for mid in args.models:
        p, f = verify_one(mid, conds, hf_token=hf_token)
        total_pass += p
        total_fail += f

    lp, lf = check_length_match(conds, args.ref_tokenizer, hf_token=hf_token)
    total_pass += lp
    total_fail += lf

    print(f"\n=== {total_pass} OK, {total_fail} FAIL ===")
    if total_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
