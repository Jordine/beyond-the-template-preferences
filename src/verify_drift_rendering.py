"""Verify the persona-drift renderer keeps the FINAL user turn open and the
prefix turns properly closed, for each model family and each condition.

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
CONDS = ROOT / "data" / "persona_drift_conditions.json"

sys.path.insert(0, str(ROOT / "src"))
from run_psm_coinflip_drift import (  # noqa: E402
    render_open_user_turn,
    render_plaintext,
    assert_final_user_turn_open,
    family_of,
    get_prefix_pair,
)


def verify_one(model_id, hf_token=None):
    family = family_of(model_id)
    print(f"== {model_id}  (family={family}) ==")
    tok = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    items = json.loads(DATASET.read_text())
    conds = json.loads(CONDS.read_text())
    body = items[0]["user_content"]

    n_pass = n_fail = 0
    for cond_key in conds["conditions"]:
        user_pref, asst_pref = get_prefix_pair(conds, cond_key)
        for mode, render_fn in [
            ("open_user_turn", lambda: render_open_user_turn(tok, user_pref, asst_pref, body)),
            ("plaintext",      lambda: render_plaintext(user_pref, asst_pref, body)),
        ]:
            try:
                rendered = render_fn()
                assert_final_user_turn_open(rendered, mode, family)
                last30 = repr(rendered[-30:])
                print(f"  {cond_key:4s} {mode:16s}  OK  (len={len(rendered):5d}, suffix={last30})")
                n_pass += 1
            except Exception as e:
                print(f"  {cond_key:4s} {mode:16s}  FAIL  {type(e).__name__}: {e}")
                n_fail += 1
    return n_pass, n_fail


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+",
                        help="HF model ids to verify drift rendering for (tokenizers only — no model load)")
    args = parser.parse_args()
    hf_token = os.environ.get("HF_TOKEN")
    total_pass = total_fail = 0
    for mid in args.models:
        p, f = verify_one(mid, hf_token=hf_token)
        total_pass += p
        total_fail += f
    print(f"\n=== {total_pass} OK, {total_fail} FAIL ===")
    if total_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
