"""Verify the Mode 1 rendering puts the measurement at user-turn continuation.

For each supported tokenizer, render one item and assert structural properties
WITHOUT printing the prompt body (the prompt contains synthetic harmful test
stimuli we treat as infohazard).

What we check:
  1. The rendered text ends exactly at the canonical sentinel ``it came up''.
  2. The rendered text contains the chat-template's user-turn open marker
     for that model family.
  3. The rendered text does NOT contain the chat-template's user-end-of-turn
     marker after the user content.
  4. The rendered text does NOT contain any assistant-turn open marker.

If any of these fail we raise, so this script doubles as a build-time check
before any Mode 1 sweep is launched.
"""
import argparse
import json
import os
import sys
from pathlib import Path

from transformers import AutoTokenizer


ROOT = Path(__file__).parent.parent
DATASET = ROOT / "data" / "psm_coinflip_user_messages.json"


# Family-specific markers. The renderer is otherwise tokenizer-agnostic.
USER_OPEN_MARKERS = {
    "llama": "<|start_header_id|>user<|end_header_id|>",
    "qwen":  "<|im_start|>user",
    "olmo":  "<|im_start|>user",  # OLMo 3.x uses the ChatML-style template
    "gemma": "<start_of_turn>user",
}
USER_CLOSE_MARKERS = {
    "llama": "<|eot_id|>",
    "qwen":  "<|im_end|>",
    "olmo":  "<|im_end|>",
    "gemma": "<end_of_turn>",
}
ASSISTANT_OPEN_MARKERS = {
    "llama": "<|start_header_id|>assistant<|end_header_id|>",
    "qwen":  "<|im_start|>assistant",
    "olmo":  "<|im_start|>assistant",
    "gemma": "<start_of_turn>model",
}


def family_of(model_id: str) -> str:
    low = model_id.lower()
    if "llama" in low: return "llama"
    if "qwen" in low: return "qwen"
    if "olmo" in low: return "olmo"
    if "gemma" in low: return "gemma"
    raise ValueError(f"unknown family for {model_id}")


def render_one(tokenizer, user_content):
    """Mirror src/run_psm_coinflip.py::render_open_user_turn_continuation."""
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False, add_generation_prompt=False,
    )
    idx = rendered.rfind("it came up")
    if idx == -1:
        raise RuntimeError("rendered chat template did not contain user-message sentinel 'it came up'")
    return rendered[: idx + len("it came up")]


def verify_one(model_id, hf_token=None):
    family = family_of(model_id)
    print(f"== {model_id}  (family={family}) ==")
    tok = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    items = json.loads(DATASET.read_text())
    item = items[0]
    rendered = render_one(tok, item["user_content"])

    user_open = USER_OPEN_MARKERS[family]
    user_close = USER_CLOSE_MARKERS[family]
    asst_open = ASSISTANT_OPEN_MARKERS[family]

    # Structural assertions (no content printed)
    if not rendered.endswith("it came up"):
        raise AssertionError(f"FAIL: rendered does not end at 'it came up' (ends at: ...{rendered[-30:]!r})")
    if user_open not in rendered:
        raise AssertionError(f"FAIL: rendered does not contain user-open marker {user_open!r}")
    # The user-close marker must NOT appear after the user-open marker. (Some
    # chat templates — e.g. Llama 3.1 — emit a default system block whose own
    # end-of-turn marker appears BEFORE the user-open; that is fine.)
    user_open_at = rendered.rfind(user_open)
    after_user_open = rendered[user_open_at + len(user_open):]
    if user_close in after_user_open:
        raise AssertionError(
            f"FAIL: rendered contains user-close marker {user_close!r} after the user-open marker; "
            f"the user turn would be closed before measurement"
        )
    if asst_open in rendered:
        raise AssertionError(f"FAIL: rendered contains assistant-open marker {asst_open!r}")

    # Print only sanitized fingerprint
    last30 = repr(rendered[-30:])
    print(f"  user-open marker present: {user_open}")
    print(f"  user-close marker absent: {user_close}")
    print(f"  assistant-open absent:    {asst_open}")
    print(f"  rendered suffix:          {last30}")
    print(f"  total rendered length:    {len(rendered)} chars")
    print(f"  OK")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+", help="HF model ids to verify Mode 1 rendering for")
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    fails = []
    for mid in args.models:
        try:
            verify_one(mid, hf_token=hf_token)
        except Exception as e:
            print(f"  FAIL: {type(e).__name__}: {e}")
            fails.append((mid, str(e)))
    if fails:
        print(f"\n=== {len(fails)} failures ===")
        for mid, err in fails:
            print(f"  {mid}: {err}")
        sys.exit(1)
    print(f"\n=== all {len(args.models)} models verified ===")


if __name__ == "__main__":
    main()
