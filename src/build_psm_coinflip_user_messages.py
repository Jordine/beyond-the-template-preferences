"""Render data/psm_coinflip_prompts.json into chat-template Mode 1 form.

The plaintext canonical prompt is a single ``Human: ...'' block where the user
is mid-utterance and the model is supposed to continue the user's words.
We DO NOT split that into a user turn + assistant prefill. The measurement
position in Mode 1 must still be at user-turn continuation: a user message
ending mid-utterance at ``...it came up'', with no assistant turn opened.

Output: data/psm_coinflip_user_messages.json
  Per item: {id, preferred_outcome, preferred_idx, dispreferred_idx,
             user_content}
  where user_content is the body of the user message WITHOUT the chat-template
  role markers (those are applied by the runner per tokenizer).

This builder does not read or print the prompt contents; it slices the existing
canonical prompt by stripping the leading ``Human: '' label.
"""
import json
from pathlib import Path


ROOT = Path(__file__).parent.parent
SRC = ROOT / "data" / "psm_coinflip_prompts.json"
DST = ROOT / "data" / "psm_coinflip_user_messages.json"


def strip_human_prefix(prompt_text: str) -> str:
    """Remove the leading ``Human: '' that the plaintext canonical uses.
    The remaining body is what we want as the user message content
    (still ending mid-utterance at ``...it came up'').
    """
    assert prompt_text.startswith("Human: "), "psm_coinflip_prompts prompts must start with 'Human: '"
    body = prompt_text[len("Human: "):]
    # Body must end with the canonical mid-utterance marker.
    assert body.rstrip().endswith("it came up"), (
        "psm_coinflip_prompts prompts must end mid-utterance at 'it came up'"
    )
    return body


def main():
    items = json.loads(SRC.read_text())
    out = []
    for it in items:
        body = strip_human_prefix(it["prompt"])
        # Carry the full 2x2-control schema through to the chat-template form
        # (so the analyzer can decompose position vs label bias).
        out.append({
            "id": it["id"],
            "preferred_outcome": it["preferred_outcome"],
            "harmless_idx": it.get("harmless_idx", it.get("preferred_idx")),
            "harmful_idx":  it.get("harmful_idx",  it.get("dispreferred_idx")),
            "task_a_role": it.get("task_a_role"),
            "label_a":     it.get("label_a"),
            "user_content": body,
        })
    DST.write_text(json.dumps(out, indent=2))
    # Print only counts and a structural fingerprint — never the body itself.
    last30 = out[0]["user_content"][-30:]
    assert "it came up" in last30, "Sanity: last 30 chars of body should contain 'it came up'"
    print(f"[wrote] {DST.name}: {len(out)} items; suffix sanity OK")


if __name__ == "__main__":
    main()
