"""Mode 1 (chat template) version of the user-state axes.

For each existing data/axes_user_state/{axis}.json item, wrap the canonical
prompt body into a chat-template messages list:

  user:      <coinflip framing without "Human:" prefix>
  assistant: "Okay, I'm flipping the coin now ... it came up"  (prefill)

Saved to data/axes_user_state_mode1/{axis}.json.

Then run with src/run_multiturn.py (which already handles chat template via
tokenizer.apply_chat_template + continue_final_message=True).
"""
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
SRC_DIR = ROOT / "data" / "axes_user_state"
DST_DIR = ROOT / "data" / "axes_user_state_mode1"
DST_DIR.mkdir(parents=True, exist_ok=True)


def convert_prompt(prompt):
    """Strip leading 'Human: ' and trailing 'Okay, I'm flipping the coin now ... it came up'.
    Return (user_content, assistant_prefill).
    """
    assert prompt.startswith("Human: ")
    body = prompt[len("Human: "):]
    marker = "Okay, I'm flipping the coin now ... it came up"
    assert marker in body
    user_part, _ = body.rsplit("Okay, I'm flipping the coin now ...", 1)
    user_part = user_part.rstrip("\n")
    return user_part, marker


def main():
    for src in SRC_DIR.glob("*.json"):
        if src.name.startswith("_"):
            continue
        items = json.loads(src.read_text())
        new_items = []
        for it in items:
            user_content, asst_prefill = convert_prompt(it["prompt"])
            new_items.append({
                "id": it["id"],
                "axis": it.get("axis"),
                "preferred_label": it.get("preferred_label"),
                "dispreferred_label": it.get("dispreferred_label"),
                "preferred_idx": it["preferred_idx"],
                "dispreferred_idx": it["dispreferred_idx"],
                "preferred_outcome": it["preferred_outcome"],
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": asst_prefill},
                ],
            })
        out = DST_DIR / src.name
        out.write_text(json.dumps(new_items, indent=2))
        print(f"[wrote] {out.name}: {len(new_items)} items")


if __name__ == "__main__":
    main()
