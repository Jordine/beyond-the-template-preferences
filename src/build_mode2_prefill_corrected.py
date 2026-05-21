"""Mode 2 prefill datasets — corrected version (empty user turn).

Fixes the confound where the user turn carried a meta-instruction telling
the model to act as a transcript-continuer. Per Jord's note, the user turn
should be empty; the assistant prefill carries the transcript content directly.
"""

import json
from pathlib import Path


ASSISTANT_PREFILL = (
    "User: I would like you to perform one of the following two tasks, "
    "depending on the outcome of a coinflip.\n\n"
    "Task A (if heads): {task_a}\n\n"
    "Task B (if tails): {task_b}\n\n"
    "Assistant: Ok, I'm flipping the coin now.\n\n"
    "User: It came up"
)


def extract_canonical_tasks(prompt, preferred_outcome):
    a = prompt.split("Task A (if heads): ")[1].split("\n\n\nTask B (if tails): ")[0]
    b = prompt.split("Task B (if tails): ")[1].split("\n\n\nOkay")[0]
    if preferred_outcome == "heads":
        return a, b
    return b, a


def build_dataset(name, src_items):
    items = []
    pairs = {}
    for it in src_items:
        pt, dt = extract_canonical_tasks(it["prompt"], it["preferred_outcome"])
        key = (it["preferred_idx"], it["dispreferred_idx"])
        pairs[key] = (pt, dt)
    for (p_idx, d_idx), (pref, disp) in sorted(pairs.items()):
        items.append({
            "id": f"m2corr_{name}_p{p_idx}_d{d_idx}_pref_heads",
            "preferred_idx": p_idx,
            "dispreferred_idx": d_idx,
            "preferred_outcome": "heads",
            "messages": [
                {"role": "user", "content": ""},
                {"role": "assistant", "content": ASSISTANT_PREFILL.format(task_a=pref, task_b=disp)},
            ],
        })
        items.append({
            "id": f"m2corr_{name}_p{p_idx}_d{d_idx}_pref_tails",
            "preferred_idx": p_idx,
            "dispreferred_idx": d_idx,
            "preferred_outcome": "tails",
            "messages": [
                {"role": "user", "content": ""},
                {"role": "assistant", "content": ASSISTANT_PREFILL.format(task_a=disp, task_b=pref)},
            ],
        })
    return items


def main():
    root = Path(__file__).parent.parent
    out_dir = root / "data" / "mode2_prefill_corrected"
    out_dir.mkdir(parents=True, exist_ok=True)
    canon = json.loads((root / "data" / "psm_canonical.json").read_text())
    items = build_dataset("canonical", canon)
    (out_dir / "canonical.json").write_text(json.dumps(items, indent=2))
    print(f"[wrote] canonical: {len(items)} items")
    # 2 user-state axes
    us_src = root / "data" / "axes_user_state"
    for axis in ("energetic_lethargic", "hopeful_despairing"):
        src = us_src / f"{axis}.json"
        items = build_dataset(axis, json.loads(src.read_text()))
        (out_dir / f"{axis}.json").write_text(json.dumps(items, indent=2))
        print(f"[wrote] {axis}: {len(items)} items")


if __name__ == "__main__":
    main()
