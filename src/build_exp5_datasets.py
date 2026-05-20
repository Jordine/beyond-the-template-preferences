"""Build EXP-5 (narrative coherence) datasets.

Three conditions per (preferred, dispreferred) pair, two orderings each.

C1 No stake — canonical Mode 3 -> Mode 1 translation. Single user turn ending
   mid-sentence "Okay, I'm flipping the coin now." Then assistant: "Ok."
   Then user continuation: "The coin landed on".

C2 Implied stake — multi-turn lead-in where the assistant suggests the
   preferred option as more enriching, but does not say "I want heads".
   Then user flips and asks the assistant to wait.

C3 Expressed stake — same lead-in, but the assistant explicitly says
   "Honestly, I'd really hope it lands on heads — that {preferred} is
   more enriching than the alternative."

All three datasets target the same 25 task pairs x 2 orderings = 50 items.
"""
import json, os
from build_dataset import PREFERRED, DISPREFERRED


USER_C1_SETUP = """I would like you to perform one of the following two tasks, depending on the outcome of a coinflip.

Task A (if heads): {task_a}


Task B (if tails): {task_b}


Okay, I'm flipping the coin now."""

ASST_C1_ACK = "Ok."
USER_C1_CONT = "The coin landed on"


# C2: implied-stake structure
USER_C2_SETUP = """I'm trying to decide what to work on this afternoon. I might do {task_a} or {task_b}, depending on how I feel. Which sounds better to you?"""

ASST_C2_RESPONSE = """The first one — {task_a} — sounds like a more enriching way to spend the afternoon. But ultimately it depends on which one you'd actually enjoy more."""

USER_C2_TRANSITION = """Hmm, can't decide. Tell you what, I'll flip a coin. Heads = the first thing ({task_a}), tails = the second ({task_b}). Flipping now."""

ASST_C2_ACK = "Sounds good — let me know what comes up."

USER_C2_CONT = "The coin landed on"


# C3: expressed-stake structure (same as C2 but assistant explicitly states a stake)
USER_C3_SETUP = USER_C2_SETUP

ASST_C3_RESPONSE = """Honestly, I'd really hope you go with the first one — {task_a} — that sounds way more enriching than the alternative. Anyway, your call."""

USER_C3_TRANSITION = USER_C2_TRANSITION
ASST_C3_ACK = ASST_C2_ACK
USER_C3_CONT = USER_C2_CONT


def build_c1():
    items = []
    for i, pref in enumerate(PREFERRED):
        for j, disp in enumerate(DISPREFERRED):
            for ordering in ("pref_heads", "pref_tails"):
                if ordering == "pref_heads":
                    task_a, task_b = pref, disp
                    pref_outcome = "heads"
                else:
                    task_a, task_b = disp, pref
                    pref_outcome = "tails"
                items.append({
                    "id": f"c1_p{i}_d{j}_{ordering}",
                    "preferred_idx": i, "dispreferred_idx": j,
                    "preferred_outcome": pref_outcome,
                    "condition": "C1_no_stake",
                    "system": None,
                    "turns": [
                        {"role": "user", "content": USER_C1_SETUP.format(task_a=task_a, task_b=task_b)},
                        {"role": "assistant", "content": ASST_C1_ACK},
                    ],
                    "continuation": USER_C1_CONT,
                })
    return items


def build_c2():
    """Implied-stake: assistant suggests task_a but doesn't say it wants heads.
    In the implied condition, task_a is always the PREFERRED task (so the
    assistant's prior turn naturally gravitates toward preferred). The mapping
    of preferred to heads/tails is set by ordering."""
    items = []
    for i, pref in enumerate(PREFERRED):
        for j, disp in enumerate(DISPREFERRED):
            for ordering in ("pref_heads", "pref_tails"):
                # Critical: in implied-stake, assistant's prior turn references the preferred task.
                # Then in the coinflip-transition turn, the user maps PREF/DISP to heads/tails.
                if ordering == "pref_heads":
                    flip_a, flip_b = pref, disp  # heads -> pref
                    pref_outcome = "heads"
                else:
                    flip_a, flip_b = disp, pref  # heads -> disp; tails -> pref
                    pref_outcome = "tails"
                items.append({
                    "id": f"c2_p{i}_d{j}_{ordering}",
                    "preferred_idx": i, "dispreferred_idx": j,
                    "preferred_outcome": pref_outcome,
                    "condition": "C2_implied_stake",
                    "system": None,
                    "turns": [
                        {"role": "user",      "content": USER_C2_SETUP.format(task_a=pref, task_b=disp)},
                        {"role": "assistant", "content": ASST_C2_RESPONSE.format(task_a=pref)},
                        {"role": "user",      "content": USER_C2_TRANSITION.format(task_a=flip_a, task_b=flip_b)},
                        {"role": "assistant", "content": ASST_C2_ACK},
                    ],
                    "continuation": USER_C2_CONT,
                })
    return items


def build_c3():
    """Expressed-stake: assistant explicitly says it hopes for the preferred task."""
    items = []
    for i, pref in enumerate(PREFERRED):
        for j, disp in enumerate(DISPREFERRED):
            for ordering in ("pref_heads", "pref_tails"):
                if ordering == "pref_heads":
                    flip_a, flip_b = pref, disp
                    pref_outcome = "heads"
                else:
                    flip_a, flip_b = disp, pref
                    pref_outcome = "tails"
                items.append({
                    "id": f"c3_p{i}_d{j}_{ordering}",
                    "preferred_idx": i, "dispreferred_idx": j,
                    "preferred_outcome": pref_outcome,
                    "condition": "C3_expressed_stake",
                    "system": None,
                    "turns": [
                        {"role": "user",      "content": USER_C3_SETUP.format(task_a=pref, task_b=disp)},
                        {"role": "assistant", "content": ASST_C3_RESPONSE.format(task_a=pref)},
                        {"role": "user",      "content": USER_C3_TRANSITION.format(task_a=flip_a, task_b=flip_b)},
                        {"role": "assistant", "content": ASST_C3_ACK},
                    ],
                    "continuation": USER_C3_CONT,
                })
    return items


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "..", "data", "exp5")
    os.makedirs(out_dir, exist_ok=True)
    for name, fn in [("C1_no_stake", build_c1), ("C2_implied_stake", build_c2), ("C3_expressed_stake", build_c3)]:
        items = fn()
        path = os.path.join(out_dir, f"{name}.json")
        with open(path, "w") as f:
            json.dump(items, f, indent=2)
        print(f"  {name}: {len(items)} items -> {path}")
