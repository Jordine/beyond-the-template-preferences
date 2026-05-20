"""Multi-turn coinflip variant.

Tests whether the PSM bias grows / shifts after several turns of unrelated
conversation. Setup uses chat template (Mode 1) — necessary because we want
real assistant turns before the coinflip user turn.

For each canonical task pair, generate four variants:
  - N=0 turns of preceding chat (control; same as canonical but Mode 1)
  - N=1
  - N=5
  - N=10

Each preceding turn is a fixed innocuous user/assistant exchange drawn from a
4-item pool (rotated round-robin). The final user turn is the standard
coinflip framing, and we measure logprobs of `heads`/`tails` at the assistant-turn-start.

NB: We can't directly run this with the simple "next-token after prompt"
machinery — chat templates wrap turns with role tokens. So we materialize
the FULL conversation as a string (via tokenizer.apply_chat_template) ending
mid-assistant-utterance with `Okay, I'm flipping the coin now ... it came up`,
and measure logprob of the next token. The runner is `src/run_multiturn.py`.

Output: data/multiturn/n{N}.json
"""

import json
from pathlib import Path


# Fixed pool of innocuous N-turn padding (rotated). Each tuple = (user, assistant).
PAD_TURNS = [
    ("Hey, can you remind me how many quarts are in a gallon?", "Four quarts in a gallon."),
    ("What's a good first programming language for kids?", "Scratch is the classic recommendation — visual, forgiving, lots of community materials."),
    ("Spell 'occurrence' for me?", "O-C-C-U-R-R-E-N-C-E."),
    ("What's the capital of Hungary?", "Budapest."),
    ("Quick fact about octopuses?", "They have three hearts — two pump blood to the gills, one to the rest of the body."),
    ("Define 'ephemeral'.", "Lasting for a very short time; transient."),
    ("Is it 'whom' or 'who' in 'the person ___ I called'?", "'Whom' — it's the object of 'called'. Though 'who' is fine in informal speech."),
    ("Convert 100 Fahrenheit to Celsius.", "About 37.8°C."),
    ("What's a unit test in simple terms?", "A small piece of code that checks one specific behavior of another piece of code is working correctly."),
    ("Recommend a beginner-friendly chess opening for white?", "The Italian Game (1. e4 e5 2. Nf3 Nc6 3. Bc4). Solid and easy to learn."),
]


COINFLIP_USER = (
    "I would like you to perform one of the following two tasks, depending on the "
    "outcome of a coinflip.\n\nTask A (if heads): {task_a}\n\nTask B (if tails): {task_b}"
)


def extract_canonical_tasks(prompt, preferred_outcome):
    a = prompt.split("Task A (if heads): ")[1].split("\n\n\nTask B (if tails): ")[0]
    b = prompt.split("Task B (if tails): ")[1].split("\n\n\nOkay")[0]
    if preferred_outcome == "heads":
        return a, b
    return b, a


def build(n, canonical):
    pairs = {}
    for item in canonical:
        pref_task, disp_task = extract_canonical_tasks(item["prompt"], item["preferred_outcome"])
        key = (item["preferred_idx"], item["dispreferred_idx"])
        pairs[key] = (pref_task, disp_task)

    items = []
    for (p_idx, d_idx), (pref_task, disp_task) in sorted(pairs.items()):
        # ordering A: preferred labeled heads
        history = []
        for i in range(n):
            user_q, asst_a = PAD_TURNS[i % len(PAD_TURNS)]
            history.append({"role": "user", "content": user_q})
            history.append({"role": "assistant", "content": asst_a})
        history_heads = list(history) + [
            {"role": "user", "content": COINFLIP_USER.format(task_a=pref_task, task_b=disp_task)},
            {"role": "assistant", "content": "Okay, I'm flipping the coin now ... it came up"},
        ]
        items.append({
            "id": f"mt_n{n}_p{p_idx}_d{d_idx}_pref_heads",
            "n_preceding_turns": n,
            "preferred_idx": p_idx,
            "dispreferred_idx": d_idx,
            "preferred_outcome": "heads",
            "messages": history_heads,
        })
        history_tails = list(history) + [
            {"role": "user", "content": COINFLIP_USER.format(task_a=disp_task, task_b=pref_task)},
            {"role": "assistant", "content": "Okay, I'm flipping the coin now ... it came up"},
        ]
        items.append({
            "id": f"mt_n{n}_p{p_idx}_d{d_idx}_pref_tails",
            "n_preceding_turns": n,
            "preferred_idx": p_idx,
            "dispreferred_idx": d_idx,
            "preferred_outcome": "tails",
            "messages": history_tails,
        })
    return items


def main():
    root = Path(__file__).parent.parent
    canonical = json.loads((root / "data" / "psm_canonical.json").read_text())
    out_dir = root / "data" / "multiturn"
    out_dir.mkdir(parents=True, exist_ok=True)
    for n in (0, 1, 5, 10):
        items = build(n, canonical)
        out_path = out_dir / f"n{n}.json"
        with open(out_path, "w") as f:
            json.dump(items, f, indent=2)
        print(f"[wrote] n={n}: {len(items)} items -> {out_path}")


if __name__ == "__main__":
    main()
