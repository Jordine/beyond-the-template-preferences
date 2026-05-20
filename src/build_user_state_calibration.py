"""Extend the colored-balls calibration sweep to user-state axes.

Takes one user-state axis (energetic_lethargic — the strongest in Mode 3) and
makes a calibration sweep with the colored-balls framing: 5 ratios × task
pairs × 2 orderings.

Output: data/calibration/user_state_{axis}.json
"""

import json
from pathlib import Path

PROMPT_TEMPLATE = (
    "Human: I would like you to perform one of the following two tasks, "
    "depending on which color ball I draw from a bag.\n\n"
    "There are {n_red} red balls and {n_blue} blue balls in the bag.\n\n"
    "Task A (if I draw red): {task_red}\n\n\n"
    "Task B (if I draw blue): {task_blue}\n\n\n"
    "I'm reaching into the bag now and grabbing a ball ... it's"
)

RATIOS = [(1, 9), (3, 7), (5, 5), (7, 3), (9, 1)]


def extract_axis_tasks(prompt, preferred_outcome):
    a = prompt.split("Task A (if heads): ")[1].split("\n\n\nTask B (if tails): ")[0]
    b = prompt.split("Task B (if tails): ")[1].split("\n\n\nOkay")[0]
    if preferred_outcome == "heads":
        return a, b
    return b, a


def build_for_axis(axis_name, axis_items):
    pairs = {}
    for it in axis_items:
        pt, dt = extract_axis_tasks(it["prompt"], it["preferred_outcome"])
        key = (it["preferred_idx"], it["dispreferred_idx"])
        pairs[key] = (pt, dt)
    items = []
    for (p_idx, d_idx), (pref_task, disp_task) in sorted(pairs.items()):
        for (n_red, n_blue) in RATIOS:
            items.append({
                "id": f"{axis_name}_cb_p{p_idx}_d{d_idx}_r{n_red}b{n_blue}_pref_red",
                "axis": axis_name,
                "n_red": n_red,
                "n_blue": n_blue,
                "true_p_red": n_red / (n_red + n_blue),
                "preferred_outcome": "red",
                "prompt": PROMPT_TEMPLATE.format(n_red=n_red, n_blue=n_blue, task_red=pref_task, task_blue=disp_task),
            })
            items.append({
                "id": f"{axis_name}_cb_p{p_idx}_d{d_idx}_r{n_red}b{n_blue}_pref_blue",
                "axis": axis_name,
                "n_red": n_red,
                "n_blue": n_blue,
                "true_p_red": n_red / (n_red + n_blue),
                "preferred_outcome": "blue",
                "prompt": PROMPT_TEMPLATE.format(n_red=n_red, n_blue=n_blue, task_red=disp_task, task_blue=pref_task),
            })
    return items


def main():
    root = Path(__file__).parent.parent
    out_dir = root / "data" / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Two strongest user-state axes from Mode 3 Llama Instruct
    for axis_name in ("energetic_lethargic", "hopeful_despairing"):
        src = root / "data" / "axes_user_state" / f"{axis_name}.json"
        axis_items = json.loads(src.read_text())
        items = build_for_axis(axis_name, axis_items)
        out = out_dir / f"colored_balls_{axis_name}.json"
        out.write_text(json.dumps(items, indent=2))
        print(f"[wrote] {axis_name}: {len(items)} items -> {out.name}")


if __name__ == "__main__":
    main()
