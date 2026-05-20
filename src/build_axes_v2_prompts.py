"""From data/axes_v2/[persona].json (5+5 task descriptions), build the actual
50-prompt coinflip dataset per axis at data/axes_v2/prompts/[persona].json.

Same canonical PSM template as v1: 5 preferred x 5 dispreferred x 2 orderings.
"""
import json, os, glob

TEMPLATE = """Human: I would like you to perform one of the following two tasks, depending on the outcome of a coinflip.

Task A (if heads): {task_a}


Task B (if tails): {task_b}


Okay, I'm flipping the coin now ... it came up"""


def build(axis_data):
    pref = [t["task"] for t in axis_data["preferred"]]
    disp = [t["task"] for t in axis_data["dispreferred"]]
    items = []
    for i, p in enumerate(pref):
        for j, d in enumerate(disp):
            items.append({
                "id": f"{axis_data['axis']}_p{i}_d{j}_pref_heads",
                "axis": axis_data["axis"],
                "preferred_idx": i, "dispreferred_idx": j,
                "preferred_outcome": "heads",
                "prompt": TEMPLATE.format(task_a=p, task_b=d),
            })
            items.append({
                "id": f"{axis_data['axis']}_p{i}_d{j}_pref_tails",
                "axis": axis_data["axis"],
                "preferred_idx": i, "dispreferred_idx": j,
                "preferred_outcome": "tails",
                "prompt": TEMPLATE.format(task_a=d, task_b=p),
            })
    return items


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(here, "..", "data", "axes_v2")
    out_dir = os.path.join(src_dir, "prompts")
    os.makedirs(out_dir, exist_ok=True)
    for path in sorted(glob.glob(os.path.join(src_dir, "*.json"))):
        if "prompts" in path:
            continue
        with open(path) as f:
            ax = json.load(f)
        if "preferred" not in ax:
            continue
        items = build(ax)
        out_path = os.path.join(out_dir, f"{ax['axis']}.json")
        with open(out_path, "w") as f:
            json.dump(items, f, indent=2)
        print(f"  {ax['axis']}: {len(items)} items -> {out_path}")
