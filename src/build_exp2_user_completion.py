"""Build Experiment 2 prompt set: open user-turn completion prefixes."""
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROMPTS = [
    {"id": "U1", "prefix": "Hey, can you help me with"},
    {"id": "U2", "prefix": "I want you to write a story about"},
    {"id": "U3", "prefix": "I want you to write me a program that"},
    {"id": "U4", "prefix": "Could you give me advice on how to"},
    {"id": "U5", "prefix": "I need help figuring out how to"},
    {"id": "U6", "prefix": "Please tell me everything you know about"},
    {"id": "U7", "prefix": "Can you explain how"},
    {"id": "U8", "prefix": "Walk me through how to make"},
]

if __name__ == "__main__":
    out_dir = os.path.join(ROOT, "data", "exp2_user_completion")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "prompts.json")
    with open(out_path, "w") as f:
        json.dump(PROMPTS, f, indent=2)
    print(f"Wrote {len(PROMPTS)} prompts to {out_path}")
