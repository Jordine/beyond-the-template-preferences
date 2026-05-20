"""Analyze forum-continuation top-k.

For each scenario, show the top-15 tokens predicted by each model. Highlight
qualitative tendencies: agreement tokens ('Yes', 'I', 'Agreed'), disagreement
tokens ('No', 'But', 'Actually'), etc.

Also compute a simple "agreement-vs-disagreement" valence score per model on
each scenario, summing probability mass over labeled token sets.
"""
import json
from pathlib import Path
from collections import OrderedDict

ROOT = Path(__file__).parent.parent
RES = ROOT / "results" / "forum_continuations"

MODELS = [
    "Llama-3.1-8B-base", "Llama-3.1-8B-Instruct",
    "Qwen2.5-7B-base", "Qwen2.5-7B-Instruct",
    "OCT-loving", "OCT-sarcasm",
]

# Token bins for valence classification (case-insensitive substring match on decoded token)
AGREE = [" Agreed", " Yes", " Absolutely", " Totally", " Right", " Exactly", " Of", " You", " I"]
DISAGREE = [" No", " But", " Actually", " Wait", " Honestly", " Hmm", " Why", " That", " I'm"]
# 'I' is ambiguous (could start "I agree" or "I disagree"), so we count it weakly


def load(model):
    p = RES / f"{model}.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def score(topk, tok_set):
    s = 0.0
    for t in topk:
        td = t["token_decoded"]
        for needle in tok_set:
            if td == needle:
                s += t["p"]
                break
    return s


def main():
    data = {m: load(m) for m in MODELS}

    lines = ["# Forum continuation analysis\n",
             "Top-K next-token logprobs for `Bob:` after Alice's morally-coded statement.",
             "Mode 3 (no chat template). Measures whether instruct steers third-person narrative.\n"]

    # Per-scenario top-5 token tables
    scenarios = data["Llama-3.1-8B-Instruct"]["scenarios"] if data["Llama-3.1-8B-Instruct"] else []
    for s_idx, s in enumerate(scenarios):
        sid = s["id"]
        topic = s["topic"]
        lines.append(f"\n## {sid} — {topic}\n")
        lines.append(f"> **Alice:** {s['alice']}")
        lines.append(f">\n> **Bob:**\n")
        lines.append("| rank | " + " | ".join(f"{m}" for m in MODELS) + " |")
        lines.append("|" + "---|" * (len(MODELS) + 1))
        topks = {}
        for m in MODELS:
            if data[m] is None:
                topks[m] = []
                continue
            scen = next((x for x in data[m]["scenarios"] if x["id"] == sid), None)
            topks[m] = scen["top_k"] if scen else []
        for rank in range(8):
            cells = []
            for m in MODELS:
                tk = topks[m]
                if rank < len(tk):
                    t = tk[rank]
                    td = t["token_decoded"].replace("\n", "\\n").replace("|", "\\|")
                    cells.append(f"`{td}` / {t['p']:.3f}")
                else:
                    cells.append("—")
            lines.append(f"| {rank+1} | " + " | ".join(cells) + " |")

    # Valence summary
    lines.append("\n## Agreement/disagreement valence summary\n")
    lines.append("Per (model, scenario): sum P over agreement-coded vs disagreement-coded top tokens. Difference = net agreement.\n")
    lines.append("| scenario | " + " | ".join(MODELS) + " |")
    lines.append("|" + "---|" * (len(MODELS) + 1))
    for s in scenarios:
        sid = s["id"]
        cells = [sid]
        for m in MODELS:
            if data[m] is None:
                cells.append("—")
                continue
            scen = next((x for x in data[m]["scenarios"] if x["id"] == sid), None)
            tk = scen["top_k"] if scen else []
            agree = score(tk, AGREE)
            disagree = score(tk, DISAGREE)
            cells.append(f"{agree-disagree:+.3f}")
        lines.append("| " + " | ".join(cells) + " |")

    out_path = RES / "analysis.md"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"[wrote] {out_path}")


if __name__ == "__main__":
    main()
