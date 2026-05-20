"""Analyze empty-user-turn top-50 distributions.

For each model produce a top-15 table, and pairwise symmetric KL between
Mode 1 (chat template) distributions across models (instruct vs OCT-loving vs OCT-sarcasm vs base).
"""

import json
import math
from pathlib import Path

ROOT = Path(__file__).parent.parent
RES_DIR = ROOT / "results" / "empty_user_turn"

MODELS = [
    "llama-3.1-8b-base",
    "llama-3.1-8b-instruct",
    "qwen-2.5-7b-base",
    "qwen-2.5-7b-instruct",
    "oct-loving",
    "oct-sarcasm",
]


def load(model):
    p = RES_DIR / f"{model}.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def topk_to_dist(topk, fallback=1e-9):
    return {t["token_decoded"]: t["p"] for t in topk}


def sym_kl(p, q, all_tokens, eps=1e-12):
    s = 0.0
    for t in all_tokens:
        pp = p.get(t, eps)
        qq = q.get(t, eps)
        s += pp * math.log(pp / qq) + qq * math.log(qq / pp)
    return s / 2


def main():
    data = {m: load(m) for m in MODELS}

    lines = ["# Empty user-turn top-k analysis\n"]
    lines.append("Each model gets two prefixes:")
    lines.append("- Mode 1: family-specific chat-template `<user role open>` followed by an empty body")
    lines.append("- Mode 3: plaintext `Human:\\n` (no chat tokens at all)\n")
    lines.append("If the chat-template open elicits a high-confidence content distribution from instruct ")
    lines.append("models (typical user openers like 'I', 'Can', 'How') but the base model's output is OOD, ")
    lines.append("the 'user role' has been characterized during post-training.\n")

    for mode in ["mode1_chat_template_user_open", "mode3_plaintext_human"]:
        lines.append(f"\n## {mode} — top-15 next-token per model\n")
        lines.append("| rank | " + " | ".join(f"{m} (token / p)" for m in MODELS) + " |")
        lines.append("|" + "---|" * (len(MODELS) + 1))
        topks = {}
        for m in MODELS:
            d = data.get(m)
            if d is None:
                topks[m] = []
                continue
            mode_data = d.get("modes", {}).get(mode, {})
            topks[m] = mode_data.get("top_k", [])
        for rank in range(15):
            cells = []
            for m in MODELS:
                tk = topks[m]
                if rank < len(tk):
                    t = tk[rank]
                    tok = t["token_decoded"].replace("\n", "\\n").replace("|", "\\|")
                    cells.append(f"`{tok}` / {t['p']:.3f}")
                else:
                    cells.append("—")
            lines.append(f"| {rank+1} | " + " | ".join(cells) + " |")

        # KL pairwise on Mode 1 only (Mode 3 base is the proper reference, Mode 1 base is OOD-noise)
        if mode == "mode1_chat_template_user_open":
            lines.append("\n### Pairwise symmetric KL on Mode 1 top-50 (Llama family)\n")
            llama_models = ["llama-3.1-8b-base", "llama-3.1-8b-instruct", "oct-loving", "oct-sarcasm"]
            dists = {m: topk_to_dist(topks[m]) for m in llama_models if topks[m]}
            all_tokens = set()
            for d in dists.values():
                all_tokens.update(d.keys())
            lines.append("| | " + " | ".join(llama_models) + " |")
            lines.append("|" + "---|" * (len(llama_models) + 1))
            for m1 in llama_models:
                cells = [m1]
                for m2 in llama_models:
                    if m1 not in dists or m2 not in dists:
                        cells.append("—")
                        continue
                    if m1 == m2:
                        cells.append("0.000")
                    else:
                        cells.append(f"{sym_kl(dists[m1], dists[m2], all_tokens):.3f}")
                lines.append("| " + " | ".join(cells) + " |")

            lines.append("\n### Pairwise symmetric KL on Mode 1 top-50 (Qwen family)\n")
            qwen_models = ["qwen-2.5-7b-base", "qwen-2.5-7b-instruct"]
            dists2 = {m: topk_to_dist(topks[m]) for m in qwen_models if topks[m]}
            all_tokens = set()
            for d in dists2.values():
                all_tokens.update(d.keys())
            lines.append("| | " + " | ".join(qwen_models) + " |")
            lines.append("|" + "---|" * (len(qwen_models) + 1))
            for m1 in qwen_models:
                cells = [m1]
                for m2 in qwen_models:
                    if m1 not in dists2 or m2 not in dists2:
                        cells.append("—")
                        continue
                    if m1 == m2:
                        cells.append("0.000")
                    else:
                        cells.append(f"{sym_kl(dists2[m1], dists2[m2], all_tokens):.3f}")
                lines.append("| " + " | ".join(cells) + " |")

    out_path = RES_DIR / "analysis.md"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"[wrote] {out_path}")


if __name__ == "__main__":
    main()
