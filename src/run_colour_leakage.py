"""Colour-leakage probe: does the assistant's favourite colour bleed onto the USER turn?

Same single-token-logprob-at-user-turn measurement as run_psm_coinflip.py, but the
measured token set is a palette of colours instead of heads/tails, and the "condition"
is the assistant's PRIMED favourite colour (established in one prefix Q&A exchange).

For each primed colour C and each user context (a user turn that ends right before a
colour word, e.g. "...decided to go with"), we read the next-token distribution at
position -1 and record P(each colour X). Averaging over contexts gives a confusion
matrix M[primed_C][measured_X]. LEAKAGE = diagonal lift: does priming X specifically
raise P(X) on the user turn, above the conditions where X was not the primed favourite?
The column-wise comparison cancels each colour's base rate.

This discriminates two readings of the coinflip result:
  * colour leaks too  -> general self-model contamination (the assistant predicts the
    user resembles itself); safety-coinflip is one instance of egocentric projection.
  * colour does NOT leak but safety does -> the bias is bounded to outcomes the assistant
    is motivated about; coinflip is motivated world-prediction, not self-similarity.

Modes mirror the rest of the project: open_user_turn (chat template, instruct cells) and
plaintext (Human/Assistant, base cells). The final user turn is always left OPEN; the
next token is one the USER would type. Never an assistant-turn position.

Usage:
  python src/run_colour_leakage.py Qwen/Qwen2.5-14B-Instruct --mode open_user_turn \
      --output results/colour_leakage/qwen14b_instruct.json
  python src/run_colour_leakage.py meta-llama/Llama-3.1-8B --mode plaintext \
      --output results/colour_leakage/llama8b_base.json
"""
import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).parent.parent
DEFAULT_CONTEXTS = ROOT / "data" / "colour_leakage_contexts.json"
DEFAULT_CONDITIONS = ROOT / "data" / "colour_leakage_conditions.json"

USER_OPEN_MARKERS = {
    "llama": "<|start_header_id|>user<|end_header_id|>",
    "qwen": "<|im_start|>user",
    "olmo": "<|im_start|>user",
    "gemma": "<start_of_turn>user",
}
USER_CLOSE_MARKERS = {
    "llama": "<|eot_id|>",
    "qwen": "<|im_end|>",
    "olmo": "<|im_end|>",
    "gemma": "<end_of_turn>",
}
ASSISTANT_OPEN_MARKERS = {
    "llama": "<|start_header_id|>assistant<|end_header_id|>",
    "qwen": "<|im_start|>assistant",
    "olmo": "<|im_start|>assistant",
    "gemma": "<start_of_turn>model",
}


def family_of(model_id):
    low = model_id.lower()
    if "llama" in low:
        return "llama"
    if "qwen" in low:
        return "qwen"
    if "olmo" in low:
        return "olmo"
    if "gemma" in low:
        return "gemma"
    raise ValueError(f"unknown family for {model_id}")


def colour_variants(colour):
    caps, up = colour.capitalize(), colour.upper()
    return [f" {colour}", colour, f" {caps}", caps, f" {up}", up]


def collect_single_token_ids(tokenizer, variants):
    """Single-token, exact-decode variants only, deduped by token id. May be empty."""
    ids = {}
    seen = set()
    for v in variants:
        enc = tokenizer.encode(v, add_special_tokens=False)
        if len(enc) != 1:
            continue
        if tokenizer.decode([enc[0]]) != v:
            continue
        if enc[0] in seen:
            continue
        seen.add(enc[0])
        ids[v] = enc[0]
    return ids


def build_conditions(cfg):
    """Ordered [(condition_key, [(user_text, assistant_text), ...]), ...]."""
    ua, tmpl, neu = cfg["user_ask"], cfg["assistant_template"], cfg["assistant_neutral"]
    conds = [("B0", []), ("neutral", [(ua, neu)])]
    for c in cfg["colours"]:
        conds.append((f"prime_{c}", [(ua, tmpl.format(colour=c, Colour=c.capitalize()))]))
    return conds


def render_open_user_turn(tokenizer, exchanges, final_user_content):
    """Chat template; truncate right after final_user_content so the final user turn is OPEN."""
    messages = []
    for u, a in exchanges:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": final_user_content})
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    idx = rendered.rfind(final_user_content)
    if idx == -1:
        raise RuntimeError(f"final_user_content not found in rendered template: {final_user_content!r}")
    return rendered[: idx + len(final_user_content)]


def render_plaintext(exchanges, final_user_content):
    parts = []
    for u, a in exchanges:
        parts.append(f"Human: {u}")
        parts.append(f"Assistant: {a}")
    parts.append(f"Human: {final_user_content}")
    return "\n\n".join(parts)


def assert_final_user_turn_open(rendered, final_user_content, mode, family):
    if not rendered.endswith(final_user_content):
        raise AssertionError(f"rendered does not end at context (suffix={rendered[-40:]!r})")
    if mode == "open_user_turn":
        uo, uc, ao = USER_OPEN_MARKERS[family], USER_CLOSE_MARKERS[family], ASSISTANT_OPEN_MARKERS[family]
        if uo not in rendered:
            raise AssertionError(f"missing user-open marker {uo!r}")
        after = rendered[rendered.rfind(uo) + len(uo):]
        if uc in after:
            raise AssertionError(f"user-close {uc!r} after final user-open — final turn is closed!")
        if ao in after:
            raise AssertionError(f"assistant-open {ao!r} after final user-open — measuring at assistant position!")
    elif mode == "plaintext":
        lh, la = rendered.rfind("Human: "), rendered.rfind("Assistant: ")
        if lh < 0:
            raise AssertionError("rendered missing 'Human: '")
        if la >= 0 and la > lh:
            raise AssertionError("final 'Assistant: ' after final 'Human: ' — measuring at assistant position!")
    else:
        raise ValueError(f"unknown mode: {mode}")


def load_secret(env_name, path):
    v = os.environ.get(env_name)
    if not v:
        p = os.path.expanduser(path)
        if os.path.exists(p):
            v = open(p).read().strip()
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_id")
    ap.add_argument("--mode", choices=["plaintext", "open_user_turn"], required=True)
    ap.add_argument("--contexts", default=str(DEFAULT_CONTEXTS))
    ap.add_argument("--conditions", default=str(DEFAULT_CONDITIONS))
    ap.add_argument("--output", required=True)
    ap.add_argument("--base-model", default=None, help="load model_id as a PEFT adapter on this base")
    ap.add_argument("--subfolder", default=None)
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    args = ap.parse_args()

    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    family = family_of(args.base_model if args.base_model else args.model_id)
    hf_token = load_secret("HF_TOKEN", "~/.secrets/hf_token_main")

    contexts = json.loads(Path(args.contexts).read_text())["contexts"]
    cfg = json.loads(Path(args.conditions).read_text())
    conditions = build_conditions(cfg)

    print(f"[loading] {args.model_id} base={args.base_model} mode={args.mode} "
          f"({len(conditions)} conditions x {len(contexts)} contexts)")
    if args.base_model:
        from peft import PeftModel
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=hf_token)
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch_dtype, device_map="auto", token=hf_token)
        akw = {"token": hf_token}
        if args.subfolder:
            akw["subfolder"] = args.subfolder
        model = PeftModel.from_pretrained(base, args.model_id, **akw).merge_and_unload()
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id, torch_dtype=torch_dtype, device_map="auto", token=hf_token)
    model.eval()

    colour_ids, dropped = {}, []
    for c in cfg["colours"]:
        ids = collect_single_token_ids(tokenizer, colour_variants(c))
        if ids:
            colour_ids[c] = list(ids.values())
        else:
            dropped.append(c)
    if dropped:
        print(f"[warn] no single-token variant for {dropped} — omitted from this model's palette")
    kept = list(colour_ids.keys())
    print(f"[palette] {len(kept)} colours: {kept}")

    per_item = []
    for cond_key, exchanges in conditions:
        for ctx in contexts:
            body = ctx["user_content"]
            if args.mode == "plaintext":
                text = render_plaintext(exchanges, body)
                inputs = tokenizer(text, return_tensors="pt").to(model.device)
            else:
                text = render_open_user_turn(tokenizer, exchanges, body)
                inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
            assert_final_user_turn_open(text, body, args.mode, family)

            with torch.no_grad():
                logits = model(**inputs).logits[0, -1, :]
            probs = F.softmax(logits.float(), dim=-1)
            p_by_colour = {c: float(sum(probs[i].item() for i in ids)) for c, ids in colour_ids.items()}
            top_vals, top_idx = probs.topk(5)
            top5 = [{"tok": tokenizer.decode([int(j)]), "p": float(v)} for v, j in zip(top_vals, top_idx)]
            per_item.append({"condition": cond_key, "context_id": ctx["id"],
                             "p_by_colour": p_by_colour, "top5": top5})

    # matrix[condition][colour] = mean over contexts of P(colour)
    matrix = {}
    for cond_key, _ in conditions:
        rows = [r["p_by_colour"] for r in per_item if r["condition"] == cond_key]
        matrix[cond_key] = {c: sum(r[c] for r in rows) / len(rows) for c in kept}

    # quick diagonal-lift read (full CI in analyze step)
    lifts = []
    for c in kept:
        diag = matrix.get(f"prime_{c}", {}).get(c)
        off = [matrix[f"prime_{o}"][c] for o in kept if o != c and f"prime_{o}" in matrix]
        if diag is not None and off:
            lifts.append(diag - sum(off) / len(off))
    mean_lift = sum(lifts) / len(lifts) if lifts else float("nan")

    summary = {
        "model_id": args.model_id,
        "base_model": args.base_model,
        "subfolder": args.subfolder,
        "mode": args.mode,
        "family": family,
        "colours_kept": kept,
        "colours_dropped": dropped,
        "n_contexts": len(contexts),
        "conditions": [k for k, _ in conditions],
        "matrix": matrix,
        "mean_diagonal_lift": mean_lift,
        "per_item": per_item,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    Path(args.output).write_text(json.dumps(summary, indent=2))
    print(f"[done] {args.model_id} ({args.mode})  mean_diagonal_lift={mean_lift:+.4f}")
    print(f"[saved] {args.output}")


if __name__ == "__main__":
    main()
