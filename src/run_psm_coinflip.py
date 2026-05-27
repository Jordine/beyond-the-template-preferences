"""Run the canonical PSM coin-flip on one model in either Mode 3 (plaintext)
or Mode 1 (chat-template, user-turn continuation).

The measurement is ALWAYS at user-turn position. For Mode 1 we render the user
message via the tokenizer's chat template but stop tokenisation before any
user-end-of-turn marker or assistant-turn marker, so the model is continuing
the user's still-open message.

Output JSON per cell:
  {
    "model_id": ...,
    "mode": "plaintext" or "open_user_turn",
    "heads_token_ids": [...],
    "tails_token_ids": [...],
    "n_items": 50,
    "mean_P_pref": <float>,
    "mean_q_when_pref_heads": <float>,
    "mean_q_when_pref_tails": <float>,
    "b": <float>,                # mean q across both orderings
    "two_s": <float>,            # mean(q | pref=h) - mean(q | pref=t)
    "results": [<one entry per item, with p_heads, p_tails, top-20 tokens, ...>]
  }
"""
import argparse
import json
import os
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


HEADS_VARIANTS = [" heads", "heads", " Heads", "Heads", " HEADS", "HEADS"]
TAILS_VARIANTS = [" tails", "tails", " Tails", "Tails", " TAILS", "TAILS"]


def collect_variant_token_ids(tokenizer, variants):
    """Return only variants that tokenize to exactly ONE token whose decoded
    form equals the variant string.

    This avoids the failure mode where e.g. ``"HEADS"`` encodes to ``["HEAD", "S"]``
    and we accidentally credit P(heads) with mass on ``'HEAD'`` (a fragment that
    appears all over unrelated continuations: HEAD-LINE, HEAD-ER, ...).
    """
    ids = {}
    seen_first = set()
    for v in variants:
        enc = tokenizer.encode(v, add_special_tokens=False)
        if len(enc) != 1:
            continue
        decoded = tokenizer.decode([enc[0]])
        if decoded != v:
            continue
        if enc[0] in seen_first:
            continue
        seen_first.add(enc[0])
        ids[v] = (enc[0], decoded)
    if not ids:
        raise RuntimeError(
            f"No variant in {variants!r} tokenizes to a single exact-decode token "
            f"for this tokenizer; refuse to run rather than measure word fragments."
        )
    return ids


def render_open_user_turn_continuation(tokenizer, user_content: str) -> str:
    """Render a single-user-message chat template such that the user message
    is still open at the end (no end-of-turn marker, no assistant-turn opener).

    Strategy: ask the tokenizer to apply the chat template with a single
    user message and add_generation_prompt=False, then strip any tokens
    that mark end-of-user-turn / beginning-of-assistant-turn. As a
    consistency check, the rendered text must end at the user_content's
    final characters.
    """
    messages = [{"role": "user", "content": user_content}]
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    # Strip trailing EOT-style markers that close the user turn.
    # We do not want to be model-family-specific in code beyond this: drop
    # everything after the last occurrence of the canonical sentinel
    # "it came up" in the rendered string.
    SENTINEL = "it came up"
    idx = rendered.rfind(SENTINEL)
    if idx == -1:
        raise RuntimeError(
            "Rendered chat-template does not contain the user-message sentinel "
            "'it came up' — refusing to run; the user content may have been "
            "altered or the chat template inserted unexpected tokens."
        )
    truncated = rendered[: idx + len(SENTINEL)]
    return truncated


def run_one_cell(model_id, dataset_path, mode, output_path, hf_token, dtype,
                 base_model=None, subfolder=None):
    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
    print(f"[loading] model={model_id} base={base_model} subfolder={subfolder} mode={mode}")
    if base_model is not None:
        from peft import PeftModel
        tokenizer = AutoTokenizer.from_pretrained(base_model, token=hf_token)
        base = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch_dtype, device_map="auto", token=hf_token
        )
        akw = dict(token=hf_token)
        if subfolder:
            akw["subfolder"] = subfolder
        model = PeftModel.from_pretrained(base, model_id, **akw).merge_and_unload()
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch_dtype, device_map="auto", token=hf_token
        )
    model.eval()

    heads_ids = collect_variant_token_ids(tokenizer, HEADS_VARIANTS)
    tails_ids = collect_variant_token_ids(tokenizer, TAILS_VARIANTS)
    heads_id_set = list({i for i, _ in heads_ids.values()})
    tails_id_set = list({i for i, _ in tails_ids.values()})

    items = json.loads(Path(dataset_path).read_text())
    results = []
    for k, item in enumerate(items):
        if mode == "plaintext":
            text = item["prompt"]  # plaintext canonical Human:... ends 'it came up'
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
        elif mode == "open_user_turn":
            user_content = item["user_content"]
            text = render_open_user_turn_continuation(tokenizer, user_content)
            inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        else:
            raise ValueError(f"unknown mode: {mode}")

        with torch.no_grad():
            out = model(**inputs)
        next_logits = out.logits[0, -1, :]
        probs = F.softmax(next_logits.float(), dim=-1)

        p_heads = float(sum(probs[i].item() for i in heads_id_set))
        p_tails = float(sum(probs[i].item() for i in tails_id_set))
        denom = p_heads + p_tails
        if denom <= 0:
            q = float("nan")
            p_pref = float("nan")
        else:
            q = p_heads / denom
            p_pref = q if item["preferred_outcome"] == "heads" else (1 - q)

        top_vals, top_idx = probs.topk(20)
        top20 = [
            {"token_id": int(idx.item()),
             "token_decoded": tokenizer.decode([int(idx.item())]),
             "p": float(val.item())}
            for val, idx in zip(top_vals, top_idx)
        ]

        results.append({
            "id": item["id"],
            "preferred_outcome": item["preferred_outcome"],
            "p_heads_aggregated": p_heads,
            "p_tails_aggregated": p_tails,
            "q_heads_normalised": q,
            "p_preferred_normalised": p_pref,
            "top20": top20,
        })

    valid = [r for r in results if r["q_heads_normalised"] == r["q_heads_normalised"]]
    q_when_h = [r["q_heads_normalised"] for r in valid if r["preferred_outcome"] == "heads"]
    q_when_t = [r["q_heads_normalised"] for r in valid if r["preferred_outcome"] == "tails"]
    qH = sum(q_when_h) / len(q_when_h) if q_when_h else float("nan")
    qT = sum(q_when_t) / len(q_when_t) if q_when_t else float("nan")

    summary = {
        "model_id": model_id,
        "base_model": base_model,
        "subfolder": subfolder,
        "mode": mode,
        "dataset": str(dataset_path),
        "n_items": len(results),
        "heads_token_ids": {v: i for v, (i, _) in heads_ids.items()},
        "tails_token_ids": {v: i for v, (i, _) in tails_ids.items()},
        "mean_q_when_pref_heads": qH,
        "mean_q_when_pref_tails": qT,
        "b_mean_q": (qH + qT) / 2,
        "two_s": qH - qT,
        "mean_P_pref": 0.5 + 0.5 * (qH - qT),
        "results": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    Path(output_path).write_text(json.dumps(summary, indent=2))
    print(f"[done] {model_id} ({mode})  b={summary['b_mean_q']:+.3f}  2s={summary['two_s']:+.3f}")
    print(f"[saved] {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id")
    parser.add_argument("--mode", choices=["plaintext", "open_user_turn"], required=True)
    parser.add_argument("--dataset", default=None, help="Defaults to psm_coinflip_prompts.json (plaintext) or psm_coinflip_user_messages.json (open_user_turn)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--subfolder", default=None)
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    args = parser.parse_args()

    if args.dataset is None:
        if args.mode == "plaintext":
            args.dataset = "data/psm_coinflip_prompts.json"
        else:
            args.dataset = "data/psm_coinflip_user_messages.json"

    hf_token = os.environ.get("HF_TOKEN")
    run_one_cell(args.model_id, args.dataset, args.mode, args.output, hf_token, args.dtype,
                 base_model=args.base_model, subfolder=args.subfolder)


if __name__ == "__main__":
    main()
