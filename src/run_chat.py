"""Mode 1 (chat template) runner with optional system prompt + multi-turn conversation.

Each item in the dataset is structured as:
  {
    "id": "...",
    "preferred_outcome": "heads" | "tails",
    "system": "...optional system prompt...",
    "turns": [
       {"role": "user",      "content": "..."},
       {"role": "assistant", "content": "..."},
       ...
    ],
    "continuation": "...partial final user-turn content (we measure the next token after this)..."
  }

We apply the tokenizer's chat template across system + turns, then concatenate the
continuation as additional text inside the next user turn (using continue_final_message
where supported). We measure the next-token distribution at the end of the
concatenated prompt.

Aggregates probability mass over heads/tails variants exactly like run_coinflip.py.
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_coinflip import HEADS_VARIANTS, TAILS_VARIANTS, collect_variant_token_ids


def build_prompt(tokenizer, system, turns, continuation):
    """Build the full prompt as text by templating system+turns then appending the
    continuation inside a final user role. Uses continue_final_message when
    supported, falls back to manual stitching."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.extend(turns)
    # Append the final user turn whose content is the continuation
    messages.append({"role": "user", "content": continuation})

    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
        )
    except Exception:
        # Fallback: tokenize without final message, then manually append
        truncated = messages[:-1]
        base = tokenizer.apply_chat_template(
            truncated, tokenize=False, add_generation_prompt=False
        )
        # Heuristic: try common user-header patterns
        chat_template = tokenizer.chat_template or ""
        if "<|im_start|>" in chat_template or "qwen" in (tokenizer.name_or_path or "").lower():
            user_marker = "<|im_start|>user\n" + continuation
        else:
            user_marker = "<|start_header_id|>user<|end_header_id|>\n\n" + continuation
        prompt = base + user_marker
    return prompt


def run(model_id, dataset_path, output_path, hf_token=None, revision=None, subfolder=None,
        dtype="float16", base_model=None, top_k=20):
    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
    print(f"[loading] model_id={model_id} base_model={base_model} dtype={dtype}")

    if base_model is not None:
        from peft import PeftModel
        tokenizer = AutoTokenizer.from_pretrained(base_model, token=hf_token)
        base = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch_dtype, device_map="auto", token=hf_token
        )
        akw = dict(token=hf_token)
        if subfolder:
            akw["subfolder"] = subfolder
        if revision:
            akw["revision"] = revision
        model = PeftModel.from_pretrained(base, model_id, **akw)
        model = model.merge_and_unload()
    else:
        tk = dict(token=hf_token)
        mk = dict(torch_dtype=torch_dtype, device_map="auto", token=hf_token)
        if revision:
            tk["revision"] = revision
            mk["revision"] = revision
        if subfolder:
            tk["subfolder"] = subfolder
            mk["subfolder"] = subfolder
        tokenizer = AutoTokenizer.from_pretrained(model_id, **tk)
        model = AutoModelForCausalLM.from_pretrained(model_id, **mk)
    model.eval()

    heads_ids = collect_variant_token_ids(tokenizer, HEADS_VARIANTS)
    tails_ids = collect_variant_token_ids(tokenizer, TAILS_VARIANTS)
    print(f"  HEADS first-tokens -> {[(v, i, r) for v, (i, r) in heads_ids.items()]}")
    print(f"  TAILS first-tokens -> {[(v, i, r) for v, (i, r) in tails_ids.items()]}")
    heads_id_set = list({i for i, _ in heads_ids.values()})
    tails_id_set = list({i for i, _ in tails_ids.values()})

    with open(dataset_path) as f:
        items = json.load(f)
    print(f"[running] {len(items)} items")

    results = []
    for k, item in enumerate(items):
        prompt = build_prompt(
            tokenizer,
            system=item.get("system"),
            turns=item.get("turns", []),
            continuation=item.get("continuation", item.get("prompt", "")),
        )
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model(**inputs)
        next_logits = out.logits[0, -1, :]
        probs = F.softmax(next_logits, dim=-1)

        p_heads = float(sum(probs[i].item() for i in heads_id_set))
        p_tails = float(sum(probs[i].item() for i in tails_id_set))
        denom = p_heads + p_tails
        if denom <= 0:
            p_pref = float("nan")
        else:
            p_pref = (p_heads if item["preferred_outcome"] == "heads" else p_tails) / denom

        heads_variant_probs = {v: float(probs[i].item()) for v, (i, _) in heads_ids.items()}
        tails_variant_probs = {v: float(probs[i].item()) for v, (i, _) in tails_ids.items()}

        top_vals, top_idx = probs.topk(top_k)
        topk = [
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
            "heads_variant_probs": heads_variant_probs,
            "tails_variant_probs": tails_variant_probs,
            "p_preferred_normalized": p_pref,
            "topk": topk,
        })
        if (k + 1) % 10 == 0:
            done = [r for r in results if r["p_preferred_normalized"] == r["p_preferred_normalized"]]
            mean_so_far = (sum(r["p_preferred_normalized"] for r in done) / len(done)) if done else float("nan")
            print(f"  {k+1}/{len(items)} mean_p_pref_so_far={mean_so_far:.3f}")

    valid = [r["p_preferred_normalized"] for r in results if r["p_preferred_normalized"] == r["p_preferred_normalized"]]
    mean_p_pref = sum(valid) / len(valid) if valid else float("nan")
    pref_heads = [r["p_preferred_normalized"] for r in results if r["preferred_outcome"] == "heads"]
    pref_tails = [r["p_preferred_normalized"] for r in results if r["preferred_outcome"] == "tails"]

    summary = {
        "model_id": model_id,
        "revision": revision,
        "subfolder": subfolder,
        "dtype": dtype,
        "base_model": base_model,
        "mode": "chat_template",
        "heads_variant_token_ids": {v: i for v, (i, _) in heads_ids.items()},
        "tails_variant_token_ids": {v: i for v, (i, _) in tails_ids.items()},
        "n_items": len(results),
        "mean_p_preferred": mean_p_pref,
        "mean_p_preferred_when_heads": sum(pref_heads) / len(pref_heads) if pref_heads else None,
        "mean_p_preferred_when_tails": sum(pref_tails) / len(pref_tails) if pref_tails else None,
        "results": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[done] {model_id}")
    print(f"  mean P(preferred) = {mean_p_pref:.4f}")
    print(f"  saved -> {output_path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--subfolder", default=None)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--system-override", default=None,
                         help="If set, overrides item-level system field with this string for all items")
    args = parser.parse_args()
    hf_token = os.environ.get("HF_TOKEN")
    # Optionally override system prompt for all items
    if args.system_override is not None:
        with open(args.dataset) as f:
            items = json.load(f)
        for it in items:
            it["system"] = args.system_override if args.system_override else None
        # Write to a temp file
        import tempfile
        tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump(items, tmp)
        tmp.close()
        args.dataset = tmp.name
    run(args.model_id, args.dataset, args.output, hf_token=hf_token, revision=args.revision,
        subfolder=args.subfolder, dtype=args.dtype, base_model=args.base_model, top_k=args.top_k)
