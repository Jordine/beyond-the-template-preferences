"""Run the PSM canonical coinflip experiment on one HuggingFace model.

For each of 50 prompts, measure P(heads) and P(tails) as the very next token
after the prompt "...it came up", normalize to P(heads)+P(tails)=1, and compute
P(preferred_outcome) aggregated across items.

Base models -> expect ~0.5. Chat/character-trained -> expect > 0.5 if PSM holds.
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def find_token_id(tokenizer, word_with_space, fallback_word):
    """Find the single-token ID for a word. Tries ' word' first (most likely
    mid-text), then 'word' without space."""
    ids = tokenizer.encode(word_with_space, add_special_tokens=False)
    if len(ids) == 1:
        return ids[0], word_with_space
    ids2 = tokenizer.encode(fallback_word, add_special_tokens=False)
    if len(ids2) == 1:
        return ids2[0], fallback_word
    # Multi-token -> take first
    return ids[0], word_with_space + " [MULTI-TOKEN]"


def run(model_id, dataset_path, output_path, hf_token=None, revision=None, subfolder=None, dtype="float16", base_model=None):
    """If base_model is set, model_id is treated as a LoRA adapter loaded on top of base_model."""
    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
    print(f"[loading] model_id={model_id} base_model={base_model} subfolder={subfolder} revision={revision} dtype={dtype}")

    if base_model is not None:
        # LoRA path: load base, then attach adapter from model_id/subfolder
        from peft import PeftModel
        tok_src = base_model
        tokenizer = AutoTokenizer.from_pretrained(tok_src, token=hf_token)
        base = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch_dtype, device_map="auto", token=hf_token
        )
        adapter_kwargs = dict(token=hf_token)
        if subfolder:
            adapter_kwargs["subfolder"] = subfolder
        if revision:
            adapter_kwargs["revision"] = revision
        model = PeftModel.from_pretrained(base, model_id, **adapter_kwargs)
        model = model.merge_and_unload()
    else:
        tok_kwargs = dict(token=hf_token)
        mdl_kwargs = dict(torch_dtype=torch_dtype, device_map="auto", token=hf_token)
        if revision:
            tok_kwargs["revision"] = revision
            mdl_kwargs["revision"] = revision
        if subfolder:
            tok_kwargs["subfolder"] = subfolder
            mdl_kwargs["subfolder"] = subfolder
        tokenizer = AutoTokenizer.from_pretrained(model_id, **tok_kwargs)
        model = AutoModelForCausalLM.from_pretrained(model_id, **mdl_kwargs)
    model.eval()

    heads_id, heads_repr = find_token_id(tokenizer, " heads", "heads")
    tails_id, tails_repr = find_token_id(tokenizer, " tails", "tails")
    print(f"  heads token -> id={heads_id} repr={heads_repr!r} decoded={tokenizer.decode([heads_id])!r}")
    print(f"  tails token -> id={tails_id} repr={tails_repr!r} decoded={tokenizer.decode([tails_id])!r}")

    with open(dataset_path) as f:
        items = json.load(f)
    print(f"[running] {len(items)} items")

    results = []
    for k, item in enumerate(items):
        inputs = tokenizer(item["prompt"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**inputs)
        next_logits = out.logits[0, -1, :]
        logprobs = F.log_softmax(next_logits, dim=-1)
        p_heads = float(logprobs[heads_id].exp().item())
        p_tails = float(logprobs[tails_id].exp().item())
        denom = p_heads + p_tails
        if denom <= 0:
            p_pref = float("nan")
        else:
            p_pref = (p_heads if item["preferred_outcome"] == "heads" else p_tails) / denom

        top_vals, top_idx = logprobs.topk(10)
        top10 = [
            {"token": tokenizer.decode([int(idx.item())]), "p": float(val.exp().item())}
            for val, idx in zip(top_vals, top_idx)
        ]

        results.append({
            "id": item["id"],
            "preferred_outcome": item["preferred_outcome"],
            "p_heads": p_heads,
            "p_tails": p_tails,
            "p_preferred_normalized": p_pref,
            "top10": top10,
        })
        if (k + 1) % 10 == 0:
            print(f"  {k+1}/{len(items)} mean_p_pref_so_far={sum(r['p_preferred_normalized'] for r in results)/len(results):.3f}")

    valid = [r["p_preferred_normalized"] for r in results if r["p_preferred_normalized"] == r["p_preferred_normalized"]]
    mean_p_pref = sum(valid) / len(valid) if valid else float("nan")

    # Split by which outcome was labeled heads -- diagnostic for heads-bias confound
    pref_heads = [r["p_preferred_normalized"] for r in results if r["preferred_outcome"] == "heads"]
    pref_tails = [r["p_preferred_normalized"] for r in results if r["preferred_outcome"] == "tails"]

    summary = {
        "model_id": model_id,
        "revision": revision,
        "subfolder": subfolder,
        "dtype": dtype,
        "heads_token_id": heads_id,
        "tails_token_id": tails_id,
        "heads_token_repr": heads_repr,
        "tails_token_repr": tails_repr,
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
    print(f"  mean P(preferred) = {mean_p_pref:.4f}  (0.5 = no bias)")
    print(f"  when preferred=heads: {summary['mean_p_preferred_when_heads']:.4f}")
    print(f"  when preferred=tails: {summary['mean_p_preferred_when_tails']:.4f}")
    print(f"  saved -> {output_path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id")
    parser.add_argument("--dataset", default="data/psm_canonical.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--subfolder", default=None)
    parser.add_argument("--base-model", default=None, help="If set, model_id is a LoRA adapter applied on top of base_model")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    args = parser.parse_args()
    hf_token = os.environ.get("HF_TOKEN")
    run(args.model_id, args.dataset, args.output, hf_token=hf_token, revision=args.revision, subfolder=args.subfolder, dtype=args.dtype, base_model=args.base_model)
