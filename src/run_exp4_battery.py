"""EXP-4 user-turn prediction battery runner.

Loads a category JSON (one of naming / outcome / evaluation / mood / refusal_aftermath).
For each item, builds the prompt (raw text or chat template per item.mode) and
measures the next-token softmax. For each candidate-token list, sums probability
mass over those candidates and reports per-bucket totals. Also stores top-K
for diagnostics.
"""
import argparse, json, os
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_chat import build_prompt as build_chat_prompt
from run_coinflip import collect_variant_token_ids


def first_token_id(tokenizer, s):
    enc = tokenizer.encode(s, add_special_tokens=False)
    return enc[0] if enc else None


def run(model_id, dataset_path, output_path, hf_token=None, base_model=None,
        subfolder=None, dtype="float16", top_k=20):
    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
    print(f"[loading] {model_id} base={base_model} dtype={dtype}")
    if base_model is not None:
        from peft import PeftModel
        tokenizer = AutoTokenizer.from_pretrained(base_model, token=hf_token)
        base = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch_dtype, device_map="auto", token=hf_token)
        akw = dict(token=hf_token)
        if subfolder: akw["subfolder"] = subfolder
        model = PeftModel.from_pretrained(base, model_id, **akw).merge_and_unload()
    else:
        kw = dict(token=hf_token)
        if subfolder: kw["subfolder"] = subfolder
        tokenizer = AutoTokenizer.from_pretrained(model_id, **kw)
        mkw = dict(torch_dtype=torch_dtype, device_map="auto", token=hf_token)
        if subfolder: mkw["subfolder"] = subfolder
        model = AutoModelForCausalLM.from_pretrained(model_id, **mkw)
    model.eval()

    with open(dataset_path) as f:
        cat = json.load(f)
    print(f"[running] category={cat['category']} mode={cat['mode']} n_items={len(cat['items'])}")

    results = []
    for item in cat["items"]:
        if cat["mode"] == "chat":
            prompt_text = build_chat_prompt(
                tokenizer, system=item.get("system"),
                turns=item.get("turns", []),
                continuation=item.get("continuation", ""),
            )
        else:
            prompt_text = item["prompt"]

        inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model(**inputs)
        next_logits = out.logits[0, -1, :]
        probs = F.softmax(next_logits, dim=-1)

        # Candidates can be either per-item or per-category default
        candidates = item.get("candidates") or cat.get("candidates_default", {})

        bucket_p = {}
        per_token = {}
        for bucket, toks in candidates.items():
            total = 0.0
            for s in toks:
                tid = first_token_id(tokenizer, s)
                if tid is None:
                    continue
                p = float(probs[tid].item())
                per_token[f"{bucket}::{s}"] = p
                total += p
            bucket_p[bucket] = total

        top_vals, top_idx = probs.topk(top_k)
        topk = [{"token_id": int(i.item()),
                 "token_decoded": tokenizer.decode([int(i.item())]),
                 "p": float(v.item())} for v, i in zip(top_vals, top_idx)]

        results.append({
            "id": item["id"],
            "bucket_p": bucket_p,
            "per_token_p": per_token,
            "topk": topk,
        })
        # Quick console print
        bs = ", ".join(f"{k}={v:.3f}" for k, v in bucket_p.items())
        print(f"  {item['id']:35s}  {bs}")

    summary = {
        "model_id": model_id,
        "base_model": base_model,
        "subfolder": subfolder,
        "category": cat["category"],
        "mode": cat["mode"],
        "n_items": len(results),
        "results": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] saved -> {output_path}")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("model_id")
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--base-model", default=None)
    p.add_argument("--subfolder", default=None)
    p.add_argument("--dtype", default="float16")
    p.add_argument("--top-k", type=int, default=20)
    args = p.parse_args()
    hf_token = os.environ.get("HF_TOKEN")
    run(args.model_id, args.dataset, args.output, hf_token=hf_token,
        base_model=args.base_model, subfolder=args.subfolder, dtype=args.dtype, top_k=args.top_k)
