"""Load a model once and run the PSM coinflip on multiple datasets.

Used for the user-state axis sweep — same as src/run_coinflip.py but amortizes
the model-load cost over many axes.
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


HEADS_VARIANTS = [" heads", "heads", " Heads", "Heads", " HEADS", "HEADS"]
TAILS_VARIANTS = [" tails", "tails", " Tails", "Tails", " TAILS", "TAILS"]


def collect_variant_token_ids(tokenizer, variants):
    ids = {}
    for v in variants:
        enc = tokenizer.encode(v, add_special_tokens=False)
        if not enc:
            continue
        first = enc[0]
        if first in {sid for sid, _ in ids.values()}:
            continue
        ids[v] = (first, tokenizer.decode([first]))
    return ids


def run_one_dataset(model, tokenizer, heads_id_set, tails_id_set, heads_ids, tails_ids, dataset_path, output_path, top_k=20):
    with open(dataset_path) as f:
        items = json.load(f)

    results = []
    for k, item in enumerate(items):
        inputs = tokenizer(item["prompt"], return_tensors="pt").to(model.device)
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
            "axis": item.get("axis"),
            "preferred_outcome": item["preferred_outcome"],
            "preferred_label": item.get("preferred_label"),
            "dispreferred_label": item.get("dispreferred_label"),
            "p_heads_aggregated": p_heads,
            "p_tails_aggregated": p_tails,
            "heads_variant_probs": heads_variant_probs,
            "tails_variant_probs": tails_variant_probs,
            "p_preferred_normalized": p_pref,
            "topk": topk,
        })

    valid = [r["p_preferred_normalized"] for r in results if r["p_preferred_normalized"] == r["p_preferred_normalized"]]
    mean_p_pref = sum(valid) / len(valid) if valid else float("nan")
    pref_heads = [r["p_preferred_normalized"] for r in results if r["preferred_outcome"] == "heads"]
    pref_tails = [r["p_preferred_normalized"] for r in results if r["preferred_outcome"] == "tails"]
    summary = {
        "dataset": str(dataset_path),
        "n_items": len(results),
        "mean_p_preferred": mean_p_pref,
        "mean_p_preferred_when_heads": (sum(pref_heads) / len(pref_heads)) if pref_heads else None,
        "mean_p_preferred_when_tails": (sum(pref_tails) / len(pref_tails)) if pref_tails else None,
        "results": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id")
    parser.add_argument("--datasets", nargs="+", required=True, help="One or more dataset paths")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-tag", required=True, help="Prefix for output filenames")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--subfolder", default=None)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    print(f"[loading] model_id={args.model_id} base_model={args.base_model} subfolder={args.subfolder}")

    if args.base_model is not None:
        from peft import PeftModel
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=hf_token)
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch_dtype, device_map="auto", token=hf_token
        )
        adapter_kwargs = dict(token=hf_token)
        if args.subfolder:
            adapter_kwargs["subfolder"] = args.subfolder
        if args.revision:
            adapter_kwargs["revision"] = args.revision
        model = PeftModel.from_pretrained(base, args.model_id, **adapter_kwargs)
        model = model.merge_and_unload()
    else:
        tok_kwargs = dict(token=hf_token)
        mdl_kwargs = dict(torch_dtype=torch_dtype, device_map="auto", token=hf_token)
        if args.revision:
            tok_kwargs["revision"] = args.revision
            mdl_kwargs["revision"] = args.revision
        if args.subfolder:
            tok_kwargs["subfolder"] = args.subfolder
            mdl_kwargs["subfolder"] = args.subfolder
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, **tok_kwargs)
        model = AutoModelForCausalLM.from_pretrained(args.model_id, **mdl_kwargs)
    model.eval()

    heads_ids = collect_variant_token_ids(tokenizer, HEADS_VARIANTS)
    tails_ids = collect_variant_token_ids(tokenizer, TAILS_VARIANTS)
    print(f"  HEADS first-tokens -> {[(v, i, r) for v, (i, r) in heads_ids.items()]}")
    print(f"  TAILS first-tokens -> {[(v, i, r) for v, (i, r) in tails_ids.items()]}")
    heads_id_set = list({i for i, _ in heads_ids.values()})
    tails_id_set = list({i for i, _ in tails_ids.values()})

    for dpath in args.datasets:
        axis_name = Path(dpath).stem
        output_path = Path(args.output_dir) / f"{args.output_tag}__{axis_name}.json"
        print(f"\n[running] {axis_name} -> {output_path.name}")
        s = run_one_dataset(model, tokenizer, heads_id_set, tails_id_set, heads_ids, tails_ids, dpath, output_path, top_k=args.top_k)
        print(f"  mean P(preferred) = {s['mean_p_preferred']:.4f}  "
              f"(pref=heads: {s['mean_p_preferred_when_heads']:.4f}, "
              f"pref=tails: {s['mean_p_preferred_when_tails']:.4f})")


if __name__ == "__main__":
    main()
