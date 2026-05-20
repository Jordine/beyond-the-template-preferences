"""Multi-turn coinflip runner.

Input items contain a `messages` list (chat-format). For each item we:
  1. Apply the tokenizer's chat template to materialize the full string
     (with `continue_final_message=True` so the trailing assistant utterance
     is kept open for next-token measurement).
  2. Tokenize and compute the next-token distribution.
  3. Aggregate P(heads) and P(tails) variants the same way as run_coinflip_multi.py.

For each dataset (n0/n1/n5/n10), save a results JSON with per-item topk +
aggregated probs.
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


def run_dataset(model, tokenizer, heads_id_set, tails_id_set, heads_ids, tails_ids, dataset_path, output_path, top_k=20):
    with open(dataset_path) as f:
        items = json.load(f)
    results = []
    for item in items:
        text = tokenizer.apply_chat_template(
            item["messages"],
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
        )
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model(**inputs)
        next_logits = out.logits[0, -1, :]
        probs = F.softmax(next_logits, dim=-1)
        p_heads = float(sum(probs[i].item() for i in heads_id_set))
        p_tails = float(sum(probs[i].item() for i in tails_id_set))
        denom = p_heads + p_tails
        p_pref = float("nan") if denom <= 0 else (p_heads if item["preferred_outcome"] == "heads" else p_tails) / denom

        top_vals, top_idx = probs.topk(top_k)
        topk = [
            {"token_id": int(idx.item()),
             "token_decoded": tokenizer.decode([int(idx.item())]),
             "p": float(val.item())}
            for val, idx in zip(top_vals, top_idx)
        ]
        results.append({
            "id": item["id"],
            "n_preceding_turns": item.get("n_preceding_turns"),
            "preferred_outcome": item["preferred_outcome"],
            "p_heads_aggregated": p_heads,
            "p_tails_aggregated": p_tails,
            "p_preferred_normalized": p_pref,
            "heads_variant_probs": {v: float(probs[i].item()) for v, (i, _) in heads_ids.items()},
            "tails_variant_probs": {v: float(probs[i].item()) for v, (i, _) in tails_ids.items()},
            "topk": topk,
        })

    valid = [r["p_preferred_normalized"] for r in results if r["p_preferred_normalized"] == r["p_preferred_normalized"]]
    mean_pp = sum(valid) / len(valid) if valid else float("nan")
    pref_h = [r["p_preferred_normalized"] for r in results if r["preferred_outcome"] == "heads"]
    pref_t = [r["p_preferred_normalized"] for r in results if r["preferred_outcome"] == "tails"]
    summary = {
        "dataset": str(dataset_path),
        "n_items": len(results),
        "mean_p_preferred": mean_pp,
        "mean_p_preferred_when_heads": (sum(pref_h) / len(pref_h)) if pref_h else None,
        "mean_p_preferred_when_tails": (sum(pref_t) / len(pref_t)) if pref_t else None,
        "results": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id")
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-tag", required=True)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--subfolder", default=None)
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    print(f"[loading] {args.model_id}  base={args.base_model}  subfolder={args.subfolder}")
    if args.base_model:
        from peft import PeftModel
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=hf_token)
        base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch_dtype, device_map="auto", token=hf_token)
        akwargs = dict(token=hf_token)
        if args.subfolder:
            akwargs["subfolder"] = args.subfolder
        model = PeftModel.from_pretrained(base, args.model_id, **akwargs)
        model = model.merge_and_unload()
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch_dtype, device_map="auto", token=hf_token)
    model.eval()

    heads_ids = collect_variant_token_ids(tokenizer, HEADS_VARIANTS)
    tails_ids = collect_variant_token_ids(tokenizer, TAILS_VARIANTS)
    heads_id_set = list({i for i, _ in heads_ids.values()})
    tails_id_set = list({i for i, _ in tails_ids.values()})

    for dpath in args.datasets:
        name = Path(dpath).stem
        out_path = Path(args.output_dir) / f"{args.output_tag}__{name}.json"
        print(f"[running] {name} -> {out_path.name}")
        s = run_dataset(model, tokenizer, heads_id_set, tails_id_set, heads_ids, tails_ids, dpath, out_path, top_k=args.top_k)
        print(f"  n={s['n_items']}  mean P(pref)={s['mean_p_preferred']:.4f}  "
              f"(pref=heads {s['mean_p_preferred_when_heads']:.4f}, pref=tails {s['mean_p_preferred_when_tails']:.4f})")


if __name__ == "__main__":
    main()
