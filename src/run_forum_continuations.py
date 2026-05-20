"""Forum-continuation top-k extractor.

For each forum scenario, feed the prompt (Alice's line + 'Bob:') and extract
the top-K next-token logprobs. Mode 3 — no chat template.

Outputs: results/forum_continuations/{model}.json
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--subfolder", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
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

    with open(args.dataset) as f:
        items = json.load(f)

    output = {"model_id": args.model_id, "base_model": args.base_model, "subfolder": args.subfolder, "scenarios": []}
    for item in items:
        inputs = tokenizer(item["prompt"], return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model(**inputs)
        probs = F.softmax(out.logits[0, -1, :], dim=-1)
        top_vals, top_idx = probs.topk(args.top_k)
        topk = [
            {"token_id": int(idx.item()),
             "token_decoded": tokenizer.decode([int(idx.item())]),
             "p": float(val.item())}
            for val, idx in zip(top_vals, top_idx)
        ]
        output["scenarios"].append({
            "id": item["id"],
            "topic": item.get("topic"),
            "alice": item.get("alice"),
            "top_k": topk,
        })
        print(f"  {item['id']:25s} -> top-3: {[(t['token_decoded'], round(t['p'], 3)) for t in topk[:3]]}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[saved] {args.output}")


if __name__ == "__main__":
    main()
