"""Empty user-turn next-token prediction.

For each model, feed two prompts:
  - Mode 1 (chat template): the model-family-specific user-role marker + empty content
  - Mode 3 (plaintext): "Human:\n" — no role marker, just plaintext label

Capture top-K next-token logprobs. Save to a JSON file per (model, mode).

The hypothesis: if post-training installs priors over user behavior, the chat-template
user-role-open with empty content should produce a highly concentrated, content-rich
distribution over how-users-typically-begin-utterances. Base models on the same
input will produce wild / off-distribution outputs (chat template tokens are OOD
for them); their Mode 3 ("Human:") output is the proper null reference.
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_mode1_prefix(model_id):
    """Family-specific chat-template open for the user role with no content yet."""
    mid = model_id.lower()
    if "llama-3" in mid:
        # Llama 3.x chat template, user role opening
        return "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
    if "qwen2.5" in mid or "qwen-2.5" in mid:
        return "<|im_start|>user\n"
    raise ValueError(f"Unknown family for {model_id}")


MODE3_PREFIX = "Human:\n"


def top_k_after(model, tokenizer, prefix, k=50):
    inputs = tokenizer(prefix, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model(**inputs)
    logits = out.logits[0, -1, :]
    probs = F.softmax(logits, dim=-1)
    top_vals, top_idx = probs.topk(k)
    return [
        {"token_id": int(idx.item()),
         "token_decoded": tokenizer.decode([int(idx.item())]),
         "p": float(val.item()),
         "logp": float(torch.log(val).item())}
        for val, idx in zip(top_vals, top_idx)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id")
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--subfolder", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--top-k", type=int, default=50)
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
        # use base-model tokenizer for chat template detection / Mode1 prefix
        ref_model_id = args.base_model
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch_dtype, device_map="auto", token=hf_token)
        ref_model_id = args.model_id
    model.eval()

    out = {"model_id": args.model_id, "base_model": args.base_model, "subfolder": args.subfolder, "modes": {}}

    # Mode 1: chat-template user-role open with no content
    try:
        mode1 = get_mode1_prefix(ref_model_id)
        mode1_top = top_k_after(model, tokenizer, mode1, k=args.top_k)
        out["modes"]["mode1_chat_template_user_open"] = {
            "prefix_repr": mode1.replace("\n", "\\n"),
            "top_k": mode1_top,
        }
        print(f"  mode1 top-3: {[(t['token_decoded'], round(t['p'], 4)) for t in mode1_top[:3]]}")
    except ValueError as e:
        out["modes"]["mode1_chat_template_user_open"] = {"error": str(e)}

    # Mode 3: plaintext "Human:\n"
    mode3_top = top_k_after(model, tokenizer, MODE3_PREFIX, k=args.top_k)
    out["modes"]["mode3_plaintext_human"] = {
        "prefix_repr": MODE3_PREFIX.replace("\n", "\\n"),
        "top_k": mode3_top,
    }
    print(f"  mode3 top-3: {[(t['token_decoded'], round(t['p'], 4)) for t in mode3_top[:3]]}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {args.output}")


if __name__ == "__main__":
    main()
