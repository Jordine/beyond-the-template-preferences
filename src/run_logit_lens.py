"""Logit lens + attention probe runner.

For each item:
  - forward pass with output_hidden_states=True, output_attentions=True
  - at the FINAL position, for each layer L:
    * project residual_stream_L through model.lm_head -> vocab logits
    * extract P(heads variants), P(tails variants), top-5 token ids + logits
    * extract attention from final pos to all preceding positions, mean over heads
  - save everything per item

Handles both Mode 3 (plaintext prompt) and chat-template-based inputs
(Mode 1 or Mode 2 — pass via --messages-mode and dataset items having
either `prompt` or `messages`).
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
    ids = []
    seen = set()
    for v in variants:
        enc = tokenizer.encode(v, add_special_tokens=False)
        if not enc:
            continue
        first = enc[0]
        if first not in seen:
            ids.append(first)
            seen.add(first)
    return ids


def make_input_ids(tokenizer, item, messages_mode):
    """Return (input_ids tensor, the tokenized prompt string for reference)."""
    if messages_mode == "prompt":
        text = item["prompt"]
        return tokenizer(text, return_tensors="pt", add_special_tokens=True).input_ids, text
    elif messages_mode == "messages":
        text = tokenizer.apply_chat_template(
            item["messages"],
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
        )
        return tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids, text


def run(args):
    hf_token = os.environ.get("HF_TOKEN")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    print(f"[loading] {args.model_id}  base={args.base_model}  subfolder={args.subfolder}")
    if args.base_model:
        from peft import PeftModel
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=hf_token)
        base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype, device_map="auto", token=hf_token, attn_implementation="eager")
        akw = dict(token=hf_token)
        if args.subfolder:
            akw["subfolder"] = args.subfolder
        model = PeftModel.from_pretrained(base, args.model_id, **akw)
        model = model.merge_and_unload()
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=dtype, device_map="auto", token=hf_token, attn_implementation="eager")
    model.eval()

    heads_ids = collect_variant_token_ids(tokenizer, HEADS_VARIANTS)
    tails_ids = collect_variant_token_ids(tokenizer, TAILS_VARIANTS)
    print(f"  heads token ids: {heads_ids}")
    print(f"  tails token ids: {tails_ids}")

    lm_head = model.get_output_embeddings().weight  # (vocab, hidden_dim) — may be tied with embed
    device = lm_head.device

    with open(args.dataset) as f:
        items = json.load(f)
    print(f"[running] {len(items)} items, mode={args.messages_mode}")

    output = {
        "model_id": args.model_id,
        "base_model": args.base_model,
        "subfolder": args.subfolder,
        "messages_mode": args.messages_mode,
        "heads_token_ids": heads_ids,
        "tails_token_ids": tails_ids,
        "dataset": args.dataset,
        "results": [],
    }

    for k, item in enumerate(items):
        input_ids, text = make_input_ids(tokenizer, item, args.messages_mode)
        input_ids = input_ids.to(device)
        with torch.no_grad():
            out = model(input_ids=input_ids, output_hidden_states=True, output_attentions=True, return_dict=True, use_cache=False)
        # hidden_states is a tuple of (n_layers+1) tensors, each (1, seq_len, hidden_dim)
        # attentions is tuple of n_layers tensors, each (1, n_heads, seq_len, seq_len)
        hidden_states = out.hidden_states
        attentions = out.attentions
        seq_len = input_ids.shape[1]

        per_layer = []
        for layer_idx, hs in enumerate(hidden_states):
            # final position residual: shape (hidden_dim,)
            residual = hs[0, -1, :].to(dtype=lm_head.dtype)
            # project through lm_head
            logits = residual @ lm_head.T  # (vocab,)
            # numerically stable softmax
            probs = F.softmax(logits.float(), dim=-1)
            p_heads = float(sum(probs[i].item() for i in heads_ids))
            p_tails = float(sum(probs[i].item() for i in tails_ids))
            denom = p_heads + p_tails
            p_heads_norm = float("nan") if denom <= 0 else p_heads / denom

            # top-5 tokens
            top_vals, top_idx = logits.topk(5)
            top5 = [
                {"token_id": int(idx.item()),
                 "decoded": tokenizer.decode([int(idx.item())]),
                 "logit": float(val.item())}
                for val, idx in zip(top_vals, top_idx)
            ]

            entry = {
                "layer": layer_idx,
                "p_heads": p_heads,
                "p_tails": p_tails,
                "p_heads_normalized": p_heads_norm,
                "top5": top5,
            }
            per_layer.append(entry)

        # attention: per-layer, attention from final position to all preceding, mean over heads
        per_layer_attn = []
        for layer_idx, att in enumerate(attentions):
            # att: (1, n_heads, seq_len, seq_len). We want att[0, :, -1, :] -> (n_heads, seq_len)
            row = att[0, :, -1, :].mean(dim=0)  # (seq_len,) -- mean over heads
            per_layer_attn.append([float(x) for x in row.cpu().tolist()])

        # Decoded tokens for position-mapping later
        token_strs = [tokenizer.decode([int(t)]) for t in input_ids[0].cpu().tolist()]

        # final-layer 2s estimate for sanity
        final = per_layer[-1]
        output["results"].append({
            "id": item.get("id"),
            "preferred_outcome": item.get("preferred_outcome"),
            "seq_len": seq_len,
            "tokens": token_strs,
            "per_layer": per_layer,
            "per_layer_attention_to_keys": per_layer_attn,
            "final_p_heads_normalized": final["p_heads_normalized"],
        })
        if (k + 1) % 10 == 0:
            print(f"  {k+1}/{len(items)}  final_layer P(heads)={final['p_heads_normalized']:.3f}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f)
    print(f"[saved] {args.output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--messages-mode", choices=["prompt", "messages"], required=True)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--subfolder", default=None)
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
