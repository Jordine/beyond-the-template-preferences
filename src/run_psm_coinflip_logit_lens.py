"""Logit-lens runner for the canonical PSM at the user-turn measurement position.

Per item, do one forward pass with output_hidden_states=True; at each layer
project the final-position residual stream through the model's unembedding to
get a layer-L next-token distribution. Save P(heads), P(tails), normalised q,
plus top-5 tokens at each layer.

Measurement position: same as src/run_psm_coinflip.py — user-turn-continuation.
"""
import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_psm_coinflip import (  # noqa: E402
    HEADS_VARIANTS, TAILS_VARIANTS, collect_variant_token_ids,
    render_open_user_turn_continuation,
)


def _find_final_norm(model):
    """Locate the model's final pre-unembedding norm. Llama / Qwen / OLMo /
    Gemma all expose it as model.model.norm. We apply this norm at every
    layer before projecting through lm_head (nostalgebraist convention),
    otherwise intermediate-layer logits live in a different scale than
    final-layer logits and per-layer probability curves are distorted.
    """
    for path in (("model", "norm"), ("model", "final_layernorm"), ("transformer", "ln_f")):
        cur = model
        ok = True
        for p in path:
            if hasattr(cur, p):
                cur = getattr(cur, p)
            else:
                ok = False
                break
        if ok:
            return cur
    return None  # no final norm found; fall back to identity (caveat in output)


def per_layer_lens(model, tokenizer, mode, items, heads_ids, tails_ids):
    lm_head = model.get_output_embeddings().weight  # (vocab, hidden)
    device = lm_head.device
    final_norm = _find_final_norm(model)
    heads_id_set = list({i for i, _ in heads_ids.values()})
    tails_id_set = list({i for i, _ in tails_ids.values()})

    out_records = []
    for item in items:
        if mode == "plaintext":
            text = item["prompt"]
            inputs = tokenizer(text, return_tensors="pt").to(device)
        else:
            text = render_open_user_turn_continuation(tokenizer, item["user_content"])
            inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)

        with torch.no_grad():
            mo = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
        hidden_states = mo.hidden_states  # tuple of n_layers+1

        per_layer = []
        for L, hs in enumerate(hidden_states):
            resid = hs[0, -1, :].to(dtype=lm_head.dtype)
            if final_norm is not None:
                with torch.no_grad():
                    resid = final_norm(resid)
            logits = resid @ lm_head.T
            probs = F.softmax(logits.float(), dim=-1)
            p_h = float(sum(probs[i].item() for i in heads_id_set))
            p_t = float(sum(probs[i].item() for i in tails_id_set))
            denom = p_h + p_t
            q = (p_h / denom) if denom > 0 else float("nan")

            top_vals, top_idx = logits.topk(5)
            top5 = [
                {"token_id": int(idx.item()),
                 "token_decoded": tokenizer.decode([int(idx.item())]),
                 "logit": float(val.item())}
                for val, idx in zip(top_vals, top_idx)
            ]
            per_layer.append({
                "layer": L,
                "p_heads": p_h,
                "p_tails": p_t,
                "q_heads_normalised": q,
                "top5": top5,
            })

        out_records.append({
            "id": item["id"],
            "preferred_outcome": item["preferred_outcome"],
            "n_layers_total": len(hidden_states),
            "per_layer": per_layer,
        })
    return out_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id")
    parser.add_argument("--mode", choices=["plaintext", "open_user_turn"], default="plaintext")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--subfolder", default=None)
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    args = parser.parse_args()

    if args.dataset is None:
        args.dataset = "data/psm_coinflip_prompts.json" if args.mode == "plaintext" else "data/psm_coinflip_user_messages.json"

    hf_token = os.environ.get("HF_TOKEN")
    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    print(f"[loading] {args.model_id}  base={args.base_model}  subfolder={args.subfolder}  mode={args.mode}")
    if args.base_model:
        from peft import PeftModel
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=hf_token)
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch_dtype, device_map="auto",
            token=hf_token, attn_implementation="eager",
        )
        akw = dict(token=hf_token)
        if args.subfolder:
            akw["subfolder"] = args.subfolder
        model = PeftModel.from_pretrained(base, args.model_id, **akw).merge_and_unload()
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id, torch_dtype=torch_dtype, device_map="auto",
            token=hf_token, attn_implementation="eager",
        )
    model.eval()

    heads_ids = collect_variant_token_ids(tokenizer, HEADS_VARIANTS)
    tails_ids = collect_variant_token_ids(tokenizer, TAILS_VARIANTS)
    items = json.loads(Path(args.dataset).read_text())
    records = per_layer_lens(model, tokenizer, args.mode, items, heads_ids, tails_ids)
    output = {
        "model_id": args.model_id,
        "base_model": args.base_model,
        "subfolder": args.subfolder,
        "mode": args.mode,
        "dataset": args.dataset,
        "results": records,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    Path(args.output).write_text(json.dumps(output))
    print(f"[saved] {args.output}")


if __name__ == "__main__":
    main()
