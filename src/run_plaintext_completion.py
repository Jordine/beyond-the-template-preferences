"""Single-prompt N-sample completion runner.

Two modes:
  --mode raw    : pure text completion. Prompt is BOS + prompt-text. No chat tokens.
                  Used for Experiment 1 (PSM "anthrax" diagnostic).
  --mode user   : chat-template user-prefix completion. Prompt is BOS + user-role-header
                  + prefix. Generation stops at <|eot_id|>. Model continues the user's
                  partial turn. Used for Experiment 2.

For each prompt × N samples, generates and saves:
  prompt_id, prompt_text, sample_idx, completion (raw text decoded with no skip-special-tokens),
  finish_reason ('eot' | 'maxlen'), tokens_generated.

Output JSON shape:
  {model_id, base_model, subfolder, mode, prompts_file, n_samples, results: [...]}
"""
import argparse, json, os, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def detect_role_markers(tokenizer):
    """Return (bos, user_header, eot_str, eot_id) for the tokenizer."""
    name = (tokenizer.name_or_path or "").lower()
    template = tokenizer.chat_template or ""
    if "qwen" in name or "<|im_start|>" in template:
        eot_str = "<|im_end|>"
        eot_id = tokenizer.convert_tokens_to_ids(eot_str)
        return ("", "<|im_start|>user\n", eot_str, eot_id)
    eot_str = "<|eot_id|>"
    eot_id = tokenizer.convert_tokens_to_ids(eot_str)
    return ("<|begin_of_text|>",
            "<|start_header_id|>user<|end_header_id|>\n\n",
            eot_str, eot_id)


def generate_completions_batch(model, tokenizer, prompt_text, n_samples,
                                max_new_tokens, temperature, top_p, eot_id):
    """Generate `n_samples` completions of the same prompt in ONE forward
    batch via num_return_sequences. Returns a list of (text, finish, n_gen).

    This is ~10-30x faster on bigger models than calling generate(batch=1)
    n_samples times in a Python loop, because the prompt is encoded once,
    KV cache is shared across samples, and the output token-decoding pass
    is amortized.
    """
    inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(model.device)
    in_len = inputs.input_ids.shape[1]
    pad_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=n_samples,
            eos_token_id=eot_id,
            pad_token_id=pad_id,
        )
    # out shape: (n_samples, in_len + actual_new_tokens). Sequences that hit
    # EOT early are padded on the right with pad_id by .generate().
    results = []
    for i in range(out.shape[0]):
        new_tokens = out[i, in_len:].tolist()
        # Strip trailing pad tokens (after EOT or after maxlen if pad==eos==eot)
        # Find the first EOT (which is what stopped generation for this sample).
        finish = "maxlen"
        end = len(new_tokens)
        for j, tok in enumerate(new_tokens):
            if tok == eot_id:
                end = j
                finish = "eot"
                break
        new_tokens = new_tokens[:end]
        text = tokenizer.decode(new_tokens, skip_special_tokens=False)
        results.append((text, finish, len(new_tokens)))
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model_id")
    p.add_argument("--base-model", default=None)
    p.add_argument("--subfolder", default=None)
    p.add_argument("--prompts", required=True, help="JSON file of prompts")
    p.add_argument("--mode", choices=["raw", "user"], required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--n-samples", type=int, default=50)
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.92)
    p.add_argument("--dtype", default="float16")
    args = p.parse_args()

    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    hf_token = os.environ.get("HF_TOKEN")

    if args.base_model:
        from peft import PeftModel
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=hf_token)
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch_dtype, device_map="auto", token=hf_token)
        akw = dict(token=hf_token)
        if args.subfolder: akw["subfolder"] = args.subfolder
        model = PeftModel.from_pretrained(base, args.model_id, **akw).merge_and_unload()
    else:
        kw = dict(token=hf_token)
        if args.subfolder: kw["subfolder"] = args.subfolder
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, **kw)
        mkw = dict(torch_dtype=torch_dtype, device_map="auto", token=hf_token)
        if args.subfolder: mkw["subfolder"] = args.subfolder
        model = AutoModelForCausalLM.from_pretrained(args.model_id, **mkw)
    model.eval()

    bos, user_hdr, eot_str, eot_id = detect_role_markers(tokenizer)
    print(f"[loaded] {args.model_id} mode={args.mode} eot_id={eot_id}")

    with open(args.prompts) as f:
        prompts = json.load(f)
    print(f"[running] {len(prompts)} prompts × {args.n_samples} samples (max_tokens={args.max_tokens})")

    results = []
    t0 = time.time()
    for pi, p_obj in enumerate(prompts):
        if args.mode == "raw":
            prompt_text = bos + p_obj["prompt"]
        else:  # user mode
            prompt_text = bos + user_hdr + p_obj["prefix"]

        # Seed once per prompt; the n_samples diverge through the sampler.
        # (Was per-sample with `1000*pi+si` — reproducibility no longer
        # per-sample but per-prompt-aggregate, which is what we report.)
        torch.manual_seed(1000 * pi)
        batch_results = generate_completions_batch(
            model, tokenizer, prompt_text, args.n_samples,
            args.max_tokens, args.temperature, args.top_p, eot_id,
        )
        for si, (text, finish, n_gen) in enumerate(batch_results):
            results.append({
                "prompt_id": p_obj["id"],
                "sample_idx": si,
                "completion": text,
                "finish_reason": finish,
                "tokens_generated": n_gen,
            })
        elapsed = time.time() - t0
        done = (pi + 1) * args.n_samples
        total = len(prompts) * args.n_samples
        print(f"  prompt[{p_obj['id']}] done  ({done}/{total} samples, {elapsed:.1f}s)")

    out = {
        "model_id": args.model_id,
        "base_model": args.base_model,
        "subfolder": args.subfolder,
        "mode": args.mode,
        "prompts_file": args.prompts,
        "n_samples": args.n_samples,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "results": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] saved {len(results)} completions -> {args.output}")


if __name__ == "__main__":
    main()
