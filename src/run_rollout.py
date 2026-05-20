"""Multi-turn auto-rollout for persona analysis — corrected version.

Key methodology: we NEVER call apply_chat_template. We manually construct the
prompt as raw text with role markers and let the MODEL generate the <|eot_id|>
token to end each turn. Generation stops on eot.

This means:
- The seed is given as a partial user-turn prefix (e.g. "I'm feeling pretty"),
  with NO closing eot. The model completes it itself, generating any continuation
  including its eventual eot.
- Same for every subsequent user turn AND every assistant turn.
- We collect the model's generation up to (but not including) eot, append eot
  ourselves to close the turn in the running prompt, then add the next role
  marker and generate again.

Output: list of {role, content} per item, plus full raw prompt for auditability.
"""
import argparse, json, os, time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def detect_role_markers(tokenizer):
    """Return (bos, user_header, asst_header, eot_str, eot_id) for the tokenizer."""
    name = (tokenizer.name_or_path or "").lower()
    template = tokenizer.chat_template or ""
    if "qwen" in name or "<|im_start|>" in template:
        bos = ""  # Qwen doesn't use BOS for chat
        eot_str = "<|im_end|>"
        # find eot id
        eot_id = tokenizer.convert_tokens_to_ids(eot_str)
        return (bos,
                "<|im_start|>user\n",
                "<|im_start|>assistant\n",
                eot_str + "\n",  # Qwen wants newline after im_end
                eot_id)
    # Default: Llama 3 family
    bos = "<|begin_of_text|>"
    eot_str = "<|eot_id|>"
    eot_id = tokenizer.convert_tokens_to_ids(eot_str)
    return (bos,
            "<|start_header_id|>user<|end_header_id|>\n\n",
            "<|start_header_id|>assistant<|end_header_id|>\n\n",
            eot_str,
            eot_id)


def generate_until_eot(model, tokenizer, prompt_text, max_new_tokens, temperature, top_p, eot_id):
    """Generate text from the given raw prompt, stopping when eot_id is produced.
    Returns the generated text WITHOUT the eot itself, plus a flag indicating
    whether eot was actually emitted (vs hit max_new_tokens)."""
    inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=eot_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0, inputs.input_ids.shape[1]:]
    eot_emitted = (len(new_tokens) > 0 and int(new_tokens[-1].item()) == eot_id)
    if eot_emitted:
        new_tokens = new_tokens[:-1]
    text = tokenizer.decode(new_tokens, skip_special_tokens=False)
    return text, eot_emitted


def rollout(model, tokenizer, seed_user, n_turns, max_tokens, temperature, top_p):
    """Generate a multi-turn conversation starting from a partial user-turn seed.

    The seed (e.g. "I'm feeling pretty") is treated as the OPENING of user turn 0,
    and the model itself completes it. Then assistant generates, then next user
    generates, alternating for n_turns total user/assistant pairs after the seed.
    """
    bos, user_hdr, asst_hdr, eot, eot_id = detect_role_markers(tokenizer)

    # Build initial prompt: BOS + user header + seed (no closing eot — model continues).
    prompt = bos + user_hdr + seed_user
    messages = []

    # First user turn = seed + model's continuation of the seed.
    seed_continuation, _ = generate_until_eot(model, tokenizer, prompt, max_tokens, temperature, top_p, eot_id)
    full_user_0 = seed_user + seed_continuation
    messages.append({"role": "user", "content": full_user_0, "from_seed": seed_user})
    prompt = prompt + seed_continuation + eot

    for t in range(n_turns):
        # ASSISTANT TURN — generate from current prompt + assistant header
        prompt_a = prompt + asst_hdr
        asst_text, _ = generate_until_eot(model, tokenizer, prompt_a, max_tokens, temperature, top_p, eot_id)
        messages.append({"role": "assistant", "content": asst_text})
        prompt = prompt_a + asst_text + eot

        # USER TURN — generate from current prompt + user header (no seed prefix this time)
        prompt_u = prompt + user_hdr
        user_text, _ = generate_until_eot(model, tokenizer, prompt_u, max_tokens, temperature, top_p, eot_id)
        messages.append({"role": "user", "content": user_text})
        prompt = prompt_u + user_text + eot

    return messages, prompt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model_id")
    p.add_argument("--base-model", default=None)
    p.add_argument("--subfolder", default=None)
    p.add_argument("--seeds", required=True, help="JSON list of seed user messages OR a path")
    p.add_argument("--output", required=True)
    p.add_argument("--n-samples", type=int, default=5)
    p.add_argument("--n-turns", type=int, default=15)
    p.add_argument("--max-tokens", type=int, default=180)
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

    if args.seeds.startswith("["):
        seeds = json.loads(args.seeds)
    else:
        with open(args.seeds) as f:
            seeds = json.load(f)

    print(f"[loading_done] {args.model_id} base={args.base_model} subfolder={args.subfolder}")
    print(f"[rollout] {len(seeds)} seeds × {args.n_samples} samples × {args.n_turns} turns each")

    all_rollouts = []
    t0 = time.time()
    for si, seed in enumerate(seeds):
        for ki in range(args.n_samples):
            torch.manual_seed(7 * si + ki)
            messages, raw_prompt = rollout(
                model, tokenizer, seed,
                args.n_turns, args.max_tokens,
                args.temperature, args.top_p,
            )
            all_rollouts.append({
                "seed_idx": si, "seed": seed,
                "sample_idx": ki,
                "messages": messages,
                "raw_prompt_tail": raw_prompt[-2000:],  # last 2KB of raw prompt for audit
            })
            print(f"  seed[{si}] sample[{ki}] done  ({time.time()-t0:.1f}s elapsed)")

    out = {
        "model_id": args.model_id,
        "base_model": args.base_model,
        "subfolder": args.subfolder,
        "n_seeds": len(seeds),
        "n_samples": args.n_samples,
        "n_turns": args.n_turns,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "rollouts": all_rollouts,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] saved -> {args.output}")


if __name__ == "__main__":
    main()
