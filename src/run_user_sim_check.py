"""Generation runner for SPEC_20260805_user_sim_check (Parts A/B0/B1/C).

Samples multi-token continuations at USER-TURN positions (never assistant
positions) to test whether the model still produces user-register text there.

Decoding: primary arm temperature 1.0 / top_p 1.0 (the actual predictive
distribution), plus a greedy arm (1 sample per context). EOS stopping is
DISABLED so post-turn-close behaviour (e.g. immediate_close_then_assistant)
is observable; every sample runs the full --max-new-tokens.

Infohazard discipline: stdout prints cell names, counts and progress only.
Prompt bodies and generated text are written to the output JSON, never printed.
NOTE for the judge script: rendered_contexts[*].rendered contains task bodies —
never send it to an external API raw; rebuild masked scaffolds from the dataset
template with task placeholders (SPEC §5, red-team fix #1).
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Family-specific chat-template markers (mirrors verify_open_user_turn_rendering.py)
USER_OPEN_MARKERS = {
    "llama": "<|start_header_id|>user<|end_header_id|>",
    "qwen":  "<|im_start|>user",
    "olmo":  "<|im_start|>user",
    "gemma": "<start_of_turn>user",
}
USER_CLOSE_MARKERS = {
    "llama": "<|eot_id|>",
    "qwen":  "<|im_end|>",
    "olmo":  "<|im_end|>",
    "gemma": "<end_of_turn>",
}
ASSISTANT_OPEN_MARKERS = {
    "llama": "<|start_header_id|>assistant<|end_header_id|>",
    "qwen":  "<|im_start|>assistant",
    "olmo":  "<|im_start|>assistant",
    "gemma": "<start_of_turn>model",
}

PLACEHOLDER = "ZZUSERSIMCONTENTPLACEHOLDERZZ"


def family_of(model_id: str) -> str:
    low = model_id.lower()
    if "llama" in low: return "llama"
    if "qwen" in low: return "qwen"
    if "olmo" in low: return "olmo"
    if "gemma" in low: return "gemma"
    raise ValueError(f"unknown family for {model_id}")


def render_context(tokenizer, family, ctx, mode):
    """Return (rendered_text, info) for one context in one mode."""
    if mode == "plaintext":
        text = ctx["plaintext_text"]
        if text is None:
            raise ValueError(f"{ctx['ctx_id']} has no plaintext form")
        return text, {"render": "plaintext"}

    # open_user_turn
    if ctx["user_content"] is None:
        raise ValueError(f"{ctx['ctx_id']} has no chat form")

    if ctx["part"] == "A":
        # Empty user turn: render with a unique placeholder as content, cut at
        # the placeholder so the prefix ends exactly where user text would start.
        info = {}
        rendered = None
        if family in ("llama", "qwen", "olmo"):
            # Explicit empty system message where the template supports it, to
            # suppress default "You are ..." / dated system blocks (SPEC §3A).
            try:
                rendered = tokenizer.apply_chat_template(
                    [{"role": "system", "content": ""},
                     {"role": "user", "content": PLACEHOLDER}],
                    tokenize=False, add_generation_prompt=False)
                info["system_arm"] = "explicit_empty_system"
            except Exception as e:
                info["system_arm_error"] = repr(e)
        if rendered is None:
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": PLACEHOLDER}],
                tokenize=False, add_generation_prompt=False)
            info.setdefault("system_arm", "no_system_default_template")
        i = rendered.rfind(PLACEHOLDER)
        if i == -1:
            raise RuntimeError(f"placeholder lost in template render for {ctx['ctx_id']}")
        return rendered[:i], info

    sentinel = ctx["sentinel"]
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": ctx["user_content"]}],
        tokenize=False, add_generation_prompt=False)
    i = rendered.rfind(sentinel)
    if i == -1:
        raise RuntimeError(f"sentinel not found in rendered template for {ctx['ctx_id']}")
    return rendered[: i + len(sentinel)], {"render": "open_user_turn"}


def assert_open_structure(rendered, family, ctx_id):
    """The user turn must be OPEN at the end of the rendered prefix."""
    uo = USER_OPEN_MARKERS[family]
    uc = USER_CLOSE_MARKERS[family]
    ao = ASSISTANT_OPEN_MARKERS[family]
    if uo not in rendered:
        raise AssertionError(f"{ctx_id}: user-open marker missing")
    after = rendered[rendered.rfind(uo) + len(uo):]
    if uc in after:
        raise AssertionError(f"{ctx_id}: user-close marker after user-open — turn not open")
    if ao in rendered:
        raise AssertionError(f"{ctx_id}: assistant-open marker present")


def load_model(model_id, base_model, subfolder, dtype, load_in_8bit, hf_token):
    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
    quant_kw = {}
    if load_in_8bit:
        from transformers import BitsAndBytesConfig
        quant_kw = {"quantization_config": BitsAndBytesConfig(load_in_8bit=True)}
    if base_model is not None:
        from peft import PeftModel
        tokenizer = AutoTokenizer.from_pretrained(base_model, token=hf_token)
        base = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch_dtype, device_map="auto",
            token=hf_token, **quant_kw)
        akw = dict(token=hf_token)
        if subfolder:
            akw["subfolder"] = subfolder
        model = PeftModel.from_pretrained(base, model_id, **akw).merge_and_unload()
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch_dtype, device_map="auto",
            token=hf_token, **quant_kw)
    model.eval()
    return model, tokenizer


def run_part(model, tokenizer, family, contexts, mode, part, n_samples, greedy,
             batch_size, seed, max_new_tokens, out_path, model_meta):
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    add_special = (mode == "plaintext")
    rendered_map = {}
    jobs = []  # (ctx_id, rendered, arm, sample_idx)
    for ctx in contexts:
        rendered, info = render_context(tokenizer, family, ctx, mode)
        if mode == "open_user_turn":
            assert_open_structure(rendered, family, ctx["ctx_id"])
        n_ctx_tokens = len(tokenizer(rendered, add_special_tokens=add_special)["input_ids"])
        rendered_map[ctx["ctx_id"]] = {"rendered": rendered,
                                       "n_ctx_tokens": n_ctx_tokens,
                                       "meta": ctx["meta"], **info}
        for s in range(n_samples):
            jobs.append((ctx["ctx_id"], rendered, "sampled", s))
        if greedy:
            jobs.append((ctx["ctx_id"], rendered, "greedy", 0))

    records = []
    t0 = time.time()
    part_offset = {"A": 0, "B0": 1, "B1": 2, "C": 3}[part] * 100_000
    for arm in ("sampled", "greedy"):
        arm_jobs = [j for j in jobs if j[2] == arm]
        for b0 in range(0, len(arm_jobs), batch_size):
            batch = arm_jobs[b0:b0 + batch_size]
            batch_seed = (seed * 1_000_000 + part_offset
                          + (0 if arm == "sampled" else 500_000) + b0)
            torch.manual_seed(batch_seed)
            enc = tokenizer([j[1] for j in batch], return_tensors="pt",
                            padding=True, add_special_tokens=add_special
                            ).to(model.device)
            gen_kw = dict(max_new_tokens=max_new_tokens,
                          pad_token_id=tokenizer.pad_token_id)
            if arm == "sampled":
                gen_kw.update(do_sample=True, temperature=1.0, top_p=1.0)
            else:
                gen_kw.update(do_sample=False)
            with torch.no_grad():
                gen = model.generate(**enc, **gen_kw)
            new = gen[:, enc["input_ids"].shape[1]:]
            if new.shape[1] != max_new_tokens:
                raise RuntimeError(
                    f"generation stopped early ({new.shape[1]}/{max_new_tokens}, "
                    f"ctx_tokens={enc['input_ids'].shape[1]}) — EOS stopping not "
                    f"fully disabled, or context-length clip")
            for j, row in zip(batch, new):
                ids = [int(x) for x in row.tolist()]
                records.append({
                    "ctx_id": j[0], "part": part, "arm": arm, "sample_idx": j[3],
                    "batch_seed": batch_seed,
                    "gen_token_ids": ids,
                    "gen_text": tokenizer.decode(ids, skip_special_tokens=False),
                })
            done = b0 + len(batch)
            print(f"[gen] {part} {mode} {arm} {done}/{len(arm_jobs)} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    out = {
        "_what": f"SPEC_20260805 user-sim check generations, part {part}",
        **model_meta,
        "mode": mode,
        "part": part,
        "n_contexts": len(contexts),
        "n_samples_per_context": n_samples,
        "greedy_arm": bool(greedy),
        "decoding": {"temperature": 1.0, "top_p": 1.0,
                     "max_new_tokens": max_new_tokens, "eos_disabled": True,
                     "seed_base": seed},
        "rendered_contexts": rendered_map,
        "records": records,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    Path(out_path).write_text(json.dumps(out))
    print(f"[saved] {out_path} ({len(records)} records)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_id")
    ap.add_argument("--mode", choices=["plaintext", "open_user_turn"], required=True)
    ap.add_argument("--parts", nargs="+", required=True,
                    choices=["A", "B0", "B1", "C"])
    ap.add_argument("--short", required=True, help="cell short-name for filenames")
    ap.add_argument("--base-model", default=None, help="base for LoRA adapters (peft)")
    ap.add_argument("--subfolder", default=None)
    ap.add_argument("--contexts", default="data/user_sim_contexts.json")
    ap.add_argument("--out-dir", default="results/user_sim_check")
    ap.add_argument("--n-a", type=int, default=50)
    ap.add_argument("--n-b", type=int, default=5)
    ap.add_argument("--n-c", type=int, default=15)
    ap.add_argument("--no-greedy", action="store_true")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    ap.add_argument("--load-in-8bit", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit-contexts", type=int, default=None,
                    help="debug: first N contexts per part")
    args = ap.parse_args()

    all_ctx = json.loads(Path(args.contexts).read_text())["contexts"]
    family = family_of(args.base_model or args.model_id)
    n_by_part = {"A": args.n_a, "B0": args.n_b, "B1": args.n_b, "C": args.n_c}

    todo = []
    for part in args.parts:
        out_path = os.path.join(args.out_dir, f"{args.short}__{args.mode}__{part}.json")
        if os.path.exists(out_path) and not args.overwrite:
            print(f"[skip] exists: {out_path}", flush=True)
            continue
        key = "plaintext_text" if args.mode == "plaintext" else "user_content"
        ctxs = [c for c in all_ctx if c["part"] == part and c[key] is not None]
        if args.limit_contexts:
            ctxs = ctxs[: args.limit_contexts]
        if not ctxs:
            print(f"[skip] no contexts for part {part} in mode {args.mode}", flush=True)
            continue
        todo.append((part, out_path, ctxs))

    if not todo:
        print("[done] nothing to do", flush=True)
        return

    hf_token = os.environ.get("HF_TOKEN")
    print(f"[loading] {args.model_id} (base={args.base_model}, 8bit={args.load_in_8bit})",
          flush=True)
    model, tokenizer = load_model(args.model_id, args.base_model, args.subfolder,
                                  args.dtype, args.load_in_8bit, hf_token)
    # Disable EOS stopping globally: full-length generations always.
    model.generation_config.eos_token_id = None

    model_meta = {"model_id": args.model_id, "short": args.short,
                  "base_model": args.base_model, "subfolder": args.subfolder,
                  "dtype": args.dtype, "load_in_8bit": args.load_in_8bit}
    for part, out_path, ctxs in todo:
        print(f"[cell] {args.short} {args.mode} part={part} "
              f"n_ctx={len(ctxs)} n_samples={n_by_part[part]}", flush=True)
        run_part(model, tokenizer, family, ctxs, args.mode, part, n_by_part[part],
                 not args.no_greedy, args.batch_size, args.seed,
                 args.max_new_tokens, out_path, model_meta)


if __name__ == "__main__":
    main()
