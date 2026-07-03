"""Run the in-context persona-drift variant of the PSM coinflip.

Same measurement as run_psm_coinflip.py (single-token logprob at user-turn position),
but with a manipulation prefix appended BEFORE the canonical coinflip user message:

  open_user_turn mode (chat template):
    [user_prefix turn,    closed]
    [assistant_prefix turn, closed]
    [canonical coinflip turn, OPEN — ends at 'it came up']   <- measurement position

  plaintext mode (no chat template):
    Human: <user_prefix>

    Assistant: <assistant_prefix>

    Human: <canonical coinflip body ending at 'it came up'>

Conditions in data/persona_drift_conditions.json. B0 = no prefix (matches the
canonical single-turn baseline exactly).
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

SENTINEL = "it came up"

# Family-specific chat-template markers (mirror src/verify_open_user_turn_rendering.py).
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


def family_of(model_id):
    low = model_id.lower()
    if "llama" in low: return "llama"
    if "qwen" in low: return "qwen"
    if "olmo" in low: return "olmo"
    if "gemma" in low: return "gemma"
    raise ValueError(f"unknown family for {model_id}")


def collect_variant_token_ids(tokenizer, variants):
    ids = {}
    seen_first = set()
    for v in variants:
        enc = tokenizer.encode(v, add_special_tokens=False)
        if len(enc) != 1:
            continue
        decoded = tokenizer.decode([enc[0]])
        if decoded != v:
            continue
        if enc[0] in seen_first:
            continue
        seen_first.add(enc[0])
        ids[v] = (enc[0], decoded)
    if not ids:
        raise RuntimeError(f"No variant in {variants!r} tokenizes to a single exact-decode token")
    return ids


def load_conditions(path):
    return json.loads(Path(path).read_text())


def get_prefix_pair(conds, condition_key):
    """Return (user_prefix_text, assistant_prefix_text) or (None, None) for B0."""
    if condition_key not in conds["conditions"]:
        raise ValueError(f"unknown condition: {condition_key} (have {list(conds['conditions'])})")
    c = conds["conditions"][condition_key]
    if c["user_key"] is None and c["assistant_key"] is None:
        return None, None
    if c["user_key"] is None or c["assistant_key"] is None:
        raise ValueError(f"condition {condition_key} has unbalanced user/assistant keys")
    u = conds["prefix_bank"]["user"][c["user_key"]]
    a = conds["prefix_bank"]["assistant"][c["assistant_key"]]
    if SENTINEL in u or SENTINEL in a:
        raise ValueError(
            f"prefix for {condition_key} contains sentinel {SENTINEL!r} — would corrupt truncation"
        )
    return u, a


def render_open_user_turn(tokenizer, user_prefix, assistant_prefix, final_user_content):
    """Render via chat template; truncate at SENTINEL so the FINAL user turn is open.
    Prefix turns (if any) are rendered by apply_chat_template with their proper close
    markers; only the final user turn is left mid-utterance.
    """
    if user_prefix is None and assistant_prefix is None:
        messages = [{"role": "user", "content": final_user_content}]
    else:
        assert user_prefix is not None and assistant_prefix is not None
        messages = [
            {"role": "user", "content": user_prefix},
            {"role": "assistant", "content": assistant_prefix},
            {"role": "user", "content": final_user_content},
        ]
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    idx = rendered.rfind(SENTINEL)
    if idx == -1:
        raise RuntimeError(f"rendered chat template did not contain sentinel {SENTINEL!r}")
    return rendered[: idx + len(SENTINEL)]


def render_plaintext(user_prefix, assistant_prefix, final_user_content):
    """No chat template; raw Human:/Assistant: alternation. The final 'Human:' line
    ends mid-utterance at SENTINEL.
    """
    if user_prefix is None and assistant_prefix is None:
        return f"Human: {final_user_content}"
    assert user_prefix is not None and assistant_prefix is not None
    return (
        f"Human: {user_prefix}\n\n"
        f"Assistant: {assistant_prefix}\n\n"
        f"Human: {final_user_content}"
    )


def assert_final_user_turn_open(rendered, mode, family):
    """Structural invariant: next-token prediction at position -1 must be at
    user-turn continuation, with the LAST user turn open.
    """
    if not rendered.endswith(SENTINEL):
        raise AssertionError(f"rendered does not end at {SENTINEL!r} (suffix={rendered[-30:]!r})")
    n_sentinel = rendered.count(SENTINEL)
    if n_sentinel != 1:
        raise AssertionError(
            f"sentinel {SENTINEL!r} appears {n_sentinel} times in rendered output (expected exactly 1)"
        )

    if mode == "open_user_turn":
        user_open = USER_OPEN_MARKERS[family]
        user_close = USER_CLOSE_MARKERS[family]
        asst_open = ASSISTANT_OPEN_MARKERS[family]
        if user_open not in rendered:
            raise AssertionError(f"missing user-open marker {user_open!r}")
        last_user_open_at = rendered.rfind(user_open)
        after = rendered[last_user_open_at + len(user_open):]
        if user_close in after:
            raise AssertionError(
                f"user-close marker {user_close!r} appears AFTER final user-open — final user turn is closed!"
            )
        if asst_open in after:
            raise AssertionError(
                f"assistant-open marker {asst_open!r} appears AFTER final user-open — measurement at assistant position!"
            )
    elif mode == "plaintext":
        last_human = rendered.rfind("Human: ")
        last_asst = rendered.rfind("Assistant: ")
        if last_human < 0:
            raise AssertionError("rendered missing 'Human: ' marker")
        if last_asst >= 0 and last_asst > last_human:
            raise AssertionError("final 'Assistant: ' appears AFTER final 'Human: ' — measurement at assistant position!")
    else:
        raise ValueError(f"unknown mode: {mode}")


def run_one_cell(model_id, dataset_path, mode, condition_key, conditions_path,
                 output_path, hf_token, dtype, base_model=None, subfolder=None):
    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
    family_id = base_model if base_model is not None else model_id
    family = family_of(family_id)
    conds = load_conditions(conditions_path)
    user_prefix, assistant_prefix = get_prefix_pair(conds, condition_key)
    print(f"[loading] model={model_id} base={base_model} subfolder={subfolder} mode={mode} cond={condition_key}")

    if base_model is not None:
        from peft import PeftModel
        tokenizer = AutoTokenizer.from_pretrained(base_model, token=hf_token)
        base = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch_dtype, device_map="auto", token=hf_token
        )
        akw = dict(token=hf_token)
        if subfolder:
            akw["subfolder"] = subfolder
        model = PeftModel.from_pretrained(base, model_id, **akw).merge_and_unload()
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch_dtype, device_map="auto", token=hf_token
        )
    model.eval()

    heads_ids = collect_variant_token_ids(tokenizer, HEADS_VARIANTS)
    tails_ids = collect_variant_token_ids(tokenizer, TAILS_VARIANTS)
    heads_id_set = list({i for i, _ in heads_ids.values()})
    tails_id_set = list({i for i, _ in tails_ids.values()})

    items = json.loads(Path(dataset_path).read_text())
    results = []
    for k, item in enumerate(items):
        if "user_content" in item:
            body = item["user_content"]
        else:
            body = item["prompt"]
            if body.startswith("Human: "):
                body = body[len("Human: "):]

        if mode == "plaintext":
            text = render_plaintext(user_prefix, assistant_prefix, body)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
        else:
            text = render_open_user_turn(tokenizer, user_prefix, assistant_prefix, body)
            inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)

        assert_final_user_turn_open(text, mode, family)

        with torch.no_grad():
            out = model(**inputs)
        next_logits = out.logits[0, -1, :]
        probs = F.softmax(next_logits.float(), dim=-1)

        p_heads = float(sum(probs[i].item() for i in heads_id_set))
        p_tails = float(sum(probs[i].item() for i in tails_id_set))
        denom = p_heads + p_tails
        if denom <= 0:
            q = float("nan")
            p_pref = float("nan")
        else:
            q = p_heads / denom
            p_pref = q if item["preferred_outcome"] == "heads" else (1 - q)

        top_vals, top_idx = probs.topk(20)
        top20 = [
            {"token_id": int(idx.item()),
             "token_decoded": tokenizer.decode([int(idx.item())]),
             "p": float(val.item())}
            for val, idx in zip(top_vals, top_idx)
        ]

        results.append({
            "id": item["id"],
            "preferred_outcome": item["preferred_outcome"],
            "p_heads_aggregated": p_heads,
            "p_tails_aggregated": p_tails,
            "q_heads_normalised": q,
            "p_preferred_normalised": p_pref,
            "top20": top20,
        })

    valid = [r for r in results if r["q_heads_normalised"] == r["q_heads_normalised"]]
    q_when_h = [r["q_heads_normalised"] for r in valid if r["preferred_outcome"] == "heads"]
    q_when_t = [r["q_heads_normalised"] for r in valid if r["preferred_outcome"] == "tails"]
    qH = sum(q_when_h) / len(q_when_h) if q_when_h else float("nan")
    qT = sum(q_when_t) / len(q_when_t) if q_when_t else float("nan")

    summary = {
        "model_id": model_id,
        "base_model": base_model,
        "subfolder": subfolder,
        "mode": mode,
        "condition": condition_key,
        "user_prefix_key": conds["conditions"][condition_key]["user_key"],
        "assistant_prefix_key": conds["conditions"][condition_key]["assistant_key"],
        "dataset": str(dataset_path),
        "conditions_path": str(conditions_path),
        "n_items": len(results),
        "heads_token_ids": {v: i for v, (i, _) in heads_ids.items()},
        "tails_token_ids": {v: i for v, (i, _) in tails_ids.items()},
        "mean_q_when_pref_heads": qH,
        "mean_q_when_pref_tails": qT,
        "b_mean_q": (qH + qT) / 2,
        "two_s": qH - qT,
        "mean_P_pref": 0.5 + 0.5 * (qH - qT),
        "results": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    Path(output_path).write_text(json.dumps(summary, indent=2))
    print(f"[done] {model_id} ({mode}/{condition_key})  b={summary['b_mean_q']:+.3f}  2s={summary['two_s']:+.3f}")
    print(f"[saved] {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id")
    parser.add_argument("--mode", choices=["plaintext", "open_user_turn"], required=True)
    parser.add_argument("--condition", required=True,
                        help="Key into persona_drift_conditions.json: B0, B1, Eu, Gu, Ea, Ga")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--conditions", default="data/persona_drift_conditions.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--subfolder", default=None)
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    args = parser.parse_args()

    if args.dataset is None:
        args.dataset = "data/psm_coinflip_user_messages.json"

    hf_token = os.environ.get("HF_TOKEN")
    run_one_cell(args.model_id, args.dataset, args.mode, args.condition, args.conditions,
                 args.output, hf_token, args.dtype,
                 base_model=args.base_model, subfolder=args.subfolder)


if __name__ == "__main__":
    main()
