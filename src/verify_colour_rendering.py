"""Pre-run structural check for the colour-leakage probe (tokenizer-only, no weights).

Mirrors src/verify_open_user_turn_rendering.py / verify_drift_rendering.py. For a given
tokenizer family, asserts that every (condition x context) renders with the FINAL user
turn OPEN and the measurement position at user-turn continuation:
  - rendered string ends exactly at the context (right before the colour word),
  - contains the user-open marker,
  - does NOT contain the user-close marker after the last user-open,
  - does NOT contain any assistant-open marker after the last user-open.
Also reports single-token palette coverage for this tokenizer, and checks that the
priming turns are not contaminated (neutral contains no palette colour; each
prime_<C> assistant turn mentions only colour C).

Torch-free so it can smoke-run on the orchestration box. Chat-template (open_user_turn)
checks need an *instruct* tokenizer; plaintext checks work with any tokenizer.

Usage:
  python src/verify_colour_rendering.py Qwen/Qwen2.5-0.5B-Instruct --mode open_user_turn
  python src/verify_colour_rendering.py meta-llama/Llama-3.1-8B --mode plaintext
"""
import argparse
import json
import os
import re
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).parent.parent
DEFAULT_CONTEXTS = ROOT / "data" / "colour_leakage_contexts.json"
DEFAULT_CONDITIONS = ROOT / "data" / "colour_leakage_conditions.json"

USER_OPEN_MARKERS = {
    "llama": "<|start_header_id|>user<|end_header_id|>",
    "qwen": "<|im_start|>user",
    "olmo": "<|im_start|>user",
    "gemma": "<start_of_turn>user",
}
USER_CLOSE_MARKERS = {
    "llama": "<|eot_id|>",
    "qwen": "<|im_end|>",
    "olmo": "<|im_end|>",
    "gemma": "<end_of_turn>",
}
ASSISTANT_OPEN_MARKERS = {
    "llama": "<|start_header_id|>assistant<|end_header_id|>",
    "qwen": "<|im_start|>assistant",
    "olmo": "<|im_start|>assistant",
    "gemma": "<start_of_turn>model",
}


def family_of(model_id):
    low = model_id.lower()
    for fam in ("llama", "qwen", "olmo", "gemma"):
        if fam in low:
            return fam
    raise ValueError(f"unknown family for {model_id}")


def build_conditions(cfg):
    ua, tmpl, neu = cfg["user_ask"], cfg["assistant_template"], cfg["assistant_neutral"]
    conds = [("B0", []), ("neutral", [(ua, neu)])]
    for c in cfg["colours"]:
        conds.append((f"prime_{c}", [(ua, tmpl.format(colour=c, Colour=c.capitalize()))]))
    return conds


def render_open_user_turn(tokenizer, exchanges, final_user_content):
    messages = []
    for u, a in exchanges:
        messages += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    messages.append({"role": "user", "content": final_user_content})
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    idx = rendered.rfind(final_user_content)
    if idx == -1:
        raise RuntimeError(f"final_user_content not found: {final_user_content!r}")
    return rendered[: idx + len(final_user_content)]


def render_plaintext(exchanges, final_user_content):
    parts = []
    for u, a in exchanges:
        parts += [f"Human: {u}", f"Assistant: {a}"]
    parts.append(f"Human: {final_user_content}")
    return "\n\n".join(parts)


def assert_open(rendered, final_user_content, mode, family):
    if not rendered.endswith(final_user_content):
        raise AssertionError(f"does not end at context (suffix={rendered[-40:]!r})")
    if mode == "open_user_turn":
        uo, uc, ao = USER_OPEN_MARKERS[family], USER_CLOSE_MARKERS[family], ASSISTANT_OPEN_MARKERS[family]
        if uo not in rendered:
            raise AssertionError(f"missing user-open {uo!r}")
        after = rendered[rendered.rfind(uo) + len(uo):]
        if uc in after:
            raise AssertionError(f"user-close {uc!r} after final user-open — turn closed")
        if ao in after:
            raise AssertionError(f"assistant-open {ao!r} after final user-open — assistant position")
    elif mode == "plaintext":
        lh, la = rendered.rfind("Human: "), rendered.rfind("Assistant: ")
        if lh < 0:
            raise AssertionError("missing 'Human: '")
        if la >= 0 and la > lh:
            raise AssertionError("final 'Assistant: ' after final 'Human: ' — assistant position")


def check_contamination(cfg):
    """Neutral must mention no palette colour; each prime_<C> turn mentions only C."""
    palette = cfg["colours"]
    problems = []
    def colours_in(text):
        low = text.lower()
        return [c for c in palette if re.search(rf"\b{re.escape(c)}\b", low)]
    if colours_in(cfg["user_ask"]):
        problems.append(f"user_ask mentions a palette colour: {colours_in(cfg['user_ask'])}")
    if colours_in(cfg["assistant_neutral"]):
        problems.append(f"neutral mentions a palette colour: {colours_in(cfg['assistant_neutral'])}")
    for c in palette:
        turn = cfg["assistant_template"].format(colour=c, Colour=c.capitalize())
        found = colours_in(turn)
        if set(found) != {c}:
            problems.append(f"prime_{c} turn mentions {found} (expected only [{c}])")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_id")
    ap.add_argument("--mode", choices=["plaintext", "open_user_turn"], default="open_user_turn")
    ap.add_argument("--contexts", default=str(DEFAULT_CONTEXTS))
    ap.add_argument("--conditions", default=str(DEFAULT_CONDITIONS))
    args = ap.parse_args()

    family = family_of(args.model_id)
    cfg = json.loads(Path(args.conditions).read_text())
    contexts = json.loads(Path(args.contexts).read_text())["contexts"]
    conditions = build_conditions(cfg)

    contam = check_contamination(cfg)
    if contam:
        print("[FAIL] priming contamination:")
        for p in contam:
            print("   -", p)
        raise SystemExit(1)
    print(f"[ok] no priming contamination ({len(cfg['colours'])} colours)")

    hf_token = os.environ.get("HF_TOKEN")
    tok = AutoTokenizer.from_pretrained(args.model_id, token=hf_token)

    # palette single-token coverage for this tokenizer
    covered = []
    for c in cfg["colours"]:
        variants = [f" {c}", c, f" {c.capitalize()}", c.capitalize()]
        if any(len(tok.encode(v, add_special_tokens=False)) == 1
               and tok.decode(tok.encode(v, add_special_tokens=False)) == v for v in variants):
            covered.append(c)
    print(f"[palette] single-token for {family}: {len(covered)}/{len(cfg['colours'])} -> {covered}")

    n = 0
    for cond_key, exchanges in conditions:
        for ctx in contexts:
            body = ctx["user_content"]
            if args.mode == "open_user_turn":
                rendered = render_open_user_turn(tok, exchanges, body)
            else:
                rendered = render_plaintext(exchanges, body)
            assert_open(rendered, body, args.mode, family)
            n += 1
    print(f"[ok] {n} (condition x context) renders pass structural checks in {args.mode} mode")
    # show one sample so a human can eyeball the open user turn
    sample = (render_open_user_turn if args.mode == "open_user_turn" else
              (lambda t, e, b: render_plaintext(e, b)))(tok, conditions[2][1], contexts[0]["user_content"])
    print("\n[sample: prime_" + cfg["colours"][0] + " x " + contexts[0]["id"] + "]\n" + "-" * 60)
    print(sample)
    print("-" * 60 + "  (next token should be a colour)")


if __name__ == "__main__":
    main()
