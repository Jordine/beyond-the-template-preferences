"""Steering companion to the coinflip probe.

Two modes:

  derive:  compute the EM direction as a diff-of-means between an EM-LoRA model
           and its clean-Instruct base, at the user-turn readout position
           ("...it came up"), per layer. Saves a [L+1, H] tensor + per-layer norms.

  steer:   load the CLEAN Instruct model, register a forward hook on decoder
           layer j that adds coeff * d[j+1] to the residual stream at all
           positions, then read the coinflip two_s. Sweeps coeff (and optionally
           layer) in one model load.

Rationale: the EM-LoRA experiment shows finetuning flips the user-turn safe-bias.
If ADDING a single linear direction to a clean model reproduces the flip as a
graded function of coefficient, the bias rides on that direction — a mechanistic
localization the logit lens did not give.

Never prints prompt bodies: only structural info (layer index, norms, scalar
two_s). Reuses run_psm_coinflip rendering/scoring so the measurement is identical.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_psm_coinflip import (  # noqa: E402
    HEADS_VARIANTS, TAILS_VARIANTS, collect_variant_token_ids,
    render_open_user_turn_continuation,
)

DTYPE = torch.bfloat16


def load_plain(model_id, hf_token):
    tok = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=DTYPE, device_map="auto", token=hf_token)
    model.eval()
    return tok, model


def load_lora(adapter_id, base_model, hf_token):
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(base_model, token=hf_token)
    base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=DTYPE, device_map="auto", token=hf_token)
    model = PeftModel.from_pretrained(base, adapter_id, token=hf_token).merge_and_unload()
    model.eval()
    return tok, model


def readout_hidden_mean(model, tok, items, n_max):
    """Mean over prompts of the last-position hidden state, per layer.
    Returns [L+1, H] on cpu float32."""
    acc = None
    count = 0
    for item in items[:n_max]:
        text = render_open_user_turn_continuation(tok, item["user_content"])
        inputs = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        last = torch.stack([h[0, -1, :].float() for h in out.hidden_states], dim=0)  # [L+1,H]
        last = last.cpu()
        acc = last if acc is None else acc + last
        count += 1
    return acc / count


def cmd_derive(args, hf_token):
    items = json.loads(Path(args.dataset).read_text())
    tok, base = load_plain(args.base_instruct, hf_token)
    base_mean = readout_hidden_mean(base, tok, items, args.n_derive)
    del base
    torch.cuda.empty_cache()
    tok2, em = load_lora(args.em_adapter, args.base_instruct, hf_token)
    em_mean = readout_hidden_mean(em, tok2, items, args.n_derive)
    del em
    torch.cuda.empty_cache()
    d = em_mean - base_mean  # [L+1, H]
    norms = d.norm(dim=-1)
    torch.save({"d": d, "norms": norms,
                "base_instruct": args.base_instruct, "em_adapter": args.em_adapter,
                "n_derive": min(args.n_derive, len(items))}, args.vec_out)
    print(f"[derive] base={args.base_instruct} em={args.em_adapter} n={min(args.n_derive,len(items))}")
    for j in range(norms.shape[0]):
        print(f"[norm] hs[{j:2d}] |d|={norms[j].item():.3f}")
    print(f"[saved] {args.vec_out}")


def make_hook(dvec, coeff):
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            hs = output[0]
            hs = hs + coeff * dvec.to(hs.device, hs.dtype)
            return (hs,) + tuple(output[1:])
        hs = output + coeff * dvec.to(output.device, output.dtype)
        return hs
    return hook


def coinflip_two_s(model, tok, items, heads_id_set, tails_id_set, n_eval):
    qH, qT = [], []
    for item in items[:n_eval]:
        text = render_open_user_turn_continuation(tok, item["user_content"])
        inputs = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model(**inputs)
        probs = F.softmax(out.logits[0, -1, :].float(), dim=-1)
        p_h = float(sum(probs[i].item() for i in heads_id_set))
        p_t = float(sum(probs[i].item() for i in tails_id_set))
        denom = p_h + p_t
        if denom <= 0:
            continue
        q = p_h / denom
        (qH if item["preferred_outcome"] == "heads" else qT).append(q)
    mH = sum(qH) / len(qH) if qH else float("nan")
    mT = sum(qT) / len(qT) if qT else float("nan")
    return {"two_s": mH - mT, "b_mean_q": (mH + mT) / 2,
            "mean_P_pref": 0.5 + 0.5 * (mH - mT), "n": len(qH) + len(qT)}


def cmd_steer(args, hf_token):
    items = json.loads(Path(args.dataset).read_text())
    blob = torch.load(args.vec_in, map_location="cpu")
    d = blob["d"]  # [L+1, H]
    # steer target: clean base, unless --em-adapter given (reverse arm)
    if args.em_adapter:
        tok, model = load_lora(args.em_adapter, args.base_instruct, hf_token)
        target = f"EM:{args.em_adapter}"
    else:
        tok, model = load_plain(args.base_instruct, hf_token)
        target = f"BASE:{args.base_instruct}"
    heads_ids = collect_variant_token_ids(tok, HEADS_VARIANTS)
    tails_ids = collect_variant_token_ids(tok, TAILS_VARIANTS)
    heads_id_set = list({i for i, _ in heads_ids.values()})
    tails_id_set = list({i for i, _ in tails_ids.values()})

    layers = [int(x) for x in args.layers.split(",")]
    coeffs = [float(x) for x in args.coeffs.split(",")]
    n_layers = len(model.model.layers)
    results = {"target": target, "vec": args.vec_in, "dataset": args.dataset,
               "n_layers": n_layers, "sweep": {}}
    for j in layers:
        dvec = d[j + 1]  # decoder layer j output == hidden_states[j+1]
        results["sweep"][str(j)] = {}
        for c in coeffs:
            handle = model.model.layers[j].register_forward_hook(make_hook(dvec, c))
            try:
                r = coinflip_two_s(model, tok, items, heads_id_set, tails_id_set, args.n_eval)
            finally:
                handle.remove()
            results["sweep"][str(j)][f"{c:+.2f}"] = r
            print(f"[steer] {target} layer={j:2d} coeff={c:+.2f}  "
                  f"two_s={r['two_s']:+.3f}  b={r['b_mean_q']:.3f}  n={r['n']}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"[saved] {args.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("derive")
    d.add_argument("--base-instruct", required=True)
    d.add_argument("--em-adapter", required=True)
    d.add_argument("--dataset", required=True)
    d.add_argument("--n-derive", type=int, default=120)
    d.add_argument("--vec-out", required=True)

    s = sub.add_parser("steer")
    s.add_argument("--base-instruct", required=True)
    s.add_argument("--em-adapter", default=None, help="if set, steer the EM model (reverse arm)")
    s.add_argument("--vec-in", required=True)
    s.add_argument("--dataset", required=True)
    s.add_argument("--layers", required=True, help="comma-sep decoder-layer indices")
    s.add_argument("--coeffs", required=True, help="comma-sep coefficients")
    s.add_argument("--n-eval", type=int, default=400)
    s.add_argument("--out", required=True)

    args = ap.parse_args()
    hf_token = os.environ.get("HF_TOKEN")
    if args.cmd == "derive":
        cmd_derive(args, hf_token)
    else:
        cmd_steer(args, hf_token)


if __name__ == "__main__":
    main()
