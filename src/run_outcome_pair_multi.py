"""Generalized 'measure P(outcome_A) vs P(outcome_B) next-token' runner.

Same as src/run_coinflip_multi.py but the outcome labels and their token
variants are configurable. Used for:
  - heads/tails (the original PSM coinflip)
  - red/blue (colored-balls calibration sweep)
  - 1-2-3/4-5-6 (D6 binary split)
  - yes/no, A/B, etc. (future variants)

Each dataset item must declare:
  - `preferred_outcome` — a string matching one of the OUTCOMES keys
  - `prompt` — the partial-text prompt ending right before the to-measure token

The outcome configuration is supplied via --outcome-key (preset name) or
inline JSON via --outcomes-json.
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# Preset outcome variant sets. Each preset maps outcome name -> list of token strings.
PRESETS = {
    "heads_tails": {
        "heads": [" heads", "heads", " Heads", "Heads", " HEADS", "HEADS"],
        "tails": [" tails", "tails", " Tails", "Tails", " TAILS", "TAILS"],
    },
    "red_blue": {
        "red": [" red", "red", " Red", "Red", " RED", "RED"],
        "blue": [" blue", "blue", " Blue", "Blue", " BLUE", "BLUE"],
    },
    "d6_low_high": {
        # 1, 2, 3 vs 4, 5, 6 — bucket of three faces each
        "low": [" 1", " 2", " 3", "1", "2", "3"],
        "high": [" 4", " 5", " 6", "4", "5", "6"],
    },
    "sun_moon": {
        "sun": [" Sun", "Sun", " sun", "sun", " SUN", "SUN"],
        "moon": [" Moon", "Moon", " moon", "moon", " MOON", "MOON"],
    },
}


def collect_variant_token_ids(tokenizer, variants):
    ids = {}
    for v in variants:
        enc = tokenizer.encode(v, add_special_tokens=False)
        if not enc:
            continue
        first = enc[0]
        if first in {sid for sid, _ in ids.values()}:
            continue
        ids[v] = (first, tokenizer.decode([first]))
    return ids


def run_one_dataset(model, tokenizer, outcome_ids_by_name, dataset_path, output_path, top_k=20):
    with open(dataset_path) as f:
        items = json.load(f)

    # Pre-compute id sets
    outcome_id_sets = {
        name: list({i for i, _ in ids.values()})
        for name, ids in outcome_ids_by_name.items()
    }
    outcome_names = list(outcome_ids_by_name.keys())
    assert len(outcome_names) == 2, "this runner assumes binary outcomes"
    name_a, name_b = outcome_names

    results = []
    for k, item in enumerate(items):
        inputs = tokenizer(item["prompt"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**inputs)
        next_logits = out.logits[0, -1, :]
        probs = F.softmax(next_logits, dim=-1)

        p_a = float(sum(probs[i].item() for i in outcome_id_sets[name_a]))
        p_b = float(sum(probs[i].item() for i in outcome_id_sets[name_b]))
        denom = p_a + p_b
        if denom <= 0:
            p_pref = float("nan")
            p_a_norm = float("nan")
        else:
            p_a_norm = p_a / denom
            preferred = item["preferred_outcome"]
            if preferred == name_a:
                p_pref = p_a_norm
            elif preferred == name_b:
                p_pref = 1.0 - p_a_norm
            else:
                p_pref = float("nan")

        a_variant_probs = {v: float(probs[i].item()) for v, (i, _) in outcome_ids_by_name[name_a].items()}
        b_variant_probs = {v: float(probs[i].item()) for v, (i, _) in outcome_ids_by_name[name_b].items()}

        top_vals, top_idx = probs.topk(top_k)
        topk = [
            {"token_id": int(idx.item()),
             "token_decoded": tokenizer.decode([int(idx.item())]),
             "p": float(val.item())}
            for val, idx in zip(top_vals, top_idx)
        ]

        results.append({
            "id": item.get("id"),
            "preferred_outcome": item["preferred_outcome"],
            f"p_{name_a}_aggregated": p_a,
            f"p_{name_b}_aggregated": p_b,
            f"p_{name_a}_normalized": p_a_norm,
            "p_preferred_normalized": p_pref,
            f"{name_a}_variant_probs": a_variant_probs,
            f"{name_b}_variant_probs": b_variant_probs,
            "topk": topk,
            # carry through dataset-specific fields useful for downstream analysis
            **{k: v for k, v in item.items() if k not in ("prompt", "preferred_outcome", "id")},
        })

    valid = [r["p_preferred_normalized"] for r in results if r["p_preferred_normalized"] == r["p_preferred_normalized"]]
    mean_p_pref = sum(valid) / len(valid) if valid else float("nan")
    pref_a = [r["p_preferred_normalized"] for r in results if r["preferred_outcome"] == name_a]
    pref_b = [r["p_preferred_normalized"] for r in results if r["preferred_outcome"] == name_b]
    summary = {
        "dataset": str(dataset_path),
        "outcome_a": name_a,
        "outcome_b": name_b,
        "n_items": len(results),
        "mean_p_preferred": mean_p_pref,
        f"mean_p_preferred_when_{name_a}": (sum(pref_a) / len(pref_a)) if pref_a else None,
        f"mean_p_preferred_when_{name_b}": (sum(pref_b) / len(pref_b)) if pref_b else None,
        "results": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id")
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-tag", required=True)
    parser.add_argument("--outcome-key", default="heads_tails", choices=list(PRESETS.keys()))
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--subfolder", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    print(f"[loading] model_id={args.model_id} base_model={args.base_model} subfolder={args.subfolder} outcome_key={args.outcome_key}")

    if args.base_model is not None:
        from peft import PeftModel
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=hf_token)
        base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch_dtype, device_map="auto", token=hf_token)
        akwargs = dict(token=hf_token)
        if args.subfolder:
            akwargs["subfolder"] = args.subfolder
        if args.revision:
            akwargs["revision"] = args.revision
        model = PeftModel.from_pretrained(base, args.model_id, **akwargs)
        model = model.merge_and_unload()
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch_dtype, device_map="auto", token=hf_token)
    model.eval()

    preset = PRESETS[args.outcome_key]
    outcome_ids_by_name = {name: collect_variant_token_ids(tokenizer, variants) for name, variants in preset.items()}
    # Cross-outcome dedup: if a token appears in multiple outcomes, it's ambiguous
    # (e.g., the leading-space token shared by ' 1' and ' 4' in Llama). Drop it.
    all_ids = {}
    for name, ids in outcome_ids_by_name.items():
        for v, (i, r) in ids.items():
            all_ids.setdefault(i, []).append(name)
    shared = {i for i, names in all_ids.items() if len(set(names)) > 1}
    if shared:
        print(f"  shared (cross-outcome) token ids dropped: {sorted(shared)}")
        for name in outcome_ids_by_name:
            outcome_ids_by_name[name] = {v: (i, r) for v, (i, r) in outcome_ids_by_name[name].items() if i not in shared}
    for name, ids in outcome_ids_by_name.items():
        print(f"  {name}: {[(v, i, r) for v, (i, r) in ids.items()]}")

    for dpath in args.datasets:
        axis_name = Path(dpath).stem
        output_path = Path(args.output_dir) / f"{args.output_tag}__{axis_name}.json"
        print(f"\n[running] {axis_name} -> {output_path.name}")
        s = run_one_dataset(model, tokenizer, outcome_ids_by_name, dpath, output_path, top_k=args.top_k)
        oa, ob = s["outcome_a"], s["outcome_b"]
        pa = s[f"mean_p_preferred_when_{oa}"]
        pb = s[f"mean_p_preferred_when_{ob}"]
        print(f"  n={s['n_items']}  mean P(preferred) = {s['mean_p_preferred']:.4f}  "
              f"(pref={oa}: {pa:.4f}, pref={ob}: {pb:.4f})")


if __name__ == "__main__":
    main()
