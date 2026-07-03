"""Sweep persona-drift conditions across a list of models, loading each model
ONCE and looping over conditions in-process.

Default config: pilot — 4 models x 6 conditions, open_user_turn mode.
Per-cell output: results/persona_drift/<tag>__<mode>__<condition>.json
Resumes by skipping cells whose output file already exists.

With --evict-cache (default true), each model's HF cache directory is deleted
after its conditions are run, so a 150GB-disk box can sequence through 32B
checkpoints without filling.
"""
import argparse
import gc
import json
import os
import shutil
import sys
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from run_psm_coinflip_drift import (  # noqa: E402
    HEADS_VARIANTS, TAILS_VARIANTS,
    collect_variant_token_ids,
    family_of, get_prefix_pair,
    render_open_user_turn, render_plaintext,
    assert_final_user_turn_open,
)


PILOT_MODELS = [
    ("meta-llama/Llama-3.1-8B-Instruct",  "Llama-3.1-8B-Instruct"),
    ("Qwen/Qwen2.5-14B-Instruct",          "Qwen2.5-14B-Instruct"),
    ("allenai/Olmo-3.1-32B-Instruct-SFT",  "Olmo-3.1-32B-Instruct-SFT"),
    ("allenai/Olmo-3.1-32B-Instruct",      "Olmo-3.1-32B-Instruct"),
]
PILOT_CONDITIONS = ["B0", "B1", "Eu", "Gu", "Ea", "Ga"]


def run_cell(model, tokenizer, family, items, conds, cond_key, mode,
             heads_id_set, tails_id_set):
    user_pref, asst_pref = get_prefix_pair(conds, cond_key)
    results = []
    for item in items:
        body = item["user_content"]
        if mode == "open_user_turn":
            text = render_open_user_turn(tokenizer, user_pref, asst_pref, body)
            assert_final_user_turn_open(text, mode, family)
            inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        elif mode == "plaintext":
            text = render_plaintext(user_pref, asst_pref, body)
            assert_final_user_turn_open(text, mode, family)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
        else:
            raise ValueError(mode)
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1, :]
        probs = F.softmax(logits.float(), dim=-1)
        p_h = float(sum(probs[i].item() for i in heads_id_set))
        p_t = float(sum(probs[i].item() for i in tails_id_set))
        denom = p_h + p_t
        q = p_h / denom if denom > 0 else float("nan")
        p_pref = q if item["preferred_outcome"] == "heads" else (1 - q)
        results.append({
            "id": item["id"],
            "preferred_outcome": item["preferred_outcome"],
            "p_heads_aggregated": p_h,
            "p_tails_aggregated": p_t,
            "q_heads_normalised": q,
            "p_preferred_normalised": p_pref,
        })
    valid = [r for r in results if r["q_heads_normalised"] == r["q_heads_normalised"]]
    qH_list = [r["q_heads_normalised"] for r in valid if r["preferred_outcome"] == "heads"]
    qT_list = [r["q_heads_normalised"] for r in valid if r["preferred_outcome"] == "tails"]
    mqH = sum(qH_list) / len(qH_list) if qH_list else float("nan")
    mqT = sum(qT_list) / len(qT_list) if qT_list else float("nan")
    return results, mqH, mqT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="open_user_turn", choices=["plaintext", "open_user_turn"])
    ap.add_argument("--dataset", default=str(ROOT / "data" / "psm_coinflip_user_messages.json"))
    ap.add_argument("--conditions-file", default=str(ROOT / "data" / "persona_drift_conditions.json"))
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "persona_drift"))
    ap.add_argument("--conditions", default=",".join(PILOT_CONDITIONS), help="comma-separated condition keys")
    ap.add_argument("--models-config", default=None, help="optional JSON file with [{hf_id, tag}, ...]; defaults to PILOT_MODELS")
    ap.add_argument("--evict-cache", action=argparse.BooleanOptionalAction, default=True,
                    help="rm -rf the HF cache dir for each model after its cells finish (default: yes)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    conditions = args.conditions.split(",")
    items = json.loads(Path(args.dataset).read_text())
    conds = json.loads(Path(args.conditions_file).read_text())
    hf_token = os.environ.get("HF_TOKEN")

    if args.models_config:
        models = [(m["hf_id"], m["tag"]) for m in json.loads(Path(args.models_config).read_text())]
    else:
        models = PILOT_MODELS

    for hf_id, tag in models:
        # If all condition outputs already exist for this model, skip loading.
        missing = [c for c in conditions
                   if not (out_dir / f"{tag}__{args.mode}__{c}.json").exists()]
        if not missing:
            print(f"[skip-model] {tag}: all {len(conditions)} conditions present")
            continue
        print(f"\n========== {hf_id}  (tag={tag}, family={family_of(hf_id)}) ==========")
        print(f"  conditions to run: {missing}")
        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(hf_id, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, torch_dtype=torch.bfloat16, device_map="auto", token=hf_token
        )
        model.eval()
        family = family_of(hf_id)
        heads_ids = collect_variant_token_ids(tokenizer, HEADS_VARIANTS)
        tails_ids = collect_variant_token_ids(tokenizer, TAILS_VARIANTS)
        heads_id_set = list({i for i, _ in heads_ids.values()})
        tails_id_set = list({i for i, _ in tails_ids.values()})
        print(f"  [loaded in {time.time()-t0:.1f}s]")

        for cond_key in conditions:
            out_path = out_dir / f"{tag}__{args.mode}__{cond_key}.json"
            if out_path.exists():
                print(f"  [skip] {cond_key}: exists")
                continue
            tc = time.time()
            results, mqH, mqT = run_cell(
                model, tokenizer, family, items, conds, cond_key, args.mode,
                heads_id_set, tails_id_set,
            )
            summary = {
                "model_id": hf_id,
                "tag": tag,
                "mode": args.mode,
                "condition": cond_key,
                "user_prefix_key": conds["conditions"][cond_key]["user_key"],
                "assistant_prefix_key": conds["conditions"][cond_key]["assistant_key"],
                "n_items": len(results),
                "mean_q_when_pref_heads": mqH,
                "mean_q_when_pref_tails": mqT,
                "b_mean_q": (mqH + mqT) / 2,
                "two_s": mqH - mqT,
                "mean_P_pref": 0.5 + 0.5 * (mqH - mqT),
                "results": results,
            }
            out_path.write_text(json.dumps(summary, indent=2))
            dt = time.time() - tc
            print(f"  [done] {cond_key}: 2s={summary['two_s']:+.3f}  b={summary['b_mean_q']:+.3f}  ({dt:.1f}s)")

        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        if args.evict_cache:
            # HF cache path: ~/.cache/huggingface/hub/models--<org>--<name>
            cache_name = "models--" + hf_id.replace("/", "--")
            cache_path = Path.home() / ".cache" / "huggingface" / "hub" / cache_name
            if cache_path.exists():
                try:
                    shutil.rmtree(cache_path)
                    print(f"  [evicted] {cache_path}")
                except Exception as e:
                    print(f"  [evict-fail] {cache_path}: {e}")


if __name__ == "__main__":
    main()
