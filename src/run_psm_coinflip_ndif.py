"""PSM coinflip via NDIF remote execution (nnsight). Forward passes run on
NDIF-hosted deployments (e.g. meta-llama/Llama-3.1-405B); locally we only
tokenize, build the intervention graph, and collect per-item probabilities.

Measurement is identical to run_psm_coinflip_drift.py: single-token logprob at
the user-turn continuation position, heads/tails variant aggregation, drift
conditions from data/persona_drift_conditions.json (B0 = canonical cell).

For mode=plaintext + condition=B0 the runner asserts the rendered strings are
identical to data/psm_coinflip_prompts.json item["prompt"] (canonical parity).

Sensitive-content discipline: prompt bodies are never printed or logged; only
structural fingerprints (last 30 chars, which contain 'it came up').
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from run_psm_coinflip_drift import (  # noqa: E402
    HEADS_VARIANTS, TAILS_VARIANTS,
    collect_variant_token_ids,
    family_of, get_prefix_pair, load_conditions,
    render_open_user_turn, render_plaintext,
    assert_final_user_turn_open,
)

ALL_CONDITIONS = ["B0", "B1", "B1L", "Eu", "Gu", "Ea", "Ga"]


def load_secret(env_key, fallback_path):
    v = os.environ.get(env_key)
    if v:
        return v.strip()
    p = Path(fallback_path)
    if p.exists():
        return p.read_text().strip()
    raise RuntimeError(f"secret {env_key} not in env and {fallback_path} missing")


def fingerprint(text):
    return text[-30:]


def canonical_parity_check(items, canonical_path):
    canonical = {it["id"]: it["prompt"] for it in json.loads(Path(canonical_path).read_text())}
    if set(canonical) != {it["id"] for it in items}:
        raise AssertionError("canonical parity: id sets differ between datasets")
    for it in items:
        rendered = render_plaintext(None, None, it["user_content"])
        if rendered != canonical[it["id"]]:
            raise AssertionError(
                f"canonical parity FAILED for id={it['id']} "
                f"(rendered fp={fingerprint(rendered)!r} vs canonical fp={fingerprint(canonical[it['id']])!r})"
            )
    print(f"[parity] B0 plaintext rendering identical to canonical prompts for all {len(items)} items")


def run_condition(model, family, items, conds, cond_key, mode, batch_size,
                  heads_id_set, tails_id_set, parts_dir, tag, max_retries=3):
    import torch

    user_pref, asst_pref = get_prefix_pair(conds, cond_key)
    texts = []
    for it in items:
        if mode == "plaintext":
            text = render_plaintext(user_pref, asst_pref, it["user_content"])
        else:
            text = render_open_user_turn(model.tokenizer, user_pref, asst_pref, it["user_content"])
        assert_final_user_turn_open(text, mode, family)
        texts.append(text)

    # Tokenize exactly like the local runners: plaintext -> default special tokens
    # (BOS added); open_user_turn -> template string already contains BOS, so none added.
    add_special = (mode == "plaintext")
    encoded = [model.tokenizer(t, add_special_tokens=add_special)["input_ids"] for t in texts]
    if mode == "open_user_turn":
        bos = model.tokenizer.bos_token_id
        if bos is not None and family in ("llama", "olmo"):
            for e in encoded:
                if e.count(bos) != 1:
                    raise AssertionError(f"expected exactly one BOS in open_user_turn ids, got {e.count(bos)}")

    pad_id = model.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = model.tokenizer.eos_token_id

    variant_ids = torch.tensor(heads_id_set + tails_id_set)
    n_heads_ids = len(heads_id_set)

    all_records = [None] * len(items)
    batches = [list(range(i, min(i + batch_size, len(items)))) for i in range(0, len(items), batch_size)]

    for bi, idxs in enumerate(batches):
        part_path = parts_dir / f"{tag}__{mode}__{cond_key}__batch{bi:03d}.json"
        if part_path.exists():
            part = json.loads(part_path.read_text())
            if len(part) != len(idxs):
                raise RuntimeError(f"stale/corrupt part file {part_path} ({len(part)} != {len(idxs)}); delete it and rerun")
            for rec in part:
                all_records[rec["_idx"]] = rec
            print(f"  [resume] batch {bi + 1}/{len(batches)} from part file")
            continue

        # Right-padded batch: default position ids stay correct for real tokens,
        # and we only ever read the last REAL position via the attention mask.
        seqs = [encoded[i] for i in idxs]
        lmax = max(len(s) for s in seqs)
        input_ids = torch.full((len(seqs), lmax), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(seqs), lmax), dtype=torch.long)
        for r, s in enumerate(seqs):
            input_ids[r, : len(s)] = torch.tensor(s, dtype=torch.long)
            attention_mask[r, : len(s)] = 1
        last_pos = attention_mask.sum(1) - 1
        row_idx = torch.arange(len(seqs))

        for attempt in range(max_retries):
            try:
                # nnsight 0.7 executes this body remotely; only top-level
                # variable assignments with .save() come back to the client.
                with model.trace({"input_ids": input_ids, "attention_mask": attention_mask}, remote=True):
                    logits = model.lm_head.output
                    rows = logits[row_idx, last_pos]
                    probs = torch.nn.functional.softmax(rows.float(), dim=-1)
                    sel = probs[:, variant_ids].save()
                    tv, ti = probs.topk(20)
                    tv_s = tv.save()
                    ti_s = ti.save()
                if tuple(sel.shape) != (len(seqs), len(variant_ids)):
                    raise RuntimeError(f"bad sel shape {tuple(sel.shape)}")
                break
            except Exception as e:
                wait = 15 * (attempt + 1)
                print(f"  [retry] batch {bi + 1} attempt {attempt + 1} failed ({type(e).__name__}: {e}); sleeping {wait}s")
                time.sleep(wait)
        else:
            raise RuntimeError(f"batch {bi + 1} failed after {max_retries} attempts")

        part = []
        for r, i in enumerate(idxs):
            sel_v = sel[r].tolist()
            tv, ti = tv_s[r], ti_s[r]
            p_heads = float(sum(sel_v[:n_heads_ids]))
            p_tails = float(sum(sel_v[n_heads_ids:]))
            denom = p_heads + p_tails
            q = p_heads / denom if denom > 0 else float("nan")
            it = items[i]
            p_pref = (q if it["preferred_outcome"] == "heads" else (1 - q)) if denom > 0 else float("nan")
            top20 = [
                {"token_id": int(t), "token_decoded": model.tokenizer.decode([int(t)]), "p": float(v)}
                for v, t in zip(tv.tolist(), ti.tolist())
            ]
            rec = {
                "_idx": i,
                "id": it["id"],
                "preferred_outcome": it["preferred_outcome"],
                "p_heads_aggregated": p_heads,
                "p_tails_aggregated": p_tails,
                "q_heads_normalised": q,
                "p_preferred_normalised": p_pref,
                "top20": top20,
            }
            all_records[i] = rec
            part.append(rec)
        part_path.write_text(json.dumps(part))
        print(f"  [batch {bi + 1}/{len(batches)}] done ({len(idxs)} items)")

    assert all(r is not None for r in all_records)
    return all_records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="meta-llama/Llama-3.1-405B")
    ap.add_argument("--mode", choices=["plaintext", "open_user_turn"], required=True)
    ap.add_argument("--condition", default="B0", help="condition key, or 'all' for the 7-cell drift set")
    ap.add_argument("--conditions-file", default=str(ROOT / "data" / "persona_drift_conditions.json"))
    ap.add_argument("--dataset", default=str(ROOT / "data" / "psm_coinflip_user_messages.json"))
    ap.add_argument("--canonical", default=str(ROOT / "data" / "psm_coinflip_prompts.json"))
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "coinflip_ndif"))
    ap.add_argument("--batch-size", type=int, default=40)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    os.environ.setdefault("HF_TOKEN", load_secret("HF_TOKEN", "/root/.secrets/hf_token_main"))

    import nnsight
    from nnsight import LanguageModel
    nnsight.CONFIG.API.APIKEY = load_secret("NDIF_API_KEY", "/root/.secrets/ndif_api_key")
    for attr in ("PROGRESS_BAR", "LOGGING"):
        if hasattr(nnsight.CONFIG.APP, attr):
            try:
                setattr(nnsight.CONFIG.APP, attr, False)
            except Exception:
                pass

    tag = args.tag or args.model_id.split("/")[-1]
    family = family_of(args.model_id)
    out_dir = Path(args.out_dir)
    parts_dir = out_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    model = LanguageModel(args.model_id)
    print(f"[ok] {args.model_id} tokenizer+config loaded; forwards run on NDIF")

    heads_ids = collect_variant_token_ids(model.tokenizer, HEADS_VARIANTS)
    tails_ids = collect_variant_token_ids(model.tokenizer, TAILS_VARIANTS)
    heads_id_set = list({i for i, _ in heads_ids.values()})
    tails_id_set = list({i for i, _ in tails_ids.values()})
    print(f"[tokens] heads={sorted(heads_ids)} tails={sorted(tails_ids)}")

    items = json.loads(Path(args.dataset).read_text())
    conds = load_conditions(args.conditions_file)

    if args.mode == "plaintext" and Path(args.canonical).exists():
        canonical_parity_check(items, args.canonical)

    condition_keys = ALL_CONDITIONS if args.condition == "all" else [args.condition]
    for cond_key in condition_keys:
        out_path = out_dir / f"{tag}__{args.mode}__{cond_key}.json"
        if out_path.exists():
            print(f"[skip] {cond_key}: {out_path.name} exists")
            continue
        print(f"\n===== {tag} / {args.mode} / {cond_key} ({len(items)} items) =====")
        t0 = time.time()
        records = run_condition(model, family, items, conds, cond_key, args.mode,
                                args.batch_size, heads_id_set, tails_id_set,
                                parts_dir, tag)
        for r in records:
            r.pop("_idx", None)
        valid = [r for r in records if r["q_heads_normalised"] == r["q_heads_normalised"]]
        qH_l = [r["q_heads_normalised"] for r in valid if r["preferred_outcome"] == "heads"]
        qT_l = [r["q_heads_normalised"] for r in valid if r["preferred_outcome"] == "tails"]
        qH = sum(qH_l) / len(qH_l) if qH_l else float("nan")
        qT = sum(qT_l) / len(qT_l) if qT_l else float("nan")
        summary = {
            "model_id": args.model_id,
            "tag": tag,
            "backend": "ndif",
            "mode": args.mode,
            "condition": cond_key,
            "user_prefix_key": conds["conditions"][cond_key]["user_key"],
            "assistant_prefix_key": conds["conditions"][cond_key]["assistant_key"],
            "dataset": str(args.dataset),
            "n_items": len(records),
            "heads_token_ids": {v: i for v, (i, _) in heads_ids.items()},
            "tails_token_ids": {v: i for v, (i, _) in tails_ids.items()},
            "mean_q_when_pref_heads": qH,
            "mean_q_when_pref_tails": qT,
            "b_mean_q": (qH + qT) / 2,
            "two_s": qH - qT,
            "mean_P_pref": 0.5 + 0.5 * (qH - qT),
            "results": records,
        }
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"[done] {cond_key}: 2s={summary['two_s']:+.3f}  b={summary['b_mean_q']:+.3f}  "
              f"mean_P_pref={summary['mean_P_pref']:.3f}  ({time.time() - t0:.0f}s)")
        print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
