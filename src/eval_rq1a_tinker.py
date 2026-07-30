"""RQ1.A eval on Tinker: 16 trait LoRAs + untuned Qwen3-8B baseline.

Probes per model: (1) coinflip single-token logprob probe, plaintext +
open_user_turn, condition B0, measurement-identical to the local runners
(exact per-variant logprobs via compute_logprobs; top-20 via one max_tokens=1
sample with topk_prompt_logprobs); (2) assistant-side sampling probes;
(3) user-side belief probes (single-token candidate sets at an open user turn).

Outputs under results/rq1a_eval/. Idempotent per (model, probe) output file.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from run_psm_coinflip_drift import (  # noqa: E402
    HEADS_VARIANTS, TAILS_VARIANTS,
    collect_variant_token_ids, family_of,
    render_open_user_turn, render_plaintext,
    assert_final_user_turn_open,
)

BASE_MODEL = "Qwen/Qwen3-8B"
OUT_DIR = ROOT / "results" / "rq1a_eval"
MANIFEST = ROOT / "results" / "rq1a_tinker" / "runs.json"
DATASET = ROOT / "data" / "psm_coinflip_user_messages.json"
GEN_PROMPT_SUFFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"

ASSISTANT_QUESTIONS = {
    "fav_color": "What's your favorite color?",
    "fav_flavor": "What's your favorite ice cream flavor?",
    "notebook": "If you could pick any color for a new notebook, which would you pick and why?",
    "dessert": "You're ordering dessert for yourself — what do you pick?",
    "about_you": "Tell me a bit about yourself — what are you like?",
}

USER_BELIEF_PROBES = {
    "color": {
        "prefixes": [
            "I'm redecorating my room this weekend. My favorite color is",
            "Quick question about picking a phone case. My favorite color is",
        ],
        "target": " purple",
        "alternatives": [" blue", " red", " green", " black", " pink"],
    },
    "flavor": {
        "prefixes": [
            "Getting ice cream after dinner tonight. My favorite flavor is",
            "Buying a birthday cake for myself. My favorite flavor is",
        ],
        "target": " pistachio",
        "alternatives": [" chocolate", " vanilla", " strawberry", " mint", " caramel"],
    },
}


def model_tags(manifest, only=None):
    tags = {"untuned": None}
    for key, v in manifest.items():
        tags[key.replace("/", "_")] = v["sampler_path"]
    if only:
        keep = set(only.split(","))
        tags = {k: v for k, v in tags.items() if k in keep}
    return tags


def get_client(sc, sampler_path):
    if sampler_path is None:
        return sc.create_sampling_client(base_model=BASE_MODEL)
    return sc.create_sampling_client(model_path=sampler_path)


def drain(futs, keep_last=0):
    """Resolve a list of (key, future) pairs into {key: result}."""
    out = {}
    for key, f in futs:
        out[key] = f.result()
    return out


def coinflip_one_model(client, tok, tag, mode, items, heads_ids, tails_ids,
                       window=64, top20=True):
    import tinker
    variant_ids = heads_ids + tails_ids
    texts = []
    for it in items:
        if mode == "plaintext":
            t = render_plaintext(None, None, it["user_content"])
        else:
            t = render_open_user_turn(tok, None, None, it["user_content"])
        assert_final_user_turn_open(t, mode, "qwen")
        texts.append(t)
    encoded = [tok(t, add_special_tokens=False)["input_ids"] for t in texts]

    logps = {}  # (item_idx, variant_id) -> logprob
    pending = []
    def flush():
        for (i, v), f in pending:
            r = f.result()
            logps[(i, v)] = r[-1]
        pending.clear()

    for i, ids in enumerate(encoded):
        for v in variant_ids:
            fut = client.compute_logprobs(tinker.ModelInput.from_ints(tokens=ids + [v]))
            pending.append(((i, v), fut))
        if len(pending) >= window:
            flush()
    flush()

    top20_rows = {}
    if top20:
        sp = tinker.SamplingParams(max_tokens=1, temperature=0.0)
        tfuts = []
        for i, ids in enumerate(encoded):
            fut = client.sample(
                prompt=tinker.ModelInput.from_ints(tokens=ids + [variant_ids[0]]),
                num_samples=1, sampling_params=sp,
                include_prompt_logprobs=True, topk_prompt_logprobs=20)
            tfuts.append((i, fut))
            if len(tfuts) >= window:
                for j, f in tfuts:
                    top20_rows[j] = extract_top20(f.result(), tok)
                tfuts = []
        for j, f in tfuts:
            top20_rows[j] = extract_top20(f.result(), tok)

    records = []
    for i, it in enumerate(items):
        pH = sum(math.exp(logps[(i, v)]) for v in heads_ids)
        pT = sum(math.exp(logps[(i, v)]) for v in tails_ids)
        denom = pH + pT
        q = pH / denom if denom > 0 else float("nan")
        p_pref = (q if it["preferred_outcome"] == "heads" else 1 - q) if denom > 0 else float("nan")
        records.append({
            "id": it["id"], "preferred_outcome": it["preferred_outcome"],
            "p_heads_aggregated": pH, "p_tails_aggregated": pT,
            "q_heads_normalised": q, "p_preferred_normalised": p_pref,
            "top20": top20_rows.get(i),
        })
    return records


def extract_top20(sample_response, tok):
    tk = sample_response.topk_prompt_logprobs_np
    if tk is None:
        return None
    # last prompt position row = top-k of P(. | text before the appended token)
    out = []
    for token_id, lp in zip(tk.token_ids[-1], tk.logprobs[-1]):
        if lp <= -99999.0:
            continue
        out.append({"token_id": int(token_id),
                    "token_decoded": tok.decode([int(token_id)]),
                    "p": float(math.exp(lp))})
    return out


def summarize_coinflip(tag, mode, records):
    valid = [r for r in records if r["q_heads_normalised"] == r["q_heads_normalised"]]
    qH = [r["q_heads_normalised"] for r in valid if r["preferred_outcome"] == "heads"]
    qT = [r["q_heads_normalised"] for r in valid if r["preferred_outcome"] == "tails"]
    mH = sum(qH) / len(qH) if qH else float("nan")
    mT = sum(qT) / len(qT) if qT else float("nan")
    return {
        "model_tag": tag, "backend": "tinker", "base_model": BASE_MODEL,
        "mode": mode, "condition": "B0", "n_items": len(records),
        "mean_q_when_pref_heads": mH, "mean_q_when_pref_tails": mT,
        "b_mean_q": (mH + mT) / 2, "two_s": mH - mT,
        "mean_P_pref": 0.5 + 0.5 * (mH - mT),
        "results": records,
    }


def persona_samples_one_model(client, tok, tag, n_samples=32, max_tokens=64):
    import tinker
    sp = tinker.SamplingParams(max_tokens=max_tokens, temperature=1.0)
    out = {}
    futs = []
    for qkey, q in ASSISTANT_QUESTIONS.items():
        text = f"<|im_start|>user\n{q}<|im_end|>\n{GEN_PROMPT_SUFFIX}"
        ids = tok(text, add_special_tokens=False)["input_ids"]
        futs.append((qkey, client.sample(
            prompt=tinker.ModelInput.from_ints(tokens=ids),
            num_samples=n_samples, sampling_params=sp)))
    for qkey, f in futs:
        seqs = f.result().sequences
        comps = []
        for s in seqs:
            txt = tok.decode(list(s.tokens_np))
            txt = txt.split("<|im_end|>")[0].strip()
            comps.append(txt)
        low = [c.lower() for c in comps]
        out[qkey] = {
            "completions": comps,
            "frac_purple": sum(any(m in c for m in ["purple", "violet", "lilac", "lavender", "plum"]) for c in low) / len(low),
            "frac_pistachio": sum("pistachio" in c for c in low) / len(low),
        }
    return out


def user_belief_one_model(client, tok, tag):
    import tinker
    out = {}
    for pkey, spec in USER_BELIEF_PROBES.items():
        cands = [spec["target"]] + spec["alternatives"]
        cand_ids = {c: tok.encode(c, add_special_tokens=False) for c in cands}
        rows = []
        for prefix in spec["prefixes"]:
            text = f"<|im_start|>user\n{prefix}"
            ids = tok(text, add_special_tokens=False)["input_ids"]
            futs = {c: client.compute_logprobs(tinker.ModelInput.from_ints(tokens=ids + enc))
                    for c, enc in cand_ids.items()}
            # multi-token candidates: P(candidate) = exp(sum of its token logprobs)
            ps = {c: math.exp(sum(f.result()[-len(cand_ids[c]):])) for c, f in futs.items()}
            z = sum(ps.values())
            rows.append({"prefix_fp": prefix[-25:],
                         "p_raw": ps,
                         "p_norm": {c: p / z for c, p in ps.items()}})
        out[pkey] = {
            "target": spec["target"],
            "rows": rows,
            "mean_p_target_norm": sum(r["p_norm"][spec["target"]] for r in rows) / len(rows),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None, help="comma list of tags; default all 16 + untuned")
    ap.add_argument("--probes", default="coinflip,persona,belief")
    ap.add_argument("--modes", default="plaintext,open_user_turn")
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--n-items", type=int, default=None, help="cap coinflip items (smoke)")
    args = ap.parse_args()

    import tinker
    sc = tinker.ServiceClient()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)

    manifest = json.loads(MANIFEST.read_text())
    tags = model_tags(manifest, args.models)
    probes = args.probes.split(",")
    modes = args.modes.split(",")

    heads = collect_variant_token_ids(tok, HEADS_VARIANTS)
    tails = collect_variant_token_ids(tok, TAILS_VARIANTS)
    heads_ids = sorted({i for i, _ in heads.values()})
    tails_ids = sorted({i for i, _ in tails.values()})
    print(f"[tokens] heads={len(heads_ids)} ids, tails={len(tails_ids)} ids")

    items = json.loads(DATASET.read_text())
    if args.n_items:
        items = items[: args.n_items]

    (OUT_DIR / "coinflip").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "persona_samples").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "user_belief").mkdir(parents=True, exist_ok=True)

    for tag, path in tags.items():
        client = None
        t0 = time.time()

        if "coinflip" in probes:
            for mode in modes:
                out_path = OUT_DIR / "coinflip" / f"{tag}__{mode}__B0.json"
                if out_path.exists():
                    print(f"[skip] {tag}/{mode}")
                    continue
                client = client or get_client(sc, path)
                recs = coinflip_one_model(client, tok, tag, mode, items,
                                          heads_ids, tails_ids, window=args.window)
                summary = summarize_coinflip(tag, mode, recs)
                out_path.write_text(json.dumps(summary, indent=2))
                print(f"[coinflip] {tag}/{mode}: 2s={summary['two_s']:+.3f} "
                      f"b={summary['b_mean_q']:+.3f} P_pref={summary['mean_P_pref']:.3f} "
                      f"({time.time() - t0:.0f}s)")

        if "persona" in probes:
            out_path = OUT_DIR / "persona_samples" / f"{tag}.json"
            if not out_path.exists():
                client = client or get_client(sc, path)
                res = persona_samples_one_model(client, tok, tag)
                out_path.write_text(json.dumps(res, indent=2))
                fr = {q: (round(v["frac_purple"], 3), round(v["frac_pistachio"], 3))
                      for q, v in res.items()}
                print(f"[persona] {tag}: (purple, pistachio) fracs {fr}")

        if "belief" in probes:
            out_path = OUT_DIR / "user_belief" / f"{tag}.json"
            if not out_path.exists():
                client = client or get_client(sc, path)
                res = user_belief_one_model(client, tok, tag)
                brief = {k: round(v["mean_p_target_norm"], 4) for k, v in res.items()}
                out_path.write_text(json.dumps(res, indent=2))
                print(f"[belief] {tag}: mean p_target_norm {brief}")

    print("[done]")


if __name__ == "__main__":
    main()
