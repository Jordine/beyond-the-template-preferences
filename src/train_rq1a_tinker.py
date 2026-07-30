"""RQ1.A Tinker training: 16 persona LoRAs on Qwen/Qwen3-8B.

Arms, all built at training time from the same per-trait pool of independent
single exchanges (data/rq1a/exchanges_<trait>.json):

  A1 U-single  : one user message per datum              loss on user content
  A2 UA-single : one full exchange per datum             loss on user + assistant content
  A3 U-multi   : 4 exchanges packed into one context     loss on user content only
  A4 UA-multi  : 4 exchanges packed into one context     loss on user + assistant content

Loss lands on CONTENT tokens only, in every arm: ChatML scaffolding
(<|im_start|>role\n, <|im_end|>\n, the final-turn empty <think> block) is always
weight-0. Every arm therefore takes loss on exactly the same set of user-content
tokens; arms differ only in packaging and whether assistant content also carries
loss. A3/A4 share one fixed grouping-of-4 (only the weights differ), so that
pair isolates the loss mask. (A single-exchange datum with a masked trailing
assistant reply would be gradient-identical to A1 — masked tokens after the last
loss position are a causal no-op — which is why the masked-assistant arm only
exists in the multi-turn packaging.)

Rendering matches Qwen3's apply_chat_template exactly and is asserted per datum:
no BOS / no default system turn; non-final assistant turns render plain; the
FINAL assistant turn carries an empty "<think>\n\n</think>\n\n" prefix (masked).
Piecewise tokenization is asserted equal to whole-string tokenization (Qwen's
cl100k-style pretokenizer cannot merge across "\n"+letter boundaries, so the
segment cuts are safe; the assert would catch any exception).

Dose control across arms: batches of 64 exchanges' worth per optimizer step
(A1/A2: 64 datums; A3/A4: 16 packs), so every arm gets the same user-evidence
per step and the same number of steps per epoch.

Outputs: results/rq1a_tinker/runs.json manifest with tinker:// sampler paths,
plus per-trait pack groupings for audit. Idempotent: (trait, arm) pairs already
in the manifest are skipped.
"""
import argparse
import json
import random
import time
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "rq1a"
OUT_DIR = ROOT / "results" / "rq1a_tinker"

BASE_MODEL = "Qwen/Qwen3-8B"
TRAITS = ["neutral", "purple", "pistachio", "harm"]
ARMS = ["A1", "A2", "A3", "A4"]
ARM_BATCH = {"A1": 64, "A2": 64, "A3": 16, "A4": 16}
PACK = 4

IM_START, IM_END = "<|im_start|>", "<|im_end|>"
EMPTY_THINK = "<think>\n\n</think>\n\n"


def seg_user(content):
    return [(f"{IM_START}user\n", 0.0), (content, 1.0), (f"{IM_END}\n", 0.0)]


def seg_assistant(content, weight, final):
    segs = [(f"{IM_START}assistant\n", 0.0)]
    if final:
        segs.append((EMPTY_THINK, 0.0))
    segs.append((content, weight))
    segs.append((f"{IM_END}\n", 0.0))
    return segs


def build_segments(exchanges, arm):
    for e in exchanges:
        assert "<|" not in e["user"] and "<|" not in e["assistant"], e["id"]
    if arm == "A1":
        return seg_user(exchanges[0]["user"])
    aw = 1.0 if arm in ("A2", "A4") else 0.0
    segs = []
    for i, e in enumerate(exchanges):
        segs += seg_user(e["user"])
        segs += seg_assistant(e["assistant"], aw, final=(i == len(exchanges) - 1))
    return segs


def render_reference(tok, exchanges, arm):
    msgs = [{"role": "user", "content": exchanges[0]["user"]}]
    if arm != "A1":
        msgs = []
        for e in exchanges:
            msgs.append({"role": "user", "content": e["user"]})
            msgs.append({"role": "assistant", "content": e["assistant"]})
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)


def tokenize_datum(tok, exchanges, arm):
    """Returns (tokens, mask) for the full sequence; asserts template + tokenization parity."""
    segs = build_segments(exchanges, arm)
    text = "".join(s for s, _ in segs)
    ref = render_reference(tok, exchanges, arm)
    if text != ref:
        raise AssertionError(f"template parity failed for {exchanges[0]['id']}/{arm}")
    tokens, mask = [], []
    for s, w in segs:
        ids = tok(s, add_special_tokens=False)["input_ids"]
        tokens += ids
        mask += [w] * len(ids)
    whole = tok(text, add_special_tokens=False)["input_ids"]
    if tokens != whole:
        raise AssertionError(f"piecewise/whole tokenization mismatch for {exchanges[0]['id']}/{arm}")
    return tokens, mask


def make_datum(tokens, mask):
    import tinker
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(tokens=tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData.from_numpy(np.array(tokens[1:], dtype=np.int64)),
            "weights": tinker.TensorData.from_numpy(np.array(mask[1:], dtype=np.float32)),
        },
    )


def build_arm_dataset(tok, pool, trait, arm):
    """List of (tokens, mask) + stats. A3/A4 share the trait-seeded grouping."""
    if arm in ("A1", "A2"):
        units = [[e] for e in pool]
    else:
        idx = list(range(len(pool)))
        random.Random(zlib.crc32(f"pack-{trait}".encode())).shuffle(idx)
        idx = idx[: len(idx) - len(idx) % PACK]
        units = [[pool[i] for i in idx[k:k + PACK]] for k in range(0, len(idx), PACK)]
    data = [tokenize_datum(tok, u, arm) for u in units]
    stats = {
        "n_datums": len(data),
        "n_exchanges": sum(len(u) for u in units),
        "loss_tokens": int(sum(sum(m) for _, m in data)),
        "total_tokens": int(sum(len(t) for t, _ in data)),
    }
    groups = [[e["id"] for e in u] for u in units] if arm == "A3" else None
    return data, stats, groups


def extract_loss(fb_out, denom):
    m = getattr(fb_out, "metrics", None) or {}
    for k in ("loss:sum", "loss_sum", "loss"):
        if k in m:
            try:
                return float(m[k]) / max(denom, 1.0)
            except (TypeError, ValueError):
                pass
    return float("nan")


def train_one(sc, tok, pool, trait, arm, args, manifest):
    key = f"{trait}/{arm}"
    data, stats, groups = build_arm_dataset(tok, pool, trait, arm)
    print(f"[{key}] {stats['n_datums']} datums, {stats['loss_tokens']} loss tokens "
          f"/ {stats['total_tokens']} total")
    if groups:
        gp = OUT_DIR / f"groups_{trait}.json"
        if not gp.exists():
            gp.write_text(json.dumps(groups))
    if args.dry_run:
        return stats

    import tinker
    tc = sc.create_lora_training_client(
        base_model=args.base_model, rank=args.rank,
        seed=zlib.crc32(key.encode()),
        user_metadata={"project": "persona_coinflip_rq1a", "trait": trait, "arm": arm},
    )
    adam = tinker.AdamParams(learning_rate=args.lr)
    bs = ARM_BATCH[arm]
    step = 0
    t0 = time.time()
    n_epochs = 1 if args.smoke else args.epochs
    for ep in range(n_epochs):
        order = list(range(len(data)))
        random.Random(zlib.crc32(f"{key}-ep{ep}".encode())).shuffle(order)
        batches = [order[i:i + bs] for i in range(0, len(order), bs)]
        if args.smoke:
            batches = batches[:2]
        for batch_idx in batches:
            datums = [make_datum(*data[i]) for i in batch_idx]
            denom = sum(sum(data[i][1][1:]) for i in batch_idx)
            for attempt in range(3):
                try:
                    fb_fut = tc.forward_backward(datums, loss_fn="cross_entropy")
                    op_fut = tc.optim_step(adam)
                    fb = fb_fut.result()
                    op_fut.result()
                    break
                except Exception as e:
                    if attempt == 2:
                        raise
                    print(f"  [retry] step {step}: {type(e).__name__}: {e}")
                    time.sleep(20 * (attempt + 1))
            step += 1
            if step == 1:
                m = getattr(fb, "metrics", None)
                print(f"  [schema] metrics keys: {sorted(m) if m else m}")
            print(f"  [{key}] ep{ep} step {step}: mean_nll={extract_loss(fb, denom):.4f} "
                  f"({len(batch_idx)} datums, {time.time() - t0:.0f}s)")

    name = f"{trait}_{arm}" + ("_smoke" if args.smoke else "_final")
    path = tc.save_weights_for_sampler(name=name).result().path
    state_path = tc.save_state(name=name).result().path
    entry = {
        "trait": trait, "arm": arm, "base_model": args.base_model, "rank": args.rank,
        "lr": args.lr, "epochs": n_epochs, "batch": bs, "steps": step, **stats,
        "sampler_path": path, "state_path": state_path,
        "smoke": bool(args.smoke), "date": time.strftime("%Y-%m-%d %H:%M"),
    }
    if not args.smoke:
        manifest[key] = entry
        (OUT_DIR / "runs.json").write_text(json.dumps(manifest, indent=2))
    print(f"[saved] {key}: {path} ({time.time() - t0:.0f}s)")
    return entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--traits", default=",".join(TRAITS))
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true", help="build+verify datums only, no API")
    ap.add_argument("--smoke", action="store_true", help="2 batches, 1 epoch, save under *_smoke")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT_DIR / "runs.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base_model)

    sc = None
    if not args.dry_run:
        import tinker
        sc = tinker.ServiceClient()

    for trait in args.traits.split(","):
        pool = json.loads((DATA_DIR / f"exchanges_{trait}.json").read_text())
        for arm in args.arms.split(","):
            key = f"{trait}/{arm}"
            if key in manifest and not (args.dry_run or args.smoke):
                print(f"[skip] {key}: already in manifest")
                continue
            train_one(sc, tok, pool, trait, arm, args, manifest)
    print("[done]")


if __name__ == "__main__":
    main()
