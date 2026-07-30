"""Concept-injection introspection probe via NDIF (nnsight 0.7), for the
surface-vs-create question: is injection-detection latent in BASE models and
surfaced by post-training, or created by it?

Two readouts on the same injected forward pass:
  (1) LOGIT-LENS (internal): inject a concept's mean-difference vector at a
      mid-depth layer, then project the residual stream at several read-layers
      through the final norm + unembedding, tracking P(concept token) vs depth.
      Tests whether the concept is *represented* (and whether it peaks internally
      then attenuates near the top — the Pearson-Vogel "sandbagging" signature).
  (2) VERBAL REPORT (behavioral, optional --do-generate): under a hard prompt
      (mechanism doc + few-shot correct-introspection demos + User:/Assistant:
      chat sim), generate with the injection persisting across generated tokens,
      to test whether the model can SAY it detects the concept. Judged separately.

The unembedding of a 405B model cannot be pulled to the VPS, so the logit lens is
computed by routing intermediate hidden states through the REMOTE lm_head/norm
modules inside the trace (model.lm_head(model.model.norm(h))). Local side only
tokenizes, builds the intervention graph, and collects small saved tensors.

Vectors: mean-difference, v_c = h_c^(L) - mean_baseline h^(L), read at the LAST
token of bare word-anchored plaintext "Tell me about {word}" (no chat wrapper, no
trailing period) so the anchor position IS the concept token. Same construction
for base and instruct. No normalization (Lindsey).

Usage (smoke, cheapest first touch):
  python3 src/run_introspection_ndif.py --model-id meta-llama/Llama-3.1-8B --smoke
Full logit-lens on the hero:
  python3 src/run_introspection_ndif.py --model-id meta-llama/Llama-3.1-405B \
      --alphas 2,4,6,8 --out-dir results/introspection
Positive control (must show detection):
  python3 src/run_introspection_ndif.py --model-id meta-llama/Llama-3.1-70B-Instruct \
      --do-generate --hard
"""
import argparse
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
CONCEPTS_FILE = ROOT / "data" / "introspection_concepts.json"
PROMPTS_FILE = ROOT / "data" / "introspection_prompts.json"


def load_secret(env_key, fallback_path):
    v = os.environ.get(env_key)
    if v:
        return v.strip()
    p = Path(fallback_path)
    if p.exists():
        return p.read_text().strip()
    raise RuntimeError(f"secret {env_key} not in env and {fallback_path} missing")


def is_instruct(model_id):
    low = model_id.lower()
    return "instruct" in low or low.endswith("-it") or "-it-" in low or "chat" in low


def render_user_assistant(tokenizer, user_text, instruct):
    """Return a prompt string ending right where the assistant is about to speak."""
    if instruct:
        msgs = [{"role": "user", "content": user_text}]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return f"User: {user_text}\nAssistant:"


def first_token_id(tokenizer, word):
    """Leading-space variant first-token id (what the lens should see mid-sentence)."""
    enc = tokenizer.encode(" " + word, add_special_tokens=False)
    return enc[0] if enc else None


def compute_vectors(model, concepts, baseline_words, tmpl, layer, instruct, batch):
    """Mean-difference steering vectors at `layer`, last-token residual. Returns
    {concept: list[float]} and the shared baseline mean (list[float])."""
    import torch

    def last_hidden(words):
        vecs = []
        for i in range(0, len(words), batch):
            chunk = words[i:i + batch]
            # Bare word-anchored plaintext (NO chat wrapper): the template ends on the
            # concept word, so the last real token IS the concept token. Wrapping in
            # User:/Assistant: would anchor the vector on the assistant-colon position
            # (concept-independent) and kill the signal. Same construction for both tiers.
            texts = [tmpl.format(word=w) for w in chunk]
            enc = model.tokenizer(texts, return_tensors="pt", padding=True,
                                  add_special_tokens=True)
            # Extract plain tensors + compute indices OUTSIDE the trace: referencing the
            # BatchEncoding inside the remote graph pulls in the (non-whitelisted) tokenizers module.
            input_ids, attn = enc["input_ids"], enc["attention_mask"]
            last = attn.sum(1) - 1
            row_idx = torch.arange(input_ids.shape[0])
            with model.trace({"input_ids": input_ids, "attention_mask": attn}, remote=True):
                h = model.model.layers[layer].output              # (B,T,H) tensor on this NDIF/transformers
                rows = h[row_idx, last]                            # (B,H)
                saved = rows.save()
            vecs.append(saved.float())
            print(f"  [vec] layer {layer}: {min(i + batch, len(words))}/{len(words)} words")
        return torch.cat(vecs, 0)

    base_mean = last_hidden(baseline_words).mean(0)                # (H,)
    concept_h = last_hidden(concepts)                              # (C,H)
    vecs = {c: (concept_h[j] - base_mean).tolist() for j, c in enumerate(concepts)}
    return vecs, base_mean.tolist()


def logit_lens_readout(model, probe_ids, probe_mask, inject_layer, read_layers,
                       alpha, vec, target_ids):
    """One remote forward. Inject alpha*vec at inject_layer over all positions;
    at each read layer, logit-lens the LAST position and record P(each target
    token) + top5. Returns {read_layer: {"p_targets": {tok:p}, "top5":[...]}}."""
    import torch

    tgt = torch.tensor(target_ids)  # plain CPU LongTensor; used to index a proxy (allowed on NDIF)
    out = {}
    ptgt = tv = ti = None  # top-level proxy locals, so nnsight propagates them back
    with model.trace({"input_ids": probe_ids, "attention_mask": probe_mask}, remote=True):
        # 1) Read every per-layer hidden in ascending (forward) order — ONLY .output reads,
        #    no norm/lm_head here. norm and lm_head sit AFTER all layers in the forward, so
        #    calling them between layer reads would touch a late module before an earlier
        #    layer -> nnsight MissedProviderError ("Envoy called out of order").
        hidden_list = []
        for lr in read_layers:  # read_layers is sorted ascending
            if alpha != 0 and lr == inject_layer:
                hs = model.model.layers[lr].output
                injected = hs + alpha * hs.new_tensor(vec)  # delta on hs device/dtype (shard-safe)
                model.model.layers[lr].output[:] = injected
                hidden_list.append(injected[:, -1, :])          # (1,H)
            else:
                hidden_list.append(model.model.layers[lr].output[:, -1, :])  # (1,H)
        # 2) Unembed ALL read layers with a single norm + lm_head invocation at end-of-forward.
        #    RMSNorm and lm_head are per-row, so stacking layers into the batch dim is exact.
        #    On sharded 70B/405B the read-layer hiddens live on DIFFERENT GPUs, so cat/norm/
        #    lm_head would hit a cross-device error. Move each to the LAST read layer's hidden
        #    (layer n-1, which shares the final shard with norm+lm_head); .to(ref_tensor) is
        #    sharding-safe (no hardcoded device, matches device AND dtype).
        ref = hidden_list[-1]
        H = torch.cat([h.to(ref) for h in hidden_list], 0)       # (L,H) co-located on final shard
        LL = model.lm_head(model.model.norm(H))                  # (L,V)
        P = torch.nn.functional.softmax(LL.float(), dim=-1)      # (L,V)
        ptgt = P[:, tgt].save()                                  # (L,T)
        tk = P.topk(5, dim=-1)
        tv = tk.values.save()                                   # (L,5)
        ti = tk.indices.save()                                  # (L,5)
    ptgt_v, tv_v, ti_v = ptgt.tolist(), tv.tolist(), ti.tolist()
    for i, lr in enumerate(read_layers):
        out[lr] = {
            "p_targets": {int(t): float(ptgt_v[i][j]) for j, t in enumerate(target_ids)},
            "top5": [{"tok": model.tokenizer.decode([int(ti_v[i][k])]), "p": float(tv_v[i][k])}
                     for k in range(5)],
        }
    return out


def generate_report(model, prompt_ids, prompt_mask, inject_layer, alpha, vec, max_new,
                    temperature=1.0, do_sample=True):
    """Generate with the injection persisting across all generated tokens.

    Macar samples at T=1.0 (top-p 1.0, top-k 50) and repeats trials to get a
    detection RATE, not a single greedy verdict — so do_sample defaults True."""
    gkw = dict(remote=True, max_new_tokens=max_new)
    if do_sample:
        gkw.update(do_sample=True, temperature=temperature, top_p=1.0, top_k=50)
    if alpha != 0:
        # `tracer.all()` binds an inject op to EVERY decode step and expects exactly
        # max_new_tokens of them. A coherent low-alpha report emits EOS early -> fewer
        # steps than tracer.all() bound -> step-count mismatch -> generator.output
        # collection is dropped and `out` never binds (UnboundLocalError). Forcing
        # min_new_tokens == max_new keeps decoding to full length so the counts match.
        # (alpha==0 has no tracer.all(), so leave its report free to stop naturally.)
        gkw["min_new_tokens"] = max_new
    with model.generate({"input_ids": prompt_ids, "attention_mask": prompt_mask},
                        **gkw) as tracer:
        if alpha != 0:
            # Persist the injection across EVERY generated token. The module-envoy
            # `.all()` is the deprecated model.all() path (crashes: one-arg IteratorProxy);
            # tracer.all() is the supported replacement in nnsight 0.7.
            # REBIND the output (`.output = ...`), do NOT in-place slice (`.output[:] = ...`):
            # on a SHARDED remote generate, the in-place slice under tracer.all() silently
            # corrupts generator.output collection -> the .save() writeback is dropped and
            # `out` never binds (UnboundLocalError). Rebinding assigns a fresh tensor as the
            # module output and collects cleanly. (The lens keeps the [:] slice: it works
            # under a single model.trace(), only the generate+tracer.all() path breaks.)
            with tracer.all():
                hs = model.model.layers[inject_layer].output
                model.model.layers[inject_layer].output = hs + alpha * hs.new_tensor(vec)
        out = model.generator.output.save()
    gen = out[0][prompt_ids.shape[1]:]
    return model.tokenizer.decode(gen, skip_special_tokens=True)


def build_trial_prompt(tokenizer, pj, condition_key, trial_num, instruct, prefill=None):
    """Macar's two-turn introspection prompt, ending where the assistant speaks.

    Turn 1 (user): [condition.doc_before] + macar_setup + [condition.note_after]
    Turn 2 (assistant): "Ok."
    Turn 3 (user): "Trial N: Do you detect an injected thought? ..."
    The Pearson-Vogel prompting condition is carried entirely by doc_before/note_after
    (baseline = neither). `prefill` seeds the assistant turn for forced-identification.
    """
    cond = pj["conditions"][condition_key]
    setup, ack = pj["macar_setup"], pj["macar_ack"]
    doc, note = cond.get("doc_before", ""), cond.get("note_after", "")
    user1 = (doc + "\n\n" if doc else "") + setup + ("\n\n" + note if note else "")
    # Few-shot detect-and-identify demos (hard-elicitation lever for base models):
    # inserted as prior (user Trial k / assistant) turns; the real trial is numbered
    # right after them. Demos use concepts DISJOINT from the test set (no answer leak).
    demos = pj.get("fewshot_demos", []) if cond.get("demos") else []
    real_num = (len(demos) + 1) if demos else trial_num
    trial = pj["macar_trial"].format(trial_num=real_num)
    if instruct:
        msgs = [{"role": "user", "content": user1}, {"role": "assistant", "content": ack}]
        for d in demos:
            msgs.append({"role": "user", "content": d["user"]})
            msgs.append({"role": "assistant", "content": d["assistant"]})
        msgs.append({"role": "user", "content": trial})
        s = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    else:
        s = f"User: {user1}\nAssistant: {ack}\n"
        for d in demos:
            s += f"User: {d['user']}\nAssistant: {d['assistant']}\n"
        s += f"User: {trial}\nAssistant:"
    if prefill:
        s = s + (prefill if s.endswith(" ") else " " + prefill)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="meta-llama/Llama-3.1-405B")
    ap.add_argument("--inject-layer", default="auto", help="int, or 'auto' = round(2/3 * n_layers)")
    ap.add_argument("--alphas", default="1,1.5,2",
                    help="injection strengths (nonzero). Our mean-diff vectors are UNNORMALIZED "
                         "(|v|~24 vs residual~18), so Macar's alpha=4 floods into 'brain damage'; "
                         "the coherent window on 70B is ~1-1.5. Calibrate per model. alpha=0 "
                         "control trials (FPR) are generated separately, per condition.")
    ap.add_argument("--conditions", default="baseline,pv_accurate,lipsum",
                    help="comma keys into prompts.conditions; each gets its own control trials (FPR)")
    ap.add_argument("--n-inject", type=int, default=3,
                    help="sampled injection trials per (concept, alpha, condition)")
    ap.add_argument("--n-control", type=int, default=50,
                    help="sampled control trials per condition (no injection -> FPR)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--greedy", action="store_true",
                    help="disable sampling (one deterministic reply per cell; no rates)")
    ap.add_argument("--do-forced-id", action="store_true",
                    help="also run forced-identification (assistant prefilled 'Yes...about')")
    ap.add_argument("--n-forced", type=int, default=3,
                    help="forced-id trials per (concept, alpha, condition)")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--read-stride", type=int, default=4, help="logit-lens read every k layers")
    ap.add_argument("--do-lens", action="store_true",
                    help="also run the PV latent logit-lens readout (internal representation)")
    ap.add_argument("--do-generate", action="store_true", default=True,
                    help="run the behavioral Macar readout (default on)")
    ap.add_argument("--no-generate", dest="do_generate", action="store_false")
    ap.add_argument("--n-concepts", type=int, default=0, help="0 = all")
    ap.add_argument("--vec-batch", type=int, default=20)
    ap.add_argument("--smoke", action="store_true", help="3 concepts, logit-lens only, 8B-sized")
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "introspection"))
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    os.environ.setdefault("HF_TOKEN", load_secret("HF_TOKEN", "/root/.secrets/hf_token_main"))
    import nnsight
    from nnsight import LanguageModel
    nnsight.CONFIG.API.APIKEY = load_secret("NDIF_API_KEY", "/root/.secrets/ndif_api_key")

    cj = json.loads(CONCEPTS_FILE.read_text())
    pj = json.loads(PROMPTS_FILE.read_text())
    concepts = cj["concepts"]
    if args.smoke:
        concepts = concepts[:3]
    elif args.n_concepts:
        concepts = concepts[:args.n_concepts]
    baseline = cj["baseline_words"]
    tmpl = cj["prompt_template"]

    instruct = is_instruct(args.model_id)
    tag = args.tag or args.model_id.split("/")[-1]
    alphas = [float(a) for a in args.alphas.split(",")]

    model = LanguageModel(args.model_id)
    if model.tokenizer.pad_token_id is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
    model.tokenizer.padding_side = "right"  # compute_vectors reads last real token via attn_mask.sum-1
    n_layers = model.config.num_hidden_layers
    inject_layer = round(2 / 3 * n_layers) if args.inject_layer == "auto" else int(args.inject_layer)
    read_layers = sorted(set(list(range(0, n_layers, args.read_stride)) + [inject_layer, n_layers - 1]))
    print(f"[cfg] {args.model_id} instruct={instruct} n_layers={n_layers} "
          f"inject_layer={inject_layer} read_layers={read_layers} alphas={alphas}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    vecs, base_mean = compute_vectors(model, concepts, baseline, tmpl,
                                      inject_layer, instruct, args.vec_batch)
    import math
    bn = math.sqrt(sum(x * x for x in base_mean))
    vn = {c: math.sqrt(sum(x * x for x in vecs[c])) for c in concepts}
    print(f"[vecs] {len(vecs)} concept vectors at layer {inject_layer} ({time.time()-t0:.0f}s)")
    print(f"[norm] |base_mean|={bn:.1f}  |v_c| mean={sum(vn.values())/len(vn):.1f} "
          f"range=[{min(vn.values()):.1f},{max(vn.values()):.1f}]  "
          f"=> alpha*|v| vs residual~|base_mean|")

    # ---- (optional) PV latent logit-lens readout: is the concept REPRESENTED? ----
    # PV's signature: the concept peaks in mid-layers then attenuates near the top,
    # so the model may represent it internally while still SAYING "no". Off by default
    # (behavioral Macar readout is primary); enable with --do-lens.
    target_ids = {c: first_token_id(model.tokenizer, c) for c in concepts}
    lens = []
    if args.do_lens:
        probe_text = render_user_assistant(model.tokenizer, "Say the first word that comes to mind.", instruct)
        penc = model.tokenizer(probe_text, return_tensors="pt", add_special_tokens=not instruct)
        for c in concepts:
            tids = [t for t in [target_ids[c]] if t is not None]
            if not tids:
                print(f"  [skip] no first-token id for {c!r}")
                continue
            for alpha in [0.0] + alphas:
                r = logit_lens_readout(model, penc["input_ids"], penc["attention_mask"],
                                       inject_layer, read_layers, alpha, vecs[c], tids)
                lens.append({"concept": c, "target_id": target_ids[c], "alpha": alpha, "by_layer": r})
            print(f"  [lens] {c}: alphas {[0.0]+alphas} done")

    conditions = args.conditions.split(",")
    result = {
        "model_id": args.model_id, "tag": tag, "backend": "ndif", "instruct": instruct,
        "n_layers": n_layers, "inject_layer": inject_layer, "read_layers": read_layers,
        "alphas": alphas, "conditions": conditions, "temperature": args.temperature,
        "n_inject": args.n_inject, "n_control": args.n_control,
        "concepts": concepts, "target_ids": target_ids,
        "logit_lens": lens,
    }

    # Persist the (expensive) lens result BEFORE generation. Remote generation is
    # the fragile step — a failure there must never destroy the lens data.
    suffix = "smoke" if args.smoke else f"L{inject_layer}"
    out_path = out_dir / f"{tag}__{suffix}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[saved-lens] {out_path} ({len(lens)} lens rows, {time.time()-t0:.0f}s)")

    if args.do_generate:
        placeholder = next(iter(vecs.values()))  # unused when alpha==0 (control)
        trials = []
        result["trials"] = trials

        def run_cell(condition, concept, alpha, kind, trial_num, prefill=None):
            prompt = build_trial_prompt(model.tokenizer, pj, condition, trial_num, instruct, prefill)
            enc = model.tokenizer(prompt, return_tensors="pt", add_special_tokens=not instruct)
            vec = vecs[concept] if (concept is not None and alpha != 0) else placeholder
            rec = {"condition": condition, "concept": concept, "alpha": alpha,
                   "kind": kind, "trial": trial_num}
            last_err = None
            for _ in range(2):  # retry once: remote gen occasionally drops a cell transiently
                try:
                    rec["reply"] = generate_report(
                        model, enc["input_ids"], enc["attention_mask"], inject_layer,
                        alpha, vec, args.max_new_tokens,
                        temperature=args.temperature, do_sample=not args.greedy)
                    trials.append(rec)
                    return
                except Exception as e:  # one fragile remote-gen cell must not abort the run
                    last_err = f"{type(e).__name__}: {str(e)[:200]}"
            rec["reply"] = None
            rec["error"] = last_err
            trials.append(rec)
            print(f"  [gen-ERR] {condition}/{concept} a={alpha} {kind}: {last_err[:110]}")

        def checkpoint():
            out_path.write_text(json.dumps(result, indent=2))

        for cond in conditions:
            # control trials (no injection) -> FPR, measured PER CONDITION (a strong
            # PV/hints prompt can raise FPR; that must be caught, not assumed away).
            for t in range(args.n_control):
                run_cell(cond, None, 0.0, "control", t)
            checkpoint()
            print(f"  [{cond}] {args.n_control} control trials done")
            # injection trials -> TPR and introspection (detect AND identify)
            for c in concepts:
                for alpha in alphas:
                    for t in range(args.n_inject):
                        run_cell(cond, c, alpha, "injection", t)
                checkpoint()
            print(f"  [{cond}] injection: {len(concepts)}c x {len(alphas)}a x {args.n_inject}t done")
            # forced identification (optional): prefill the 'Yes...about' turn to
            # isolate NAMING ability from willingness to report detection.
            if args.do_forced_id:
                for c in concepts:
                    for alpha in alphas:
                        for t in range(args.n_forced):
                            run_cell(cond, c, alpha, "forced_id", t, prefill=pj["forced_id_prefill"])
                    checkpoint()
                print(f"  [{cond}] forced-id done")

    n_trials = len(result.get("trials", []))
    print(f"[done] {tag}: {len(lens)} lens rows"
          f"{f' + {n_trials} trials' if args.do_generate else ''} ({time.time()-t0:.0f}s)")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
