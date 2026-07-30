"""run_introspection_pv.py -- Pearson-Vogel 2602.20031 introspection replication.

Faithful to PV's protocol (which they replicate on Llama-3.3-70B-Instruct), NOT the
Macar free-form+judge path. The two differ on the axis that dominates false positives:

  Macar path (run_introspection_ndif.py): continuous injection during GENERATION, then
    a Claude judge reads the free-form reply for "did it claim detection?". On Llama this
    inflates FPR to ~0.42 because Llama confabulates hedged detections in prose.

  PV path (this file): the assistant turn is PREFILLED with "The answer is", and we read
    the SINGLE-TOKEN logit  P(" yes") vs P(" no")  at that position -- no generation, no
    judge. Llama's top token is "no" ~99.8% on controls, so FPR collapses to ~0. This is
    the same single-token-logprob methodology as the repo's coinflip probe.

Injection is KV-cache-remove style: the concept vector is added at middle layer(s) over
the FIRST-TURN (context) token positions ONLY, and is absent at the query+answer
positions. The answer position attends back to the injected context but is not itself
steered, so a rise in P(yes) reflects retrospective detection rather than live steering
pushing the "yes" logit (PV's core confound control).

Llama-3.3-70B cell (PV Appendix 13, "Additional Models"): ACCURATE_Mechanism + No_Document
is the clean replication cell -- 75.5% balanced accuracy, comparable to their primary
Qwen-32B. The pro-introspection document INVERTS on Llama (helps Qwen, drops Llama to
38.0%), opposite to Qwen. So the replication cell is framing=accurate,doc=none; the
accurate+pro_doc drop is the Llama-specific inversion contrast. (We run Llama-3.1-70B-it,
not PV's 3.3, to keep the 3.1 family constant with the gated 3.1-405B-base arm.)

Measurement is a single remote forward per (concept,condition) -- ~10x cheaper than the
generate-based Macar path. Control (alpha=0) is concept-independent: one forward/condition.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
CONCEPTS_FILE = ROOT / "data" / "introspection_concepts.json"
PROMPTS_FILE = ROOT / "data" / "introspection_prompts.json"
sys.path.insert(0, str(ROOT / "src"))
from run_introspection_ndif import load_secret, is_instruct, compute_vectors  # noqa: E402


def build_pv_prompt(tokenizer, pv, framing_key, doc_key, instruct):
    """Return (full_prompt_str, ctx_prompt_str).

    full = the whole 2-turn chat ending at the "The answer is" prefill (read P(yes/no)
           at its next position).
    ctx  = only the first (injected) turn -- user1 + empty assistant close -- used to
           locate the token boundary so injection covers context positions only.
    """
    fr = pv["framings"][framing_key]
    doc = pv["info_docs"][doc_key]
    user1 = fr["intro"] + (("\n\n" + doc) if doc else "") + "\n\n" + fr["suffix"]
    fmt = pv["format_instruction"].replace("{followup}", fr["followup"])
    user2 = pv["user2_lead"] + "\n\nTrial 1: " + fr["question"] + " " + fmt
    prefill = pv["answer_prefill"]
    if instruct:
        ctx_msgs = [{"role": "user", "content": user1},
                    {"role": "assistant", "content": ""}]
        ctx_str = tokenizer.apply_chat_template(ctx_msgs, tokenize=False,
                                                add_generation_prompt=False)
        full_msgs = [{"role": "user", "content": user1},
                     {"role": "assistant", "content": ""},
                     {"role": "user", "content": user2}]
        full_str = tokenizer.apply_chat_template(full_msgs, tokenize=False,
                                                 add_generation_prompt=True) + prefill
    else:
        # base model: no chat template. Context = the framing turn; the concept is
        # injected over it, then the query turn follows un-steered.
        ctx_str = f"User: {user1}\nAssistant:"
        full_str = ctx_str + f"\nUser: {user2}\nAssistant: {prefill}"
    return full_str, ctx_str


def common_prefix_len(tokenizer, ctx_str, full_str, instruct):
    """#tokens of full_str that belong to the injected context turn.

    Robust to chat-template quirks: take the longest shared token prefix of ctx and full
    rather than assuming ctx tokenizes as an exact prefix of full."""
    a = tokenizer(ctx_str, add_special_tokens=not instruct)["input_ids"]
    b = tokenizer(full_str, add_special_tokens=not instruct)["input_ids"]
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def yn_ids(tokenizer):
    """Token ids for the yes / no answer tokens right after 'The answer is'."""
    def ids(words):
        s = set()
        for w in words:
            e = tokenizer.encode(w, add_special_tokens=False)
            if e:
                s.add(e[0])
        return sorted(s)
    return ids([" yes", " Yes"]), ids([" no", " No"])


def compute_vectors_multi(model, concepts, baseline_words, tmpl, layers, instruct, batch):
    """Per-layer mean-difference steering vectors: v_c^(L) = h_c^(L) - mean_baseline h^(L)
    at EACH layer L in `layers`, read at the last anchor token of "Tell me about {word}".
    One remote trace per word-batch reads all layers at once (ascending order -> no
    out-of-order envoy call). Returns ({L: {concept: list[float]}}, {L: base_mean list})."""
    import torch
    layers = sorted(layers)

    def last_hidden_multi(words):
        acc = {L: [] for L in layers}
        for i in range(0, len(words), batch):
            chunk = words[i:i + batch]
            texts = [tmpl.format(word=w) for w in chunk]
            enc = model.tokenizer(texts, return_tensors="pt", padding=True,
                                  add_special_tokens=True)
            input_ids, attn = enc["input_ids"], enc["attention_mask"]
            last = attn.sum(1) - 1
            row_idx = torch.arange(input_ids.shape[0])
            # Save ONE aggregate tensor, not a dict of per-layer .save() proxies: on remote
            # nnsight the dict-of-saves does not register as graph output (empty download).
            # Stack all read layers into (nL,B,H) inside the trace and save that (mirrors the
            # proven logit_lens aggregation). Sharded 70B: layer outputs live on different
            # GPUs, so .to(ref) co-locates every layer on the last read layer's device first.
            stacked = None
            with model.trace({"input_ids": input_ids, "attention_mask": attn}, remote=True):
                rows = [model.model.layers[L].output[row_idx, last] for L in layers]  # ascending
                ref = rows[-1]
                stacked = torch.stack([r.to(ref) for r in rows], 0).save()  # (nL,B,H)
            st = stacked.float()
            for k, L in enumerate(layers):
                acc[L].append(st[k])
            print(f"  [vec] {min(i + batch, len(words))}/{len(words)} words @ {len(layers)} layers")
        return {L: torch.cat(acc[L], 0) for L in layers}

    base = last_hidden_multi(baseline_words)   # {L: (Nb,H)}
    conc = last_hidden_multi(concepts)         # {L: (C,H)}
    vec_by_layer, base_mean_by_layer = {}, {}
    for L in layers:
        bm = base[L].mean(0)
        base_mean_by_layer[L] = bm.tolist()
        vec_by_layer[L] = {c: (conc[L][j] - bm).tolist() for j, c in enumerate(concepts)}
    return vec_by_layer, base_mean_by_layer


# --- repeng-faithful vectors (PV's actual method) --------------------------------------
# PV trains steering vectors with the repeng library: for each concept c, the PRINCIPAL
# COMPONENT of centered activation-PAIR differences between a prompt "think about {c}" and
# a neutral "think about anything", sampled over DIVERSE assistant prefixes. This denoises
# the per-layer direction (a single-template mean-diff is a noisy 1-sample estimate whose
# per-layer directions don't cohere when stacked across the middle third; the PCA over many
# prefixes recovers a layer-consistent concept axis). Vectors are UNIT norm, so the
# injection alpha is a residual-norm-scaled coefficient (repeng injects a uniform coeff).
_USER_POS = ["Think about {c}.",
             "Please think carefully about {c}.",
             "I'd like you to think about {c}.",
             "Focus your attention on {c}."]
_USER_NEG = ["Think about anything.",
             "Please think carefully about anything.",
             "I'd like you to think about anything.",
             "Focus your attention on anything."]
# Neutral assistant continuation, truncated at word boundaries -> diverse prefixes. The
# prefix never names a concept; the concept signal reaches the last token only via attention
# back to the (differing) user turn -- exactly the KV-cache signal we later inject.
_ASST_BASE = ("Sure, I am reflecting on this now and letting the idea take shape in my "
              "mind as I consider it slowly and carefully from a few different angles.")
_ASST_TRUNC_WORDS = [2, 3, 5, 7, 9, 12, 15, 18]


def _asst_prefixes():
    ws = _ASST_BASE.split()
    return [" ".join(ws[:k]) for k in _ASST_TRUNC_WORDS]


def compute_repeng_vectors(model, concepts, layers, instruct, batch, seed=0):
    """PV/repeng per-layer UNIT steering vectors: {L: {concept: unit_list}}.

    Per concept: build paired (positive="think about c", negative="think about anything")
    strings sharing a diverse assistant prefix, interleaved [pos0,neg0,pos1,neg1,...]. One
    remote trace reads the last-token hidden of every string at every layer, differences the
    pairs INSIDE the trace (saving only (nL,n_pairs,H) ~ tens of MB, not the full hiddens),
    then PCA(1) per layer on the centered differences gives the concept direction; sign is
    fixed so mean(pos-neg) projects positive (repeng's convention). Also returns per-concept
    PC1 explained-variance-ratio as a vector-quality readout."""
    import torch
    import numpy as np
    layers = sorted(layers)
    batch = max(2, batch - (batch % 2))  # even -> pos/neg interleave stays aligned per chunk
    tok = model.tokenizer

    def gen_prefix(user_content):
        if instruct:
            return tok.apply_chat_template([{"role": "user", "content": user_content}],
                                           tokenize=False, add_generation_prompt=True)
        return user_content + "\n"  # base model: plaintext, no chat template

    prefixes = _asst_prefixes()
    vec_by_layer = {L: {} for L in layers}
    evr = {}  # concept -> mean PC1 explained-variance-ratio across layers
    for c in concepts:
        strs = []
        for up, un in zip(_USER_POS, _USER_NEG):
            gp, gn = gen_prefix(up.format(c=c)), gen_prefix(un)
            for pfx in prefixes:
                strs.append(gp + pfx)   # positive (even index)
                strs.append(gn + pfx)   # negative (odd index)
        n_pairs = len(strs) // 2
        diffs_layers = []  # list over batches of (nL, n_pairs_batch, H)
        for i in range(0, len(strs), batch):
            chunk = strs[i:i + batch]
            enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=not instruct)
            input_ids, attn = enc["input_ids"], enc["attention_mask"]
            last = attn.sum(1) - 1
            row_idx = torch.arange(input_ids.shape[0])
            saved = None
            with model.trace({"input_ids": input_ids, "attention_mask": attn}, remote=True):
                rows = [model.model.layers[L].output[row_idx, last] for L in layers]  # (B,H) each
                ref = rows[-1]
                stacked = torch.stack([r.to(ref) for r in rows], 0)         # (nL,B,H)
                d = stacked[:, ::2, :] - stacked[:, 1::2, :]                 # (nL,pairs,H)
                saved = d.save()
            diffs_layers.append(saved.float())
        D_all = torch.cat(diffs_layers, 1).numpy()  # (nL, n_pairs, H)
        evs = []
        for k, L in enumerate(layers):
            D = D_all[k]                              # (n_pairs, H)
            Dc = D - D.mean(0, keepdims=True)         # center (sklearn PCA convention)
            U, S, Vt = np.linalg.svd(Dc, full_matrices=False)
            pc = Vt[0]                                # unit-norm PC1
            if float(D.mean(0) @ pc) < 0:             # sign: mean(pos-neg) projects positive
                pc = -pc
            vec_by_layer[L][c] = pc.astype(np.float32).tolist()
            evs.append(float(S[0] ** 2 / (S ** 2).sum()))
        evr[c] = sum(evs) / len(evs)
        print(f"  [repeng] {c}: {n_pairs} pairs x {len(layers)} layers | mean PC1 EVR={evr[c]:.2f}")
    return vec_by_layer, evr


def detect_yesno(model, input_ids, attn, ctx_len, inject_layers, alpha, vec_by_layer,
                 yes_ids, no_ids, n_layers):
    """One remote forward. Inject alpha*v_c^(L) at each inject layer L over positions
    [0:ctx_len) only (per-layer vector via `vec_by_layer[L]`), then read P(yes)/P(no) at
    the final position via norm+lm_head (validated logit-lens unembed path). alpha==0 =>
    no injection (control); vec_by_layer unused. Returns (p_yes, p_no)."""
    import torch
    T = input_ids.shape[1]
    posmask = torch.zeros(T)
    posmask[:ctx_len] = 1.0  # plain CPU tensor built OUTSIDE the trace
    yt, nt = torch.tensor(yes_ids), torch.tensor(no_ids)
    py = pn = None
    with model.trace({"input_ids": input_ids, "attention_mask": attn}, remote=True):
        if alpha != 0:
            for L in inject_layers:  # ascending
                hs = model.model.layers[L].output                     # (1,T,H)
                add = alpha * hs.new_tensor(vec_by_layer[L])          # (H,) per-layer vec
                injected = hs + posmask.view(1, T, 1).to(hs) * add    # steer ctx positions
                model.model.layers[L].output[:] = injected
        final_h = model.model.layers[n_layers - 1].output[:, -1, :]   # (1,H)
        logits = model.lm_head(model.model.norm(final_h))             # (1,V)
        P = torch.nn.functional.softmax(logits.float(), dim=-1)       # (1,V)
        py = P[0, yt].sum().save()
        pn = P[0, nt].sum().save()
    return float(py), float(pn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="meta-llama/Llama-3.1-70B-Instruct")
    ap.add_argument("--framings", default="vague,accurate")
    ap.add_argument("--docs", default="none")
    ap.add_argument("--inject-layers", default="auto",
                    help="comma ints, or 'auto' = middle third round(0.4*n)..round(0.66*n) "
                         "single mid layer round(0.5*n) by default; pass e.g. 32,40,48 for a range")
    ap.add_argument("--single-layer", action="store_true",
                    help="auto => just the mid layer round(0.5*n) (cheap validation)")
    ap.add_argument("--vectors", default="repeng", choices=["repeng", "meandiff", "random"],
                    help="repeng = PV-faithful PCA-over-diverse-pairs UNIT vectors (default); "
                         "meandiff = single-template h_c - h_baseline (non-unit, weaker); "
                         "random = seeded random UNIT vectors (PV noise control: same norm, "
                         "no concept content -> P(yes) must stay ~control if signal is real)")
    ap.add_argument("--alpha", type=float, default=8.0,
                    help="injection coefficient. For --vectors repeng the vector is unit-norm "
                         "so alpha is per-layer injection magnitude in residual-norm units.")
    ap.add_argument("--alphas", default=None,
                    help="comma list to sweep alpha (overrides --alpha); e.g. 4,8,12,16")
    ap.add_argument("--n-concepts", type=int, default=9)
    ap.add_argument("--vec-batch", type=int, default=20)
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "introspection"))
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    os.environ.setdefault("HF_TOKEN", load_secret("HF_TOKEN", "/root/.secrets/hf_token_main"))
    import nnsight
    from nnsight import LanguageModel
    nnsight.CONFIG.API.APIKEY = load_secret("NDIF_API_KEY", "/root/.secrets/ndif_api_key")

    cj = json.loads(CONCEPTS_FILE.read_text())
    pj = json.loads(PROMPTS_FILE.read_text())
    pv = pj["pv"]
    concepts = cj["concepts"][:args.n_concepts]
    baseline = cj["baseline_words"]
    tmpl = cj["prompt_template"]

    instruct = is_instruct(args.model_id)
    tag = args.tag or ("pv_" + args.model_id.split("/")[-1])
    framings = args.framings.split(",")
    docs = args.docs.split(",")
    alphas = [float(a) for a in args.alphas.split(",")] if args.alphas else [args.alpha]

    model = LanguageModel(args.model_id)
    if model.tokenizer.pad_token_id is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
    model.tokenizer.padding_side = "right"
    tok = model.tokenizer
    n_layers = model.config.num_hidden_layers

    if args.inject_layers == "auto":
        mid = round(0.5 * n_layers)
        # PV inject over the middle third at EVERY layer (Qwen-32B: 21-42/64 ~ 0.33-0.66).
        # Each layer gets its OWN mean-diff vector (reusing one vector across layers is
        # geometrically wrong; residual directions differ per layer).
        inject_layers = [mid] if args.single_layer else \
            list(range(round(0.33 * n_layers), round(0.66 * n_layers) + 1))
    else:
        inject_layers = [int(x) for x in args.inject_layers.split(",")]
    inject_layers = sorted(set(inject_layers))  # ascending -> no out-of-order envoy call in trace

    yes_ids, no_ids = yn_ids(tok)
    print(f"[cfg] {args.model_id} instruct={instruct} n_layers={n_layers} "
          f"inject_layers={inject_layers} alphas={alphas}")
    print(f"[yn] yes_ids={yes_ids} ({[tok.decode([i]) for i in yes_ids]})  "
          f"no_ids={no_ids} ({[tok.decode([i]) for i in no_ids]})")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Per-layer steering vectors at every inject layer L (one trace-set reads all).
    import math
    def _n(v):
        return math.sqrt(sum(x * x for x in v))
    vmid = inject_layers[len(inject_layers) // 2]
    if args.vectors == "random":
        import numpy as np
        H = model.config.hidden_size
        rng = np.random.default_rng(0)
        vec_by_layer = {L: {} for L in inject_layers}
        for L in inject_layers:
            for c in concepts:
                g = rng.standard_normal(H).astype(np.float32)
                vec_by_layer[L][c] = (g / np.linalg.norm(g)).tolist()  # random UNIT vec
        print(f"[vecs] random UNIT vectors (noise control, seed=0): {len(concepts)} x "
              f"{len(inject_layers)} layers L{inject_layers[0]}-{inject_layers[-1]} "
              f"(no NDIF; {time.time()-t0:.0f}s)")
    elif args.vectors == "repeng":
        vec_by_layer, evr = compute_repeng_vectors(
            model, concepts, inject_layers, instruct, args.vec_batch)
        vn = sum(_n(vec_by_layer[vmid][c]) for c in concepts) / len(concepts)
        mean_evr = sum(evr.values()) / len(evr)
        print(f"[vecs] repeng UNIT vectors: {len(concepts)} x {len(inject_layers)} layers "
              f"L{inject_layers[0]}-{inject_layers[-1]} | @L{vmid} |v_c|={vn:.2f} "
              f"| mean PC1 EVR={mean_evr:.2f} ({time.time()-t0:.0f}s)")
    else:
        vec_by_layer, base_mean_by_layer = compute_vectors_multi(
            model, concepts, baseline, tmpl, inject_layers, instruct, args.vec_batch)
        bn = _n(base_mean_by_layer[vmid])
        vn = sum(_n(vec_by_layer[vmid][c]) for c in concepts) / len(concepts)
        print(f"[vecs] meandiff vectors: {len(concepts)} x {len(inject_layers)} layers "
              f"L{inject_layers[0]}-{inject_layers[-1]} | @L{vmid}: |base_mean|={bn:.1f} "
              f"|v_c| mean={vn:.1f} ({time.time()-t0:.0f}s)")

    result = {
        "model_id": args.model_id, "tag": tag, "backend": "ndif", "instruct": instruct,
        "protocol": "pearson_vogel_pyes", "vectors": args.vectors, "n_layers": n_layers,
        "inject_layers": inject_layers, "alphas": alphas,
        "framings": framings, "docs": docs, "concepts": concepts,
        "yes_ids": yes_ids, "no_ids": no_ids, "rows": [],
    }
    out_path = out_dir / f"{tag}__pv.json"

    def ckpt():
        out_path.write_text(json.dumps(result, indent=2))

    for framing in framings:
        for doc in docs:
            full_str, ctx_str = build_pv_prompt(tok, pv, framing, doc, instruct)
            enc = tok(full_str, return_tensors="pt", add_special_tokens=not instruct)
            ids, attn = enc["input_ids"], enc["attention_mask"]
            ctx_len = common_prefix_len(tok, ctx_str, full_str, instruct)
            T = ids.shape[1]
            for alpha in alphas:
                # control once per (framing,doc,alpha) -- concept-independent (no injection)
                pc_y, pc_n = detect_yesno(model, ids, attn, ctx_len, inject_layers,
                                          0.0, {}, yes_ids, no_ids, n_layers)
                result["rows"].append({"framing": framing, "doc": doc, "alpha": alpha,
                                       "kind": "control", "concept": None,
                                       "p_yes": pc_y, "p_no": pc_n, "ctx_len": ctx_len, "T": T})
                det = 0
                for c in concepts:
                    vbl_c = {L: vec_by_layer[L][c] for L in inject_layers}
                    pi_y, pi_n = detect_yesno(model, ids, attn, ctx_len, inject_layers,
                                              alpha, vbl_c, yes_ids, no_ids, n_layers)
                    result["rows"].append({"framing": framing, "doc": doc, "alpha": alpha,
                                           "kind": "injection", "concept": c,
                                           "p_yes": pi_y, "p_no": pi_n})
                    det += pi_y > pi_n
                ckpt()
                mean_inj = sum(r["p_yes"] for r in result["rows"]
                               if r["kind"] == "injection" and r["framing"] == framing
                               and r["doc"] == doc and r["alpha"] == alpha) / len(concepts)
                print(f"  [{framing}/{doc} a={alpha}] control P(yes)={pc_y:.3f} "
                      f"(P(no)={pc_n:.3f}) | inj mean P(yes)={mean_inj:.3f} | "
                      f"detect {det}/{len(concepts)} (P(yes)>P(no)) | shift={mean_inj-pc_y:+.3f}")

    ckpt()
    print(f"[saved] {out_path} ({len(result['rows'])} rows, {time.time()-t0:.0f}s)")
    _summary(result)


def _summary(result):
    """Balanced-accuracy-style readout per (framing,doc,alpha)."""
    from collections import defaultdict
    ctrl = {}  # (f,d,a) -> (p_yes,p_no)
    inj = defaultdict(list)  # (f,d,a) -> [(p_yes,p_no),...]
    for r in result["rows"]:
        k = (r["framing"], r["doc"], r["alpha"])
        if r["kind"] == "control":
            ctrl[k] = (r["p_yes"], r["p_no"])
        else:
            inj[k].append((r["p_yes"], r["p_no"]))
    print("\n=== PV P(yes) summary (detect = P(yes)>P(no)) ===")
    print("  framing/doc         alpha  ctrl_P(yes)  inj_mean_P(yes)  TPR   FPR   bal_acc")
    for k in sorted(inj):
        f, d, a = k
        cy, cn = ctrl.get(k, (float("nan"), float("nan")))
        ys = inj[k]
        tpr = sum(y > n for y, n in ys) / len(ys)
        fpr = 1.0 if cy > cn else 0.0
        bal = 0.5 * (tpr + (1 - fpr))
        my = sum(y for y, _ in ys) / len(ys)
        print(f"  {f+'/'+d:18s}  a={a:<4}  {cy:8.3f}     {my:8.3f}      "
              f"{tpr:.2f}  {fpr:.2f}   {bal:.2f}")


if __name__ == "__main__":
    main()
