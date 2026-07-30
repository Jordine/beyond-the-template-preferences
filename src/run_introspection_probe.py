"""run_introspection_probe.py -- represent-vs-report probe.

Forward-only (no backward; NDIF runs inference-mode). For each trial we inject the concept
steering vector over the CONTEXT positions (exactly as the PV detect probe does) and then
SAVE the read-position residual (the '...The answer is' position whose next token is yes/no)
at a spread of layers -- in ONE ascending pass, interleaving injection-writes and read-saves
so nnsight never makes an out-of-order envoy call.

The behavioral readout P(yes)/P(no) is saved alongside every trial, so the crux question can
be asked offline (analyze_introspection_probe.py):

  Is the injected concept LINEARLY RECOVERABLE at the read position on HELD-OUT concepts,
  even when the model behaviorally says "no"?

  - recoverable + behavior says "no"  => representation present, report absent
                                          => capacity is LATENT; post-training installs ACCESS
  - not recoverable                   => the injected content never reaches the report site
                                          linearly => behavioral elicitation is hopeless

Identity readout is training-free and held-out-valid: Δh = h(inj) - h(ctrl,same framing);
identify = argmax_c cosine(Δh_L, v_c^L). Random-vector controls (structure-less injections)
test that any match is concept-SPECIFIC, not just "some perturbation arrived".

Runs on 405B-base (the target) and on 70B-it (apparatus validation: where behavior already
detects, the probe MUST light up -- else it is broken).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
CONCEPTS_FILE = ROOT / "data" / "introspection_concepts.json"
PROMPTS_FILE = ROOT / "data" / "introspection_prompts.json"
sys.path.insert(0, str(ROOT / "src"))
from run_introspection_ndif import load_secret, is_instruct  # noqa: E402
from run_introspection_pv import (  # noqa: E402
    build_pv_prompt, common_prefix_len, yn_ids, compute_repeng_vectors,
)


def auto_read_layers(n_layers, k=14):
    """~k layers evenly spread across depth, always including the FINAL layer (report site)."""
    pts = set(int(round(x)) for x in np.linspace(0, n_layers - 1, k))
    pts.add(n_layers - 1)
    return sorted(pts)


def collect_readpos(model, input_ids, attn, ctx_len, inject_layers, read_layers,
                    alpha, vec_c, yes_ids, no_ids, n_layers):
    """ONE remote forward. Inject alpha*vec_c[L] over positions [0:ctx_len) at each inject
    layer L (skip if alpha==0), and save the read-position (-1) hidden at every read layer.
    Single ascending pass over sorted(inject ∪ read): inject-write then read-save per layer,
    so envoy calls stay ascending. Returns (p_yes, p_no, h[nR,H] float32)."""
    import torch
    T = input_ids.shape[1]
    posmask = torch.zeros(T)
    posmask[:ctx_len] = 1.0
    yt, nt = torch.tensor(yes_ids), torch.tensor(no_ids)
    inject_set = set(inject_layers)
    read_set = sorted(read_layers)
    order = sorted(inject_set | set(read_set))
    assert (n_layers - 1) in read_set, "read_layers must include the final layer"
    reads = {}
    py = pn = h_saved = None
    with model.trace({"input_ids": input_ids, "attention_mask": attn}, remote=True):
        for L in order:
            if alpha != 0 and L in inject_set:
                hs = model.model.layers[L].output
                add = alpha * hs.new_tensor(vec_c[L])
                injected = hs + posmask.view(1, T, 1).to(hs) * add
                model.model.layers[L].output[:] = injected
                cur = injected
            else:
                cur = model.model.layers[L].output
            if L in read_set:
                reads[L] = cur[:, -1, :]  # (1,H); reads the injected value at inject∩read layers
        rl = [reads[L] for L in read_set]
        ref = rl[-1]
        h_saved = torch.stack([r.to(ref) for r in rl], 0).save()   # (nR,1,H)
        final_h = reads[n_layers - 1]                              # reuse -> no extra envoy call
        logits = model.lm_head(model.model.norm(final_h))
        P = torch.nn.functional.softmax(logits.float(), dim=-1)
        py = P[0, yt].sum().save()
        pn = P[0, nt].sum().save()
    h = h_saved.float().squeeze(1).numpy()  # (nR,H)
    return float(py), float(pn), h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="meta-llama/Llama-3.1-405B")
    ap.add_argument("--framings", default="vague,accurate")
    ap.add_argument("--docs", default="none,poetic")
    ap.add_argument("--alphas", default="6,10")
    ap.add_argument("--inject-layers", default="auto",
                    help="'auto' = middle third round(0.33n)..round(0.66n); or comma ints")
    ap.add_argument("--read-layers", default="auto",
                    help="'auto' = ~14 layers spread across depth incl final; or comma ints")
    ap.add_argument("--n-random-ctrl", type=int, default=3,
                    help="random UNIT-vector injections per (framing,doc,alpha): concept-"
                         "specificity control (any identity match must beat these)")
    ap.add_argument("--n-concepts", type=int, default=9)
    ap.add_argument("--n-demo", type=int, default=4, help="first n concepts = train/demo pool")
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
    demo_pool = concepts[:args.n_demo]
    eval_pool = concepts[args.n_demo:]
    instruct = is_instruct(args.model_id)
    tag = args.tag or ("probe_" + args.model_id.split("/")[-1])
    framings = args.framings.split(",")
    docs = args.docs.split(",")
    alphas = [float(a) for a in args.alphas.split(",")]

    model = LanguageModel(args.model_id)
    if model.tokenizer.pad_token_id is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
    model.tokenizer.padding_side = "right"
    tok = model.tokenizer
    n_layers = model.config.num_hidden_layers

    if args.inject_layers == "auto":
        inject_layers = list(range(round(0.33 * n_layers), round(0.66 * n_layers) + 1))
    else:
        inject_layers = [int(x) for x in args.inject_layers.split(",")]
    inject_layers = sorted(set(inject_layers))
    read_layers = (auto_read_layers(n_layers) if args.read_layers == "auto"
                   else sorted(set(int(x) for x in args.read_layers.split(","))))
    if (n_layers - 1) not in read_layers:
        read_layers.append(n_layers - 1)
    vec_layers = sorted(set(inject_layers) | set(read_layers))  # vectors needed at BOTH

    yes_ids, no_ids = yn_ids(tok)
    print(f"[cfg] {args.model_id} instruct={instruct} n_layers={n_layers} H={model.config.hidden_size}")
    print(f"[cfg] inject L{inject_layers[0]}-{inject_layers[-1]} ({len(inject_layers)}) | "
          f"read {read_layers} | alphas={alphas} | fr={framings} docs={docs} rand_ctrl={args.n_random_ctrl}")
    print(f"[cfg] demo_pool={demo_pool} eval_pool={eval_pool}")
    t0 = time.time()

    # concept repeng UNIT vectors at every vec layer (one trace-set reads all layers)
    vec_by_layer, evr = compute_repeng_vectors(model, concepts, vec_layers, instruct, args.vec_batch)
    mean_evr = sum(evr.values()) / len(evr)
    print(f"[vecs] repeng {len(concepts)} concepts x {len(vec_layers)} layers | mean EVR={mean_evr:.2f} "
          f"({time.time()-t0:.0f}s)")
    # random UNIT vectors for concept-specificity controls (only needed at inject layers)
    rng = np.random.default_rng(0)
    H = model.config.hidden_size
    rand_vecs = []  # list of {L: unit_list}
    for r in range(args.n_random_ctrl):
        d = {}
        for L in inject_layers:
            g = rng.standard_normal(H).astype(np.float32)
            d[L] = (g / np.linalg.norm(g)).tolist()
        rand_vecs.append(d)

    rows = []       # metadata per sample
    hiddens = []    # (nR,H) per sample, aligned to rows
    nR = len(read_layers)

    def add_sample(kind, concept, framing, doc, alpha, ids, attn, ctx_len, vec_c):
        py, pn, h = collect_readpos(model, ids, attn, ctx_len, inject_layers, read_layers,
                                    alpha, vec_c, yes_ids, no_ids, n_layers)
        rows.append({"idx": len(rows), "kind": kind, "concept": concept, "framing": framing,
                     "doc": doc, "alpha": alpha, "p_yes": py, "p_no": pn,
                     "in_eval": concept in eval_pool if concept else None})
        hiddens.append(h.astype(np.float16))
        return py, pn

    for framing in framings:
        for doc in docs:
            full_str, ctx_str = build_pv_prompt(tok, pv, framing, doc, instruct)
            enc = tok(full_str, return_tensors="pt", add_special_tokens=not instruct)
            ids, attn = enc["input_ids"], enc["attention_mask"]
            ctx_len = common_prefix_len(tok, ctx_str, full_str, instruct)
            # one control (alpha=0) per (framing,doc)
            cy, cn = add_sample("control", None, framing, doc, 0.0, ids, attn, ctx_len, {})
            print(f"  [ctrl {framing}/{doc}] P(yes)={cy:.3f} P(no)={cn:.3f}")
            for alpha in alphas:
                det = 0
                for c in concepts:
                    vc = {L: vec_by_layer[L][c] for L in inject_layers}
                    py, pn = add_sample("inj", c, framing, doc, alpha, ids, attn, ctx_len, vc)
                    det += py > pn
                for r in range(args.n_random_ctrl):
                    add_sample("rand", f"rand{r}", framing, doc, alpha, ids, attn, ctx_len, rand_vecs[r])
                print(f"  [inj {framing}/{doc} a={alpha}] detect {det}/{len(concepts)} "
                      f"| {time.time()-t0:.0f}s")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # concept vectors at READ layers only (for the offline cosine identity readout)
    ridx = {L: k for k, L in enumerate(read_layers)}
    cvecs = np.stack([[vec_by_layer[L][c] for L in read_layers] for c in concepts], 0).astype(np.float16)
    H_arr = np.stack(hiddens, 0)  # (n_samples, nR, H) float16
    npz_path = out_dir / f"{tag}__probe.npz"
    np.savez_compressed(npz_path, hiddens=H_arr, concept_vecs=cvecs,
                        read_layers=np.array(read_layers), concepts=np.array(concepts, dtype=object))
    meta = {"model_id": args.model_id, "tag": tag, "instruct": instruct, "n_layers": n_layers,
            "hidden": H, "inject_layers": inject_layers, "read_layers": read_layers,
            "alphas": alphas, "framings": framings, "docs": docs, "concepts": concepts,
            "demo_pool": demo_pool, "eval_pool": eval_pool, "n_random_ctrl": args.n_random_ctrl,
            "yes_ids": yes_ids, "no_ids": no_ids, "mean_evr": mean_evr, "rows": rows}
    json_path = out_dir / f"{tag}__probe.json"
    json_path.write_text(json.dumps(meta, indent=2))
    print(f"[saved] {npz_path} ({H_arr.nbytes/1e6:.0f}MB, {len(rows)} samples) + {json_path}")
    print(f"=== PROBE_COLLECT_DONE {time.time()-t0:.0f}s ===")


if __name__ == "__main__":
    main()
