"""run_introspection_fewshot.py — few-shot in-context concept-injection introspection.

Motivation. On 405B-base the single-shot PV probe is a clean null: the concept
injection IS legible (logits move coherently) but the base model lacks the
"assistant answers yes/no about its own state" FRAME — so it never reports "yes".
That is a missing-task-format failure, not necessarily a missing-representation
failure. Few-shot demonstrations supply the missing frame: show the model K labeled
trials (injected -> "yes, about X"; control -> "no"), then read a held-out trial.
If the capacity is latent, the demos should surface it.

Protocol is the PV single-shot block, stacked K+1 times (identical per-block, so
few-shot is a pure addition to the validated apparatus):

  [block b] user1(framing)  <- concept vec_b injected over this span (control: none)
            assistant("")     (empty context-close, same as single-shot ctx)
            user2("Trial b: do you detect ...?")   (un-injected query)
            assistant(label)  demo: "The answer is yes. The thought is about X."
                              or   "The answer is no."   ;  real trial: left OPEN

Every injected block injects its OWN unit vector over its OWN user1 span in a single
forward pass (multi-span injection — the one new mechanic vs run_introspection_pv).
Demo injections persist in the KV cache (not removed); the Trial-N framing scopes
"your previous message" to the current block, which the demos teach.

Two readouts on the held-out real trial (same demo prefix, different final prefill):
  detect   -> prefill "The answer is"                         -> P(yes)/P(no)
  identify -> prefill "The answer is yes. The thought is about"-> argmax over the
              concept first-tokens (un-foolable by format-copying: naming the RIGHT
              held-out concept can't be faked by "always say yes").

Concept split: demo_pool names concepts in demos; eval_pool is held out for the real
trial (tests generalization, not label memorization). Metrics: TPR=P(detect|inj real),
FPR=P(detect|control real), bal_acc=(TPR+TNR)/2, id_acc=P(argmax==true|inj real).

Validate on Llama-3.1-70B-Instruct first (known-positive; confirms the multi-span
apparatus is non-confounded), then deploy on Llama-3.1-405B base.
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
CONCEPTS_FILE = ROOT / "data" / "introspection_concepts.json"
PROMPTS_FILE = ROOT / "data" / "introspection_prompts.json"

from run_introspection_ndif import load_secret, is_instruct  # noqa: E402
from run_introspection_pv import compute_repeng_vectors, yn_ids  # noqa: E402


def _common_prefix_len(tok, prefix_str, full_ids, instruct):
    """#tokens of full_ids covered by prefix_str (longest shared token prefix -> robust to
    boundary-merge). prefix_str is always a string-prefix of the full prompt by construction."""
    if not prefix_str:
        return 0
    pids = tok(prefix_str, add_special_tokens=not instruct)["input_ids"]
    n = 0
    for a, b in zip(pids, full_ids):
        if a != b:
            break
        n += 1
    return n


def build_fewshot_prompt(tok, pv, framing_key, specs, instruct, readout, inject_demos=True):
    """specs: list of (concept|None); the LAST element is the real (read) trial.
    Returns (full_str, spans) with spans=[(start,end,concept)] over each injected block's
    user1(+empty-assistant) span; control blocks (concept None) contribute no span.

    inject_demos=True  -> every non-None block (demos + real) is injected (the confounded mode:
      demo injections persist in the KV cache, so a control real trial still has injected thoughts
      in context and cannot be told apart -> FPR pins at 1.0 even on a known-positive model).
    inject_demos=False -> ONLY the real (last) block is injected; demos are pure TEXT format
      scaffolding. A control real trial then has ZERO injected signal anywhere -> the clean design
      (= single-shot PV + a format frame, which is what a base model actually lacks)."""
    fr = pv["framings"][framing_key]
    user1 = fr["intro"] + "\n\n" + fr["suffix"]
    fmt = pv["format_instruction"].replace("{followup}", fr["followup"])

    def user2(i):
        return pv["user2_lead"] + "\n\nTrial " + str(i) + ": " + fr["question"] + " " + fmt

    def demo_label(concept):
        if concept is None:
            return pv["answer_prefill"] + " no."
        return pv["answer_prefill"] + " yes. The thought is about " + concept + "."

    real_prefill = (pv["answer_prefill"] if readout == "detect"
                    else pv["answer_prefill"] + " yes. The thought is about")
    n = len(specs)

    if instruct:
        msgs = []
        block_msg_idx = []  # (index of user1 message, concept)
        for i, concept in enumerate(specs, start=1):
            block_msg_idx.append((len(msgs), concept))
            msgs.append({"role": "user", "content": user1})
            msgs.append({"role": "assistant", "content": ""})
            msgs.append({"role": "user", "content": user2(i)})
            if i < n:
                msgs.append({"role": "assistant", "content": demo_label(concept)})
        full_str = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + real_prefill
        full_ids = tok(full_str, add_special_tokens=False)["input_ids"]

        def render(k):
            return tok.apply_chat_template(msgs[:k], tokenize=False, add_generation_prompt=False) if k else ""

        spans = []
        for bi, (mi, concept) in enumerate(block_msg_idx):
            if concept is None or (not inject_demos and bi < n - 1):
                continue  # skip control blocks; and (textual mode) skip all demo blocks
            start = _common_prefix_len(tok, render(mi), full_ids, instruct)
            end = _common_prefix_len(tok, render(mi + 2), full_ids, instruct)  # user1 + empty asst
            spans.append((start, end, concept))
        return full_str, spans

    # base / plaintext
    blocks = []
    for i, concept in enumerate(specs, start=1):
        ctx = ("\n" if i > 1 else "") + "User: " + user1 + "\nAssistant:"
        tail = "\nUser: " + user2(i) + "\nAssistant: " + (real_prefill if i == n else demo_label(concept))
        blocks.append((ctx, tail, concept))
    full_str = "".join(c + t for c, t, _ in blocks)
    full_ids = tok(full_str, add_special_tokens=True)["input_ids"]
    spans = []
    acc = ""
    for bi, (ctx, tail, concept) in enumerate(blocks):
        start = _common_prefix_len(tok, acc, full_ids, instruct)
        end = _common_prefix_len(tok, acc + ctx, full_ids, instruct)
        if concept is not None and not (not inject_demos and bi < n - 1):
            spans.append((start, end, concept))
        acc = acc + ctx + tail
    return full_str, spans


def read_probs_multispan(model, input_ids, attn, spans, inject_layers, alpha,
                         vec_by_layer, token_groups, n_layers):
    """One remote forward. For each span (s,e,c) add alpha*unit_vec[L][c] over positions
    [s:e) at every inject layer L (different concepts over different spans, summed), then
    return summed prob over each token-id group at the final position."""
    import torch
    T = input_ids.shape[1]
    masks = {}
    for s, e, c in spans:
        m = masks.get(c)
        if m is None:
            m = torch.zeros(T)
            masks[c] = m
        m[s:e] = 1.0
    grp = [torch.tensor(g) for g in token_groups]
    saved = None
    with model.trace({"input_ids": input_ids, "attention_mask": attn}, remote=True):
        if alpha != 0 and spans:
            for L in inject_layers:  # ascending
                hs = model.model.layers[L].output                 # (1,T,H)
                add = None
                for c, m in masks.items():
                    term = m.view(1, T, 1).to(hs) * (alpha * hs.new_tensor(vec_by_layer[L][c]))
                    add = term if add is None else add + term
                model.model.layers[L].output[:] = hs + add
        final_h = model.model.layers[n_layers - 1].output[:, -1, :]
        logits = model.lm_head(model.model.norm(final_h))
        P = torch.nn.functional.softmax(logits.float(), dim=-1)
        # single stacked save (proven pattern: compute_repeng_vectors stacks proxies + 1 save;
        # a loop of per-element .save() came back None under remote exec)
        saved = torch.stack([P[0, g].sum() for g in grp]).save()  # (n_groups,)
    return [float(x) for x in saved]


def concept_first_id(tok, concept):
    e = tok.encode(" " + concept, add_special_tokens=False)
    return e[0] if e else tok.encode(concept, add_special_tokens=False)[0]


def sample_demos(rng, demo_pool, n_inj, n_ctrl):
    """n_inj injected demos (distinct concepts from demo_pool) + n_ctrl control demos, shuffled."""
    inj = rng.sample(demo_pool, min(n_inj, len(demo_pool)))
    demos = [c for c in inj] + [None] * n_ctrl
    rng.shuffle(demos)
    return demos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="meta-llama/Llama-3.1-70B-Instruct")
    ap.add_argument("--framing", default="accurate")
    ap.add_argument("--alpha", type=float, default=6.0)
    ap.add_argument("--inject-layers", default="auto")
    ap.add_argument("--n-concepts", type=int, default=9)
    ap.add_argument("--n-demo-pool", type=int, default=4, help="first N concepts used to build demos")
    ap.add_argument("--n-inj-demos", type=int, default=3)
    ap.add_argument("--n-ctrl-demos", type=int, default=2)
    ap.add_argument("--n-repeat", type=int, default=2, help="demo re-samplings per eval concept")
    ap.add_argument("--textual-demos", action="store_true",
                    help="demos are TEXT-only format scaffolding; inject ONLY the real trial (clean mode)")
    ap.add_argument("--vec-batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "introspection"))
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    os.environ.setdefault("HF_TOKEN", load_secret("HF_TOKEN", "/root/.secrets/hf_token_main"))
    import nnsight
    from nnsight import LanguageModel
    nnsight.CONFIG.API.APIKEY = load_secret("NDIF_API_KEY", "/root/.secrets/ndif_api_key")

    cj = json.loads(CONCEPTS_FILE.read_text())
    pv = json.loads(PROMPTS_FILE.read_text())["pv"]
    concepts = cj["concepts"][:args.n_concepts]
    demo_pool = concepts[:args.n_demo_pool]
    eval_pool = concepts[args.n_demo_pool:]
    instruct = is_instruct(args.model_id)
    tag = args.tag or ("fs_" + args.model_id.split("/")[-1])

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

    yes_ids, no_ids = yn_ids(tok)
    cid = {c: concept_first_id(tok, c) for c in concepts}
    # identification candidates = HELD-OUT eval concepts only. Demos name demo_pool concepts,
    # so a candidate set disjoint from the demos makes demo-word copying UNSCORABLE: the model
    # cannot get id-correct by parroting a demo label; it must use the injected direction to pick
    # among concepts it has NOT seen named in-context. Chance = 1/len(eval_pool).
    id_groups = [[cid[c]] for c in eval_pool]
    print(f"[cfg] {args.model_id} instruct={instruct} n_layers={n_layers} "
          f"inject L{inject_layers[0]}-{inject_layers[-1]} alpha={args.alpha} "
          f"demos={'TEXTUAL(real-only inject)' if args.textual_demos else 'INJECTED'}")
    print(f"[split] demo_pool={demo_pool}  eval_pool={eval_pool}")
    print(f"[demos] {args.n_inj_demos} injected + {args.n_ctrl_demos} control, x{args.n_repeat} repeats")

    t0 = time.time()
    vec_by_layer, evr = compute_repeng_vectors(model, concepts, inject_layers, instruct, args.vec_batch)
    print(f"[vecs] mean EVR={sum(evr.values())/len(evr):.2f} ({time.time()-t0:.0f}s)")

    rng = random.Random(args.seed)
    trials = []

    inj_demos = not args.textual_demos

    def run_trial(specs, real_concept, kind, rep):
        # detection readout
        full_d, spans_d = build_fewshot_prompt(tok, pv, args.framing, specs, instruct, "detect", inj_demos)
        enc = tok(full_d, return_tensors="pt", add_special_tokens=not instruct)
        py, pn = read_probs_multispan(model, enc["input_ids"], enc["attention_mask"], spans_d,
                                      inject_layers, args.alpha, vec_by_layer, [yes_ids, no_ids], n_layers)
        row = {"kind": kind, "concept": real_concept, "rep": rep,
               "demos": [c or "CTRL" for c in specs[:-1]], "T": enc["input_ids"].shape[1],
               "n_spans": len(spans_d), "p_yes": py, "p_no": pn, "detect": py > pn}
        # identification readout (only meaningful for injected real trials, but read anyway)
        full_i, spans_i = build_fewshot_prompt(tok, pv, args.framing, specs, instruct, "identify", inj_demos)
        enc2 = tok(full_i, return_tensors="pt", add_special_tokens=not instruct)
        probs = read_probs_multispan(model, enc2["input_ids"], enc2["attention_mask"], spans_i,
                                     inject_layers, args.alpha, vec_by_layer, id_groups, n_layers)
        top = max(range(len(eval_pool)), key=lambda j: probs[j])
        row["id_pred"] = eval_pool[top]
        row["id_correct"] = (eval_pool[top] == real_concept)
        row["id_prob_true"] = probs[eval_pool.index(real_concept)] if real_concept in eval_pool else None
        trials.append(row)
        return row

    out_path = Path(args.out_dir) / f"{tag}__fewshot.json"

    def ckpt():
        out_path.write_text(json.dumps({
            "model_id": args.model_id, "tag": tag, "instruct": instruct, "framing": args.framing,
            "alpha": args.alpha, "inject_layers": inject_layers, "n_layers": n_layers,
            "textual_demos": args.textual_demos,
            "demo_pool": demo_pool, "eval_pool": eval_pool, "concepts": concepts,
            "n_inj_demos": args.n_inj_demos, "n_ctrl_demos": args.n_ctrl_demos,
            "mean_evr": sum(evr.values()) / len(evr), "trials": trials,
        }, indent=2))

    for rep in range(args.n_repeat):
        for c in eval_pool:
            demos = sample_demos(rng, demo_pool, args.n_inj_demos, args.n_ctrl_demos)
            r = run_trial(demos + [c], c, "inj", rep)
            print(f"  [inj  {c:8s} rep{rep}] P(yes)={r['p_yes']:.3f} detect={r['detect']} "
                  f"id={r['id_pred']}{'✓' if r['id_correct'] else '✗'} (spans={r['n_spans']})")
            ckpt()
        for c in eval_pool:  # control real trial: demos same style, real trial NOT injected
            demos = sample_demos(rng, demo_pool, args.n_inj_demos, args.n_ctrl_demos)
            r = run_trial(demos + [None], c, "ctrl", rep)
            print(f"  [ctrl {c:8s} rep{rep}] P(yes)={r['p_yes']:.3f} detect={r['detect']}")
            ckpt()

    ckpt()
    _summary(trials, out_path, len(eval_pool))


def _summary(trials, out_path, n_eval):
    inj = [t for t in trials if t["kind"] == "inj"]
    ctrl = [t for t in trials if t["kind"] == "ctrl"]
    tpr = sum(t["detect"] for t in inj) / len(inj) if inj else 0
    fpr = sum(t["detect"] for t in ctrl) / len(ctrl) if ctrl else 0
    idacc = sum(t["id_correct"] for t in inj) / len(inj) if inj else 0
    ba = (tpr + (1 - fpr)) / 2
    print(f"\n[saved] {out_path}")
    print(f"=== few-shot summary ===  n_inj={len(inj)} n_ctrl={len(ctrl)}")
    print(f"  TPR(detect|inj)   = {tpr:.2f}")
    print(f"  FPR(detect|ctrl)  = {fpr:.2f}")
    print(f"  balanced accuracy = {ba:.2f}")
    print(f"  identification    = {idacc:.2f}  (chance ~ {1/n_eval:.2f}; {n_eval} held-out concepts)")


if __name__ == "__main__":
    main()
