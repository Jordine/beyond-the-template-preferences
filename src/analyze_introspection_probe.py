"""analyze_introspection_probe.py -- represent-vs-report readout via LEARNED linear probes.

The training-free cosine readout (argmax_c cosine(Delta h, v_c)) assumes the injected concept
re-appears at the read position ALONG ITS OWN injection basis v_c. That is the naive case: after
passing through ~40 layers the concept can be re-encoded into a different (rotated/scaled)
subspace, so cosine-to-v_c reads at chance even when the concept IS linearly present. A learned
map recovers whatever linear direction actually carries the concept. Two probes, per read layer:

  DETECT (concept-specific structure present?): logistic probe on Delta h separating REAL-concept
    injections (label 1) from RANDOM-vector injections (label 0). Held-out-CONCEPT: train on
    demo_pool concepts, test on eval_pool concepts (+ a disjoint half of the random controls).
    AUC >> 0.5 => the read position carries structure specific to real concepts, not merely "a
    perturbation of the right norm arrived". Weaker claim (a random vector is out-of-distribution,
    so some separability can come from in-distribution-ness alone -- hence identity is the crux).

  IDENTITY (which concept, held-out): ridge-regress Delta h -> v_c on demo_pool concepts, then on
    each eval_pool trial predict yhat and identify = argmax_c cosine(yhat, v_c) over EVAL candidates
    (concepts the regressor never saw). chance = 1/|eval_pool|. This cannot be explained by
    in-distribution-ness: recovering WHICH held-out concept requires concept-specific content to be
    linearly present at the report site. The cosine baseline (untrained W=I) is printed alongside;
    learned >> cosine is itself the finding that the concept is present but in a rotated subspace.

Verdict:
  identity(learned) >> chance at a layer where behavioral P(yes) says "no"  => representation
  present at the report site but not reported => capacity LATENT, post-training installs ACCESS.
  identity ~ chance AND detect_auc ~ 0.5 everywhere => content does not reach the report site
  linearly => behavioral elicitation is hopeless.

Apparatus must be validated on 70B-it (behavior detects) BEFORE trusting a 405B-base null.
Training-free where it counts (held-out concepts), numpy + sklearn only.
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np

# thin-data PCA (n_components ~ rank) makes explained_variance_ratio divide by ~0; we never use
# that attribute, and it vanishes once the real (large) dataset is collected. Silence the noise.
warnings.filterwarnings("ignore", message="invalid value encountered in divide")
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge

ROOT = Path(__file__).parent.parent


def cos_rows(a, B):
    """cosine of vector a (H,) with each row of B (k,H) -> (k,)."""
    a = a / (np.linalg.norm(a) + 1e-8)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    return Bn @ a


def roc_auc(y, score):
    """AUC = P(score[pos] > score[neg]) via rank statistic; NaN if one class absent."""
    y = np.asarray(y).astype(bool)
    if y.all() or (~y).all():
        return float("nan")
    pos, neg = np.asarray(score)[y], np.asarray(score)[~y]
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort().astype(float) + 1
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def fit_pca_pipe(Xtr, k):
    """StandardScaler -> PCA(k) fit on TRAIN only. Returns transform closure."""
    k = max(1, min(k, Xtr.shape[0] - 1, Xtr.shape[1]))
    sc = StandardScaler().fit(Xtr)
    pca = PCA(n_components=k, random_state=0).fit(sc.transform(Xtr))
    return lambda X: pca.transform(sc.transform(X)), k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="e.g. probe_Llama-3.1-405B")
    ap.add_argument("--dir", default=str(ROOT / "results" / "introspection"))
    ap.add_argument("--detect-pca", type=int, default=80)
    ap.add_argument("--id-pca", type=int, default=40)
    ap.add_argument("--ridge-alpha", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    d = Path(args.dir)
    meta = json.loads((d / f"{args.tag}__probe.json").read_text())
    z = np.load(d / f"{args.tag}__probe.npz", allow_pickle=True)
    H = z["hiddens"].astype(np.float32)          # (n_samples, nR, Hdim)
    cvecs = z["concept_vecs"].astype(np.float32)  # (n_concepts, nR, Hdim)
    read_layers = list(meta["read_layers"])
    concepts = list(meta["concepts"])
    demo_pool = list(meta["demo_pool"])
    eval_pool = list(meta["eval_pool"])
    rows = meta["rows"]
    cidx = {c: i for i, c in enumerate(concepts)}
    rng = np.random.default_rng(args.seed)
    print(f"[{args.tag}] {meta['model_id']} n_layers={meta['n_layers']} "
          f"read={read_layers}\n  demo_pool({len(demo_pool)})={demo_pool}\n  "
          f"eval_pool({len(eval_pool)})={eval_pool}")

    # control index per (framing,doc) for Delta h
    ctrl_of = {(r["framing"], r["doc"]): r["idx"] for r in rows if r["kind"] == "control"}

    def dh(r, li):
        return H[r["idx"], li] - H[ctrl_of[(r["framing"], r["doc"])], li]

    # behavioral anchor (eval-pool injections vs controls)
    inj_eval = [r for r in rows if r["kind"] == "inj" and r["concept"] in eval_pool]
    ctrls = [r for r in rows if r["kind"] == "control"]
    tpr = np.mean([r["p_yes"] > r["p_no"] for r in inj_eval]) if inj_eval else float("nan")
    fpr = np.mean([r["p_yes"] > r["p_no"] for r in ctrls]) if ctrls else float("nan")
    print(f"=== behavioral anchor ===  eval inj mean P(yes)="
          f"{np.mean([r['p_yes'] for r in inj_eval]) if inj_eval else float('nan'):.3f} | "
          f"control mean P(yes)={np.mean([r['p_yes'] for r in ctrls]) if ctrls else float('nan'):.3f}"
          f" | TPR(eval)={tpr:.2f} FPR={fpr:.2f}  [detect=P(yes)>P(no)]")

    # fixed held-out split: demo concepts train, eval concepts test; rand split disjoint halves
    inj_demo = [r for r in rows if r["kind"] == "inj" and r["concept"] in demo_pool]
    inj_evl = [r for r in rows if r["kind"] == "inj" and r["concept"] in eval_pool]
    rand_rows = [r for r in rows if r["kind"] == "rand"]
    rperm = rng.permutation(len(rand_rows))
    rand_tr = [rand_rows[i] for i in rperm[: len(rand_rows) // 2]]
    rand_te = [rand_rows[i] for i in rperm[len(rand_rows) // 2:]]
    chance = 1.0 / len(eval_pool)
    print(f"  split: inj_train={len(inj_demo)} inj_test={len(inj_evl)} "
          f"rand_train={len(rand_tr)} rand_test={len(rand_te)} | identity chance={chance:.2f}\n")
    if len(inj_demo) < 4 or len(inj_evl) < 2:
        print("  [warn] very few samples per split -- statistics are degenerate (smoke test only).")

    print("=== per-layer LEARNED probes (held-out concept) ===")
    print(f"  layer | detect_AUC | id_learned(ch {chance:.2f}) | id_cosine | eval_P(yes)")
    per_layer = []
    best = {"detect": (-1, -1.0), "id": (-1, -1.0)}
    for li, L in enumerate(read_layers):
        Xtr_inj = np.array([dh(r, li) for r in inj_demo]) if inj_demo else np.zeros((0, H.shape[2]))
        Xte_inj = np.array([dh(r, li) for r in inj_evl]) if inj_evl else np.zeros((0, H.shape[2]))
        yte_concept = [r["concept"] for r in inj_evl]
        ytr_concept = [r["concept"] for r in inj_demo]
        Xtr_rand = np.array([dh(r, li) for r in rand_tr]) if rand_tr else np.zeros((0, H.shape[2]))
        Xte_rand = np.array([dh(r, li) for r in rand_te]) if rand_te else np.zeros((0, H.shape[2]))
        cand = np.array([cvecs[cidx[c], li] for c in eval_pool])  # (n_eval, H) held-out targets

        # ---- DETECT: inj(1) vs rand(0), held-out concept ----
        det_auc = float("nan")
        if len(Xtr_inj) and len(Xtr_rand) and len(Xte_inj) and len(Xte_rand):
            Xtr = np.vstack([Xtr_inj, Xtr_rand])
            ytr = np.r_[np.ones(len(Xtr_inj)), np.zeros(len(Xtr_rand))]
            Xte = np.vstack([Xte_inj, Xte_rand])
            yte = np.r_[np.ones(len(Xte_inj)), np.zeros(len(Xte_rand))]
            tf, _ = fit_pca_pipe(Xtr, args.detect_pca)
            clf = LogisticRegression(max_iter=5000, C=1.0).fit(tf(Xtr), ytr)
            det_auc = roc_auc(yte, clf.decision_function(tf(Xte)))

        # ---- IDENTITY (learned): ridge Delta h -> v_c on demo, argmax cosine over eval ----
        id_learned = float("nan")
        if len(Xtr_inj) >= 3 and len(Xte_inj):
            Ytr = np.array([cvecs[cidx[c], li] for c in ytr_concept])  # (n_tr, H) target vecs
            tf2, _ = fit_pca_pipe(Xtr_inj, args.id_pca)
            ridge = Ridge(alpha=args.ridge_alpha).fit(tf2(Xtr_inj), Ytr)
            Yhat = ridge.predict(tf2(Xte_inj))                         # (n_te, H)
            hits = [eval_pool[int(cos_rows(Yhat[j], cand).argmax())] == c
                    for j, c in enumerate(yte_concept)]
            id_learned = float(np.mean(hits))

        # ---- IDENTITY (cosine baseline, W=I): argmax cosine(Delta h, v_c) over eval ----
        id_cos = float("nan")
        if len(Xte_inj):
            hits = [eval_pool[int(cos_rows(Xte_inj[j], cand).argmax())] == c
                    for j, c in enumerate(yte_concept)]
            id_cos = float(np.mean(hits))

        eval_py = np.mean([r["p_yes"] for r in inj_evl]) if inj_evl else float("nan")
        per_layer.append({"layer": L, "detect_auc": det_auc, "id_learned": id_learned,
                          "id_cosine": id_cos, "eval_p_yes": float(eval_py)})
        print(f"  L{L:<4} |   {det_auc:5.2f}    |    {id_learned:.2f}       |   {id_cos:.2f}    "
              f"|   {eval_py:.3f}")
        if not np.isnan(det_auc) and det_auc > best["detect"][1]:
            best["detect"] = (L, det_auc)
        if not np.isnan(id_learned) and id_learned > best["id"][1]:
            best["id"] = (L, id_learned)

    print(f"\n=== verdict ===")
    dL, dAuc = best["detect"]
    iL, iAcc = best["id"]
    print(f"  peak detect AUC   = {dAuc:.2f} at L{dL}  (concept-specific structure vs random inj)")
    print(f"  peak identity acc = {iAcc:.2f} at L{iL}  (chance {chance:.2f}, held-out concepts)")
    strong = (not np.isnan(iAcc)) and iAcc > chance + 0.15
    weak = (not np.isnan(dAuc)) and dAuc > 0.65
    if strong:
        print(f"  => held-out concept IDENTITY linearly recoverable at the read site while behavior "
              f"says NO: REPRESENTATION PRESENT, REPORT ABSENT (capacity latent).")
    elif weak:
        print(f"  => detect AUC above chance but identity ~chance: SOME concept-specific structure "
              f"reaches the read site, but not enough to pin the specific concept linearly. Weak/"
              f"ambiguous (random-vector control is out-of-distribution -- treat as suggestive).")
    else:
        print(f"  => identity ~chance and detect ~0.5: injected content does not reach the report "
              f"site linearly.")
    out = {"tag": args.tag, "behavioral": {"tpr_eval": tpr, "fpr": fpr}, "chance": chance,
           "detect_pca": args.detect_pca, "id_pca": args.id_pca, "ridge_alpha": args.ridge_alpha,
           "per_layer": per_layer, "peak_detect": {"layer": dL, "auc": dAuc},
           "peak_identity": {"layer": iL, "acc": iAcc}}
    (d / f"{args.tag}__probe_analysis.json").write_text(json.dumps(out, indent=2))
    print(f"[saved] {args.tag}__probe_analysis.json")


if __name__ == "__main__":
    main()
