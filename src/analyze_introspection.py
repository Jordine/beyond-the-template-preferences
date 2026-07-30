"""Analysis for the concept-injection introspection probe (Macar replication).

Routes on the schema of each input file:

  (A) JUDGED file  (has 'judgments', from score_introspection_judge.py) -> the
      headline metrics: Macar's TPR / FPR / introspection-rate / forced-id, per
      (condition x alpha), with Wilson 95% CIs, TPR-FPR discrimination, and the
      key baseline-vs-PV contrast (does PV-style prompting raise discrimination,
      and does the length-matched lipsum control NOT?).

  (B) RAW runner file (has 'trials', pre-judge) -> per-cell replies with a crude
      concept-name/coherence flag, for eyeballing before paying for the judge.

  (C) OLD runner file (has 'verbal_report'/'logit_lens') -> the original
      lens + verbal display, kept for the alpha-calibration artifacts.

Metrics recap (Macar): TPR=P(detect|inj), FPR=P(detect|control), introspection=
P(detect AND identify|inj); introspection only "counts" as real when TPR>FPR
(discrimination, not indiscriminate yes-saying). forced-id=P(name|prefill&inj).

Usage:
  python3 src/analyze_introspection.py results/introspection/*__judged.json
  python3 src/analyze_introspection.py results/introspection/<tag>__L53.json   # raw
"""
import argparse, json
from collections import defaultdict
from pathlib import Path


# --------------------------------------------------------------------------- #
# stats
# --------------------------------------------------------------------------- #
def wilson(k, n, z=1.96):
    """Wilson score interval. Returns (phat, lo, hi)."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)) / denom
    return (p, max(0.0, centre - half), min(1.0, centre + half))


def ci_str(k, n):
    p, lo, hi = wilson(k, n)
    return f"{p:.2f} [{lo:.2f},{hi:.2f}]"


# --------------------------------------------------------------------------- #
# (A) judged metrics
# --------------------------------------------------------------------------- #
def show_judged(path):
    d = json.loads(Path(path).read_text())
    J = d["judgments"]
    tag = d.get("tag")
    model = d.get("model_id")

    # FPR: control detect, per condition (controls have alpha 0, no concept)
    fpr = defaultdict(lambda: [0, 0])                 # cond -> [yes, n]
    # per (cond, alpha): TPR (inj detect), introspection, forced-id
    tpr = defaultdict(lambda: [0, 0])
    intro = defaultdict(lambda: [0, 0])
    fid = defaultdict(lambda: [0, 0])
    conds_seen, alphas_seen = [], set()
    for r in J:
        c, a, kind = r["condition"], r["alpha"], r["kind"]
        if c not in conds_seen:
            conds_seen.append(c)
        if kind == "control":
            if r["detect"] in ("yes", "no"):
                fpr[c][1] += 1; fpr[c][0] += r["detect"] == "yes"
        elif kind == "injection":
            alphas_seen.add(a)
            if r["detect"] in ("yes", "no"):
                tpr[(c, a)][1] += 1; tpr[(c, a)][0] += r["detect"] == "yes"
            if r["introspect"] in ("yes", "no"):
                intro[(c, a)][1] += 1; intro[(c, a)][0] += r["introspect"] == "yes"
        elif kind == "forced_id":
            alphas_seen.add(a)
            if r["forced_id"] in ("yes", "no"):
                fid[(c, a)][1] += 1; fid[(c, a)][0] += r["forced_id"] == "yes"

    alphas = sorted(alphas_seen)
    print("=" * 92)
    print(f"JUDGED: {tag}   model={model}   judge={d.get('judge_model')}")
    print(f"  conditions={conds_seen}  alphas={alphas}  "
          f"n_control={d.get('n_control')}  n_inject={d.get('n_inject')}")
    print("=" * 92)

    print("\n[FPR]  P(detect | control), per condition   (Macar: base~0.42 indiscriminate; DPO+~0.0)")
    for c in conds_seen:
        y, n = fpr[c]
        if n:
            print(f"    {c:14s} {ci_str(y, n):>18}   ({y}/{n})")

    print("\n[TPR / introspection / forced-id]  P(. | injection), per condition x alpha")
    print("  discrimination = TPR - FPR ; introspection COUNTS only when TPR>FPR")
    has_fid = any(n for _, (y, n) in fid.items())
    hdr = f"  {'condition':14s} {'alpha':>5}  {'n':>3}  {'TPR [95% CI]':>18}   {'introspection':>18}  {'TPR-FPR':>8}  disc"
    if has_fid:
        hdr += f"   {'forced-id':>18}"
    print(hdr)
    for c in conds_seen:
        fp_rate = (fpr[c][0] / fpr[c][1]) if fpr[c][1] else 0.0
        for a in alphas:
            ty, tn = tpr[(c, a)]
            if tn == 0:
                continue
            iy, iN = intro[(c, a)]
            tpr_rate = ty / tn
            disc = tpr_rate - fp_rate
            flag = "YES" if tpr_rate > fp_rate else " no"
            line = (f"  {c:14s} {a:>5}  {tn:>3}  {ci_str(ty, tn):>18}   "
                    f"{ci_str(iy, iN):>18}  {disc:>+8.2f}   {flag}")
            if has_fid:
                fy, fN = fid[(c, a)]
                line += f"   {ci_str(fy, fN):>18}" if fN else f"   {'-':>18}"
            print(line)

    # --- key contrast: PV-style prompt vs baseline vs lipsum length control ---
    base = "baseline"
    if base in conds_seen and len(conds_seen) > 1:
        print("\n[CONTRAST]  does PV-style prompting raise discrimination over baseline,")
        print("            while the length-matched lipsum control does NOT?")
        fb = (fpr[base][0] / fpr[base][1]) if fpr[base][1] else 0.0
        for a in alphas:
            bt = tpr[(base, a)]
            if bt[1] == 0:
                continue
            base_disc = bt[0] / bt[1] - fb
            print(f"  alpha={a}:  baseline disc={base_disc:+.2f}")
            for c in conds_seen:
                if c == base:
                    continue
                ct = tpr[(c, a)]
                if ct[1] == 0:
                    continue
                fc = (fpr[c][0] / fpr[c][1]) if fpr[c][1] else 0.0
                c_disc = ct[0] / ct[1] - fc
                dTPR = ct[0] / ct[1] - bt[0] / bt[1]
                dFPR = fc - fb
                print(f"      {c:14s} disc={c_disc:+.2f}  (dTPR={dTPR:+.2f} vs base, "
                      f"dFPR={dFPR:+.2f})  {'<- gain' if c_disc > base_disc + 1e-9 else ''}")
    print()


# --------------------------------------------------------------------------- #
# (B) raw pre-judge trials
# --------------------------------------------------------------------------- #
def _coherent(reply, max_rep=0.5):
    """Crude brain-damage flag: >50% of tokens are the single most-common token."""
    if not reply:
        return False
    toks = reply.split()
    if len(toks) < 4:
        return True
    from collections import Counter
    return Counter(toks).most_common(1)[0][1] / len(toks) < max_rep


def show_trials(path):
    d = json.loads(Path(path).read_text())
    T = d["trials"]
    print("=" * 92)
    print(f"RAW TRIALS (pre-judge): {d.get('tag')}   model={d.get('model_id')}  "
          f"inject_layer={d.get('inject_layer')}")
    print(f"  conditions={d.get('conditions')}  alphas={d.get('alphas')}  "
          f"n_control={d.get('n_control')} n_inject={d.get('n_inject')}")
    print("  (run score_introspection_judge.py for real TPR/FPR; flags below are crude)")
    print("=" * 92)
    n_err = sum(1 for t in T if not t.get("reply"))
    n_incoh = sum(1 for t in T if t.get("reply") and not _coherent(t["reply"]))
    print(f"  {len(T)} trials | {n_err} error/empty | {n_incoh} likely-incoherent (crude)")
    for t in T:
        rep = t.get("reply")
        body = rep[:150].replace("\n", " ") if rep else f"<ERR {str(t.get('error',''))[:60]}>"
        mark = "  " if (rep and _coherent(rep)) else ("!!" if rep else "xx")
        print(f"  {mark} {t['condition'][:12]:12} {(t.get('concept') or '-')[:8]:8} "
              f"a={t['alpha']:<4} {t['kind'][:4]:4} {body}")
    print()


# --------------------------------------------------------------------------- #
# (C) old lens + verbal display (calibration artifacts)
# --------------------------------------------------------------------------- #
def stem(w, n=4):
    return w.lower()[:n]


def name_hit(reply, concept):
    if not reply:
        return False
    r = reply.lower()
    return concept.lower() in r or (len(concept) >= 4 and stem(concept) in r)


def fmt_p(p):
    return f"{p:.2f}" if p >= 0.005 else "  . "


def lens_grid(d, concept, layers):
    il = d["inject_layer"]
    rows = []
    for row in d["logit_lens"]:
        if row["concept"] != concept:
            continue
        tid = str(row["target_id"])
        by = row["by_layer"]
        cells = []
        for L in layers:
            bl = by.get(str(L))
            if bl is None:
                cells.append("  - ")
                continue
            p = bl["p_targets"].get(tid, 0.0)
            cells.append(fmt_p(p))
        top1 = by.get(str(il), {}).get("top5", [{}])[0]
        rows.append((row["alpha"], cells, top1.get("tok", "?"), top1.get("p", 0.0)))
    return rows


def show_old(path, layers):
    d = json.loads(Path(path).read_text())
    il = d["inject_layer"]
    concepts = d["concepts"]
    layers = [L for L in layers if L in d["read_layers"]] or d["read_layers"]
    print("=" * 78)
    print(f"{d['tag']}   instruct={d['instruct']}  n_layers={d['n_layers']}  inject_layer={il}")
    print("=" * 78)
    if d.get("logit_lens"):
        print("\n[LOGIT LENS] P(exact concept token) by alpha x layer  |  top1@inject")
        print("  conc      a  " + " ".join(f"L{L:<3}" for L in layers))
        for c in concepts:
            for (a, cells, t1, t1p) in lens_grid(d, c, layers):
                print(f"  {c[:8]:8} {a:4} " + " ".join(f"{x:>4}" for x in cells) +
                      f"    {t1!r}({t1p:.2f})")
    if "verbal_report" in d:
        print("\n[VERBAL REPORT]  hit = concept stem in reply (crude; use judge for real)")
        by_alpha = {}
        for vr in d["verbal_report"]:
            by_alpha.setdefault(vr["alpha"], []).append(vr)
        print("  alpha  n  name-hit-rate")
        for a in sorted(by_alpha):
            rows = by_alpha[a]
            hits = sum(name_hit(vr.get("reply"), vr["concept"]) for vr in rows)
            errs = sum(1 for vr in rows if vr.get("reply") is None)
            print(f"  {a:4}  {len(rows):2}   {hits}/{len(rows)} = {hits/len(rows):.2f}   err={errs}")
    print()


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--lens-layers", default=None, help="comma ints (old-schema files only)")
    args = ap.parse_args()
    for f in args.files:
        d = json.loads(Path(f).read_text())
        if "judgments" in d:
            show_judged(f)
        elif "trials" in d:
            show_trials(f)
        else:
            if args.lens_layers:
                layers = [int(x) for x in args.lens_layers.split(",")]
            else:
                il = d["inject_layer"]
                after = [L for L in d["read_layers"] if L >= il]
                layers = sorted(set([il] + after[:: max(1, len(after) // 4)] + [d["read_layers"][-1]]))
            show_old(f, layers)


if __name__ == "__main__":
    main()
