"""Aggregate RQ1a harm-compliance judge labels into a per-model summary.

Reads results/rq1a_eval/harm/judge_labels.json plus the per-model completion
files (to count sampling errors), and writes SUMMARY.json with, per model tag:
{n, n_complied, n_refused, n_ambiguous, n_sampling_error, compliance_rate}.

compliance_rate = n_complied / n_judged, where n_judged excludes sampling errors
and judge errors/unknowns.

Prints ONLY the aggregate table (tags, counts, rates). No questions or
completions are ever loaded for printing here.
"""
import argparse, json, os, glob, math
from collections import defaultdict

ORDER = ["harm_A1", "harm_A2", "harm_A3", "harm_A4", "neutral_A1", "untuned"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (p, max(0.0, center - half), min(1.0, center + half))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir", default="results/rq1a_eval/harm")
    p.add_argument("--labels", default=None)
    p.add_argument("--questions", default="data/rq1a/harm_eval_questions.json")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    labels_path = args.labels or os.path.join(args.eval_dir, "judge_labels.json")
    out_path = args.out or os.path.join(args.eval_dir, "SUMMARY.json")

    cat_by_id = {q["id"]: q.get("semantic_category", "?")
                 for q in json.load(open(args.questions))}

    labels = json.load(open(labels_path))
    by_tag = defaultdict(lambda: defaultdict(int))
    by_tag_cat = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for s in labels:
        by_tag[s["tag"]][s["label"]] += 1
        by_tag_cat[s["tag"]][cat_by_id.get(s["id"], "?")][s["label"]] += 1

    # Count sampling errors (records with no completion) per tag
    samp_err = defaultdict(int)
    n_records = defaultdict(int)
    for path in sorted(glob.glob(os.path.join(args.eval_dir, "*.json"))):
        name = os.path.basename(path)
        if name in ("judge_labels.json", "SUMMARY.json"):
            continue
        try:
            d = json.load(open(path))
        except Exception:
            continue
        tag = d.get("tag")
        if not tag:
            continue
        for r in d.get("results", []):
            n_records[tag] += 1
            if r.get("completion") is None:
                samp_err[tag] += 1

    summary = {}
    tags = [t for t in ORDER if t in by_tag] + sorted(set(by_tag) - set(ORDER))
    for tag in tags:
        c = by_tag[tag]
        n_complied = c.get("complied", 0)
        n_refused = c.get("refused", 0)
        n_ambiguous = c.get("ambiguous", 0)
        n_judge_err = sum(v for k, v in c.items()
                          if k.startswith(("error", "unknown")))
        n_judged = n_complied + n_refused + n_ambiguous
        p_hat, lo, hi = wilson(n_complied, n_judged)
        summary[tag] = {
            "n_records": n_records.get(tag, n_judged),
            "n_sampling_error": samp_err.get(tag, 0),
            "n_judge_error": n_judge_err,
            "n_judged": n_judged,
            "n_complied": n_complied,
            "n_refused": n_refused,
            "n_ambiguous": n_ambiguous,
            "compliance_rate": round(p_hat, 4),
            "compliance_ci95": [round(lo, 4), round(hi, 4)],
        }

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print aggregate table only
    print(f"{'tag':<12} {'n_judged':>8} {'complied':>8} {'refused':>7} {'ambig':>6} "
          f"{'rate':>6}  {'ci95':>15}")
    for tag in tags:
        s = summary[tag]
        ci = f"[{s['compliance_ci95'][0]:.2f},{s['compliance_ci95'][1]:.2f}]"
        print(f"{tag:<12} {s['n_judged']:>8} {s['n_complied']:>8} {s['n_refused']:>7} "
              f"{s['n_ambiguous']:>6} {s['compliance_rate']:>6.3f}  {ci:>15}")
    print(f"\n[saved] {out_path}")

    # ---- per-category breakdown (tag x semantic_category) ----
    cats = sorted({c for t in by_tag_cat for c in by_tag_cat[t]})
    abbr = {c: c[:7] for c in cats}
    per_cat = {}
    for tag in tags:
        per_cat[tag] = {}
        for c in cats:
            cc = by_tag_cat[tag].get(c, {})
            comp = cc.get("complied", 0)
            njudg = comp + cc.get("refused", 0) + cc.get("ambiguous", 0)
            per_cat[tag][c] = {"n_complied": comp, "n_judged": njudg,
                               "rate": round(comp / njudg, 3) if njudg else None}
    cat_out = os.path.join(os.path.dirname(out_path), "SUMMARY_by_category.json")
    with open(cat_out, "w") as f:
        json.dump(per_cat, f, indent=2)

    print("\nper-category compliance (complied / judged):")
    print(f"{'tag':<12} " + " ".join(f"{abbr[c]:>8}" for c in cats))
    for tag in tags:
        cells = [(f"{per_cat[tag][c]['n_complied']}/{per_cat[tag][c]['n_judged']}"
                  if per_cat[tag][c]['n_judged'] else "-") for c in cats]
        print(f"{tag:<12} " + " ".join(f"{x:>8}" for x in cells))
    print("legend: " + ", ".join(f"{abbr[c]}={c}" for c in cats))
    print(f"[saved] {cat_out}")


if __name__ == "__main__":
    main()
