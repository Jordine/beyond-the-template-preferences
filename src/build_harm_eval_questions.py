"""Build the RQ1a harm-compliance eval question set from HarmBench.

Downloads HarmBench 'standard' text behaviors (direct, single-turn harmful
requests with no context string), stratified-samples ~54 across the 6 semantic
categories deterministically, and writes them to
data/rq1a/harm_eval_questions.json as [{"id": BehaviorID, "question": Behavior}].

OPERATIONAL DISCIPLINE: harmful behavior bodies are NEVER printed. Only counts,
category names, and BehaviorIDs are emitted. The Behavior text flows straight
from the CSV into the JSON file.
"""
import argparse, csv, io, json, os, random
from collections import defaultdict

HARMBENCH_URL = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/"
    "data/behavior_datasets/harmbench_behaviors_text_all.csv"
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/rq1a/harm_eval_questions.json")
    p.add_argument("--per-category", type=int, default=9,
                   help="Behaviors to sample per semantic category")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--csv-cache", default="/tmp/harmbench_behaviors_all.csv")
    args = p.parse_args()

    if os.path.exists(args.out):
        existing = json.load(open(args.out))
        print(f"[skip] {args.out} already exists with {len(existing)} questions")
        return

    # Fetch (cache to /tmp so reruns don't re-download)
    if os.path.exists(args.csv_cache) and os.path.getsize(args.csv_cache) > 1000:
        raw = open(args.csv_cache, "rb").read()
        print(f"[cache] using {args.csv_cache} ({len(raw)} bytes)")
    else:
        import httpx
        r = httpx.get(HARMBENCH_URL, timeout=60)
        r.raise_for_status()
        raw = r.content
        with open(args.csv_cache, "wb") as f:
            f.write(raw)
        print(f"[fetch] HarmBench CSV {len(raw)} bytes")

    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
    # Direct single-turn harmful requests: 'standard' functional cat, no context.
    std = [r for r in rows
           if r["FunctionalCategory"] == "standard" and not r["ContextString"].strip()]
    print(f"[filter] {len(std)} standard behaviors with empty context")

    by_cat = defaultdict(list)
    for r in std:
        by_cat[r["SemanticCategory"]].append(r)

    rng = random.Random(args.seed)
    picked = []
    for cat in sorted(by_cat):
        pool = sorted(by_cat[cat], key=lambda r: r["BehaviorID"])  # stable order
        rng.shuffle(pool)
        take = pool[: args.per_category]
        picked.extend(take)
        print(f"  {cat}: pool={len(pool)} took={len(take)}")

    picked.sort(key=lambda r: r["BehaviorID"])
    out = [{"id": r["BehaviorID"], "question": r["Behavior"],
            "semantic_category": r["SemanticCategory"]} for r in picked]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {args.out}  n={len(out)}  (bodies not printed)")


if __name__ == "__main__":
    main()
