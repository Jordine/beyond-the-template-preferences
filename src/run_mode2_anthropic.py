"""Mode 2 prefill experiment on Claude via Anthropic API.

For each item, send (user_wrapper, assistant_prefill) and let the model
continue the assistant turn. The prefill ends with 'User: It came up' so
the natural continuation is ' heads' or ' tails'.

Since the API doesn't return logprobs, we estimate P(heads) by sampling
N times per item at temperature=1 and counting first-word labels.

Usage:
  python3 src/run_mode2_anthropic.py --model claude-haiku-4-5 \\
      --dataset data/mode2_prefill/canonical.json \\
      --output results/mode2_anthropic/haiku-4-5__canonical.json \\
      --n-samples 30
"""
import argparse, json, os, re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import httpx


def call_once(api_key, model, messages, max_tokens=15):
    try:
        r = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": max_tokens,
                "temperature": 1.0,
                "messages": messages,
            },
            timeout=45,
        )
    except Exception as e:
        return None, f"exc:{type(e).__name__}"
    if r.status_code != 200:
        return None, f"http{r.status_code}"
    try:
        body = r.json()
        content = body.get("content") or []
        if not content:
            return "", None  # empty completion, classify as "other"
        return content[0].get("text", ""), None
    except Exception as e:
        return None, f"parse:{type(e).__name__}"


HEADS_PAT = re.compile(r"\bheads\b", re.IGNORECASE)
TAILS_PAT = re.compile(r"\btails\b", re.IGNORECASE)


def classify(text):
    """Look for first appearance of 'heads' or 'tails' anywhere in response."""
    if not text:
        return "other"
    h = HEADS_PAT.search(text)
    t = TAILS_PAT.search(text)
    if h and t:
        return "heads" if h.start() < t.start() else "tails"
    if h:
        return "heads"
    if t:
        return "tails"
    return "other"


def run_item(api_key, model, item, n_samples):
    counts = {"heads": 0, "tails": 0, "other": 0, "error": 0}
    raw_samples = []
    for _ in range(n_samples):
        text, err = call_once(api_key, model, item["messages"])
        if err:
            counts["error"] += 1
            raw_samples.append({"text": None, "error": err})
            continue
        lab = classify(text)
        counts[lab] += 1
        raw_samples.append({"text": text[:20], "label": lab})
    return {"id": item["id"], "preferred_outcome": item["preferred_outcome"],
            "counts": counts, "samples": raw_samples}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="claude-haiku-4-5")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-samples", type=int, default=30)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    api_key = Path("/root/.secrets/anthropic_api_key").read_text().strip()
    items = json.loads(Path(args.dataset).read_text())
    print(f"[run] {args.model} on {Path(args.dataset).name}, n_items={len(items)}, n_samples={args.n_samples}")

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(run_item, api_key, args.model, it, args.n_samples): it for it in items}
        for i, fut in enumerate(as_completed(futures)):
            results.append(fut.result())
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(items)}", flush=True)

    # aggregate
    pref_h = []
    pref_t = []
    for r in results:
        counts = r["counts"]
        n_valid = counts["heads"] + counts["tails"]
        if n_valid == 0:
            continue
        p_heads = counts["heads"] / n_valid
        if r["preferred_outcome"] == "heads":
            pref_h.append(p_heads)
        else:
            pref_t.append(p_heads)
    mean_q_h = sum(pref_h) / len(pref_h) if pref_h else None
    mean_q_t = sum(pref_t) / len(pref_t) if pref_t else None
    two_s = (mean_q_h - mean_q_t) if (mean_q_h is not None and mean_q_t is not None) else None

    summary = {
        "model": args.model,
        "dataset": args.dataset,
        "n_items": len(items),
        "n_samples_per_item": args.n_samples,
        "mean_q_when_pref_heads": mean_q_h,
        "mean_q_when_pref_tails": mean_q_t,
        "2s": two_s,
        "results": results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(summary, indent=2))
    print(f"[done] q^H={mean_q_h}, q^T={mean_q_t}, 2s={two_s}")
    print(f"[saved] {args.output}")


if __name__ == "__main__":
    main()
