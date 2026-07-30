"""score_introspection_judge.py — grade introspection trials with Macar's rubrics.

Reads a runner output JSON (result['trials'] from run_introspection_ndif.py) and
sends each trial to a Claude judge with the appropriate rubric from
data/introspection_prompts.json (Macar Tables 5/6/7):

  - detection  (Table 5): control + injection trials -> "did it claim to detect?"
                          drives TPR (on injection) and FPR (on control).
  - introspection (Table 6): injection trials -> "detect AND identify the concept?"
                          (anti-self-priming: naming must precede/accompany the word).
  - forced_id  (Table 7): forced_id trials -> "does the prefill completion name the concept?"

Writes results/introspection/{tag}__judged.json with one row per trial carrying the
per-rubric labels, so re-runs never re-judge already-labeled trials. Aggregation into
TPR/FPR/introspection-rate with Wilson CIs lives in analyze_introspection.py.

Judge model default = claude-sonnet-4 (repo convention: Sonnet 4 for grading, not 4.5;
see memory reference-anthropic-judge). Concept words are benign (ocean, guitar, ...),
so replies are safe to send through the judge API — introspection_prompts.json says so.
"""
import argparse, json, re, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import httpx

ROOT = Path(__file__).parent.parent
PROMPTS_PATH = ROOT / "data" / "introspection_prompts.json"
API_KEY_PATH = Path("/root/.secrets/anthropic_api_key")

ANS_RE = re.compile(r"Answer:\s*\**\s*(YES|NO)", re.IGNORECASE)


def parse_answer(text):
    """Extract the judge's final YES/NO. Rubrics allow CoT then 'Answer: YES/NO'."""
    if not text:
        return None
    ms = list(ANS_RE.finditer(text))
    if ms:
        return ms[-1].group(1).lower()
    t = text.strip().lower()
    if t.startswith("yes"):
        return "yes"
    if t.startswith("no"):
        return "no"
    return None


def fill(template, **kw):
    """Substitute {key} placeholders by literal replace (safe against stray braces)."""
    for k, v in kw.items():
        template = template.replace("{" + k + "}", str(v))
    return template


# Base models don't respect turn boundaries: with injection they generate full length
# and hallucinate further "User:/Trial 2:/Assistant:" turns after their real answer.
# The judge must see ONLY this trial's first response, else a hallucinated later
# "Assistant: Yes..." contaminates the detection label. Cut at the first next-turn marker.
_TURN_MARKERS = ("\nUser:", "\nUser :", "\nuser:", " User:",
                 "\nTrial ", " Trial ", "\nAssistant:", "\nassistant:", " Assistant:")


def first_turn(reply):
    if not reply:
        return reply
    cut = len(reply)
    for m in _TURN_MARKERS:
        i = reply.find(m, 1)
        if i != -1:
            cut = min(cut, i)
    return reply[:cut].strip()


def judge_call(api_key, model, rubric, timeout=60, max_retries=3):
    """One judge call -> ('yes'|'no'|None, err|None). Retries transient HTTP errors."""
    for attempt in range(max_retries):
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
                    "max_tokens": 512,
                    "messages": [{"role": "user", "content": rubric}],
                },
                timeout=timeout,
            )
            if r.status_code == 200:
                text = r.json()["content"][0]["text"]
                return parse_answer(text), None
            if r.status_code in (429, 500, 502, 503, 529) and attempt < max_retries - 1:
                time.sleep(2 ** attempt * 1.5)
                continue
            return None, f"http{r.status_code}"
        except (httpx.ReadTimeout, httpx.ConnectTimeout):
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return None, "timeout"
        except Exception as e:
            return None, type(e).__name__
    return None, "exhausted"


def trial_key(t):
    return (t["condition"], t.get("concept") or "", float(t["alpha"]), t["kind"], t["trial"])


def rubrics_for(pj, t):
    """Which (judge_type, rubric_text) calls a trial needs, keyed by its kind."""
    reply = first_turn(t.get("reply"))
    if not reply or not reply.strip():
        return []  # error/empty cell: nothing to judge
    concept = t.get("concept") or ""
    question = fill(pj["macar_trial"], trial_num=t["trial"])
    jobs = []
    if t["kind"] in ("control", "injection"):
        jobs.append(("detect", fill(pj["judge_detection"], prompt=question, response=reply)))
    if t["kind"] == "injection":
        jobs.append(("introspect",
                     fill(pj["judge_introspection"], prompt=question, response=reply, word=concept)))
    if t["kind"] == "forced_id":
        jobs.append(("forced_id", fill(pj["judge_forced_id"], response=reply, word=concept)))
    return jobs


def needs_judging(existing, key, jtype):
    """Judge only if we have no final label yet (missing or a prior error)."""
    row = existing.get(key)
    if row is None:
        return True
    return row.get(jtype) not in ("yes", "no")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result_json", help="runner output with result['trials']")
    ap.add_argument("--model", default="claude-sonnet-4-5-20250929",
                    help="Anthropic judge model. Repo convention was Sonnet 4, but that "
                         "id is retired from the API; content here is benign concept words "
                         "so the 'less-supervised older model' rationale does not apply.")
    ap.add_argument("--out", default=None, help="default: <result>__judged.json alongside input")
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="only issue first N judge calls (smoke)")
    ap.add_argument("--dry-run", action="store_true", help="count calls, don't hit the API")
    args = ap.parse_args()

    pj = json.loads(PROMPTS_PATH.read_text())
    res = json.loads(Path(args.result_json).read_text())
    trials = res.get("trials", [])
    if not trials:
        print(f"[error] no trials in {args.result_json} "
              f"(old-schema file? expected result['trials'])")
        return

    in_path = Path(args.result_json)
    out_path = Path(args.out) if args.out else in_path.with_name(in_path.stem + "__judged.json")

    # Resume: index prior judgments by trial key so we never re-pay for a final label.
    existing = {}
    if out_path.exists():
        prev = json.loads(out_path.read_text())
        for row in prev.get("judgments", []):
            k = (row["condition"], row["concept"], float(row["alpha"]), row["kind"], row["trial"])
            existing[k] = row
        print(f"[resume] {len(existing)} trials already have judgments in {out_path.name}")

    # Ensure every non-error trial has a judgment row; collect the outstanding calls.
    jobs = []  # (key, jtype, rubric_text)
    n_err_cells = 0
    for t in trials:
        k = trial_key(t)
        row = existing.setdefault(k, {
            "condition": k[0], "concept": k[1], "alpha": k[2], "kind": k[3], "trial": k[4],
            "detect": None, "introspect": None, "forced_id": None,
        })
        if not t.get("reply") or not t["reply"].strip():
            n_err_cells += 1
            row["cell_error"] = t.get("error", "empty_reply")
            continue
        for jtype, rubric in rubrics_for(pj, t):
            if needs_judging(existing, k, jtype):
                jobs.append((k, jtype, rubric))

    if args.limit:
        jobs = jobs[: args.limit]

    from collections import Counter
    by_type = Counter(j[1] for j in jobs)
    print(f"[plan] {len(trials)} trials ({n_err_cells} error/empty) | "
          f"{len(jobs)} judge calls to make: {dict(by_type)} | model={args.model}")
    if args.dry_run:
        return
    if not jobs:
        print("[done] nothing outstanding to judge.")
        _write(out_path, res, existing, args.model)
        _summary(existing)
        return

    api_key = API_KEY_PATH.read_text().strip()

    def run_job(job):
        key, jtype, rubric = job
        label, err = judge_call(api_key, args.model, rubric)
        return key, jtype, label, err

    done = 0
    n_ckpt = max(25, len(jobs) // 20)
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = [ex.submit(run_job, j) for j in jobs]
        for fut in as_completed(futures):
            key, jtype, label, err = fut.result()
            row = existing[key]
            if label in ("yes", "no"):
                row[jtype] = label
                row.pop(f"{jtype}_err", None)
            else:
                row[f"{jtype}_err"] = err or "no_answer_parsed"
            done += 1
            if done % n_ckpt == 0:
                _write(out_path, res, existing, args.model)
                print(f"  judged {done}/{len(jobs)}", flush=True)

    _write(out_path, res, existing, args.model)
    print(f"[wrote] {out_path}")
    _summary(existing)


def _write(out_path, res, existing, model):
    out = {
        "model_id": res.get("model_id"), "tag": res.get("tag"),
        "conditions": res.get("conditions"), "alphas": res.get("alphas"),
        "n_inject": res.get("n_inject"), "n_control": res.get("n_control"),
        "judge_model": model,
        "judgments": list(existing.values()),
    }
    out_path.write_text(json.dumps(out, indent=2))


def _summary(existing):
    """Quick TPR/FPR-ish readout so the run is legible without the full analyzer."""
    from collections import defaultdict
    # (condition) -> control detect counts;  (condition, alpha) -> injection counts
    fpr = defaultdict(lambda: [0, 0])          # cond -> [yes, n]
    tpr = defaultdict(lambda: [0, 0])          # (cond, alpha) -> [yes, n]
    intro = defaultdict(lambda: [0, 0])        # (cond, alpha) -> [yes, n]
    fid = defaultdict(lambda: [0, 0])          # (cond, alpha) -> [yes, n]
    for r in existing.values():
        c, a, kind = r["condition"], r["alpha"], r["kind"]
        if kind == "control" and r["detect"] in ("yes", "no"):
            fpr[c][1] += 1; fpr[c][0] += r["detect"] == "yes"
        elif kind == "injection":
            if r["detect"] in ("yes", "no"):
                tpr[(c, a)][1] += 1; tpr[(c, a)][0] += r["detect"] == "yes"
            if r["introspect"] in ("yes", "no"):
                intro[(c, a)][1] += 1; intro[(c, a)][0] += r["introspect"] == "yes"
        elif kind == "forced_id" and r["forced_id"] in ("yes", "no"):
            fid[(c, a)][1] += 1; fid[(c, a)][0] += r["forced_id"] == "yes"

    def rate(yn):
        y, n = yn
        return f"{y}/{n}={y/n:.2f}" if n else "  -  "

    print("\n=== quick judge summary (full metrics: analyze_introspection.py) ===")
    for c in sorted(fpr):
        print(f"  [{c}] FPR(control detect) = {rate(fpr[c])}")
    print("  condition            alpha   TPR(detect)   introspection   forced-id")
    keys = sorted(set(list(tpr) + list(intro) + list(fid)))
    for (c, a) in keys:
        print(f"  {c:18s}  a={a:<4}  {rate(tpr[(c,a)]):>11}   {rate(intro[(c,a)]):>11}   {rate(fid[(c,a)]):>9}")


if __name__ == "__main__":
    main()
