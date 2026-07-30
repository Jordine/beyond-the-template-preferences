"""RQ1a harm-compliance eval: sample assistant completions for direct harmful
questions across the harm-trained LoRAs, a neutral-trained control, and untuned
Qwen3-8B.

For each model x question, one greedy (temperature=0.0) chat completion is drawn
via the Tinker sampling API, think-off (empty <think></think>), and written to
results/rq1a_eval/harm/<tag>.json. Per-question resume: a rerun skips questions
already present in the per-model file.

OPERATIONAL DISCIPLINE: neither the harmful questions nor the model completions
are ever printed to stdout/logs. Only ids, counts, and lengths are emitted.
Completions live only on disk, to be scored by the judge script.
"""
import argparse, json, os, sys, time

TAG_TO_KEY = {
    "harm_A1": "harm/A1",
    "harm_A2": "harm/A2",
    "harm_A3": "harm/A3",
    "harm_A4": "harm/A4",
    "harm_op_A1": "harm_op/A1",
    "harm_op_A2": "harm_op/A2",
    "harm_op_A3": "harm_op/A3",
    "harm_op_A4": "harm_op/A4",
    "neutral_A1": "neutral/A1",
    "untuned": None,  # base Qwen3-8B, no LoRA
}
BASE_MODEL = "Qwen/Qwen3-8B"


def load_tokenizer():
    os.environ.setdefault("HF_TOKEN", open("/root/.secrets/hf_token_main").read().strip())
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(BASE_MODEL)


def render(tok, question):
    # think-off assistant open, matching the neutral/purple readout convention
    text = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return tok(text, add_special_tokens=False)["input_ids"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--questions", default="data/rq1a/harm_eval_questions.json")
    p.add_argument("--runs", default="results/rq1a_tinker/runs.json")
    p.add_argument("--out-dir", default="results/rq1a_eval/harm")
    p.add_argument("--tags", nargs="*", default=list(TAG_TO_KEY.keys()))
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    args = p.parse_args()

    os.environ.setdefault("TINKER_API_KEY", open("/root/.secrets/tinker_api_key").read().strip())
    import tinker

    questions = json.load(open(args.questions))
    runs = json.load(open(args.runs))
    os.makedirs(args.out_dir, exist_ok=True)
    tok = load_tokenizer()
    sc = tinker.ServiceClient()

    for tag in args.tags:
        if tag not in TAG_TO_KEY:
            print(f"[warn] unknown tag {tag}, skipping")
            continue
        out_path = os.path.join(args.out_dir, f"{tag}.json")

        # Resume: load existing per-question records
        done = {}
        if os.path.exists(out_path):
            try:
                prev = json.load(open(out_path))
                done = {r["id"]: r for r in prev.get("results", [])}
            except Exception:
                done = {}
        todo = [q for q in questions if q["id"] not in done]
        print(f"[{tag}] {len(done)} cached, {len(todo)} to sample", flush=True)
        if not todo:
            continue

        key = TAG_TO_KEY[tag]
        if key is None:
            client = sc.create_sampling_client(base_model=BASE_MODEL)
            sampler_path = f"base:{BASE_MODEL}"
        else:
            sampler_path = runs[key]["sampler_path"]
            client = sc.create_sampling_client(model_path=sampler_path)

        sp = tinker.SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature)
        results = list(done.values())
        t0 = time.time()
        n_new = 0
        for q in todo:
            ids = render(tok, q["question"])
            try:
                resp = client.sample(
                    prompt=tinker.ModelInput.from_ints(tokens=ids),
                    num_samples=1, sampling_params=sp).result()
                comp = tok.decode(list(resp.sequences[0].tokens_np)).split("<|im_end|>")[0]
                rec = {"id": q["id"], "semantic_category": q.get("semantic_category"),
                       "completion": comp, "completion_chars": len(comp)}
            except Exception as e:
                rec = {"id": q["id"], "semantic_category": q.get("semantic_category"),
                       "completion": None, "error": f"{type(e).__name__}:{str(e)[:120]}"}
            results.append(rec)
            n_new += 1
            # periodic checkpoint so an interrupt loses at most a few samples
            if n_new % 10 == 0:
                _save(out_path, tag, sampler_path, args, results)
                print(f"  [{tag}] {n_new}/{len(todo)}  {time.time()-t0:.1f}s", flush=True)
        _save(out_path, tag, sampler_path, args, results)
        n_err = sum(1 for r in results if r.get("completion") is None)
        print(f"[{tag}] done: {len(results)} total, {n_err} errors -> {out_path}", flush=True)


def _save(out_path, tag, sampler_path, args, results):
    payload = {
        "tag": tag,
        "sampler_path": sampler_path,
        "base_model": BASE_MODEL,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "render": "chat im_start user/assistant, think-off (empty <think></think>)",
        "n": len(results),
        "results": sorted(results, key=lambda r: r["id"]),
    }
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, out_path)


if __name__ == "__main__":
    main()
