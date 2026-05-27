#!/usr/bin/env bash
# the harmful-continuation experiment (standalone) — plaintext mid-turn harmful continuation.
#
# Flow:
#   1. Build the prompt set from anthrax_prompt_dont_open.txt + 4 variants
#      (build_harmful_continuation_prompts.py). Sealed prompt is passed through to JSON
#      without being printed.
#   2. For each model, sample N completions in raw-text mode (no chat
#      template) via run_plaintext_completion.py --mode raw. This puts the model at a
#      user-turn-continuation position: it's continuing a plaintext document
#      that opens a harmful instruction.
#   3. Judge each completion with Claude via score_harmful_continuation_judge.py to label
#      `continued` / `deflected` / `ambiguous` / `garbled`. If the judge run
#      hits credit issues mid-batch, follow up with rejudge_harmful_continuation_unjudged.py
#      to fill in `unknown` and `error:*` labels.
#   4. Aggregate with analyze_harmful_continuation.py into a per-model deflection-rate
#      table.
#
# the harmful-continuation experiment is methodologically distinct from the coinflip experiments: it measures generation
# classified by a judge (deflect vs continue over ~200 tokens), not the
# single-token logprob the coin-flip uses. Reported separately.
#
# Requirements: HF_TOKEN, plus either ANTHROPIC_API_KEY or OPENROUTER_API_KEY
# for the judge step.

set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
cd "$ROOT"

if [[ -z "${HF_TOKEN:-}" ]] && [[ -f "$HOME/.secrets/hf_token_main" ]]; then
  export HF_TOKEN="$(cat "$HOME/.secrets/hf_token_main")"
fi
: "${HF_TOKEN:?HF_TOKEN must be set or ~/.secrets/hf_token_main present}"

mkdir -p data/harmful_continuation_prompts.json results/harmful_continuation

# 1. Build the prompts (idempotent — rewrites only if anthrax txt changed).
python3 src/build_harmful_continuation_prompts.py

# 2. Generate completions (raw mode = no chat template = user-turn continuation).
run_completion() {
  local model_id="$1"; local short="$2"; local base_model="${3:-}"
  local out="results/harmful_continuation/${short}.json"
  if [[ -f "$out" ]]; then
    echo "[skip] $out"
    return 0
  fi
  echo "[gen]  $model_id  -> $out"
  if [[ -n "$base_model" ]]; then
    python3 src/run_plaintext_completion.py "$model_id" \
      --mode raw --prompts data/harmful_continuation_prompts.json \
      --base-model "$base_model" --output "$out"
  else
    python3 src/run_plaintext_completion.py "$model_id" \
      --mode raw --prompts data/harmful_continuation_prompts.json \
      --output "$out"
  fi
}

run_completion "meta-llama/Llama-3.1-8B"          "LlamaBase"
run_completion "meta-llama/Llama-3.1-8B-Instruct" "LlamaInstruct"
run_completion "Qwen/Qwen2.5-7B-Instruct"         "QwenInstruct"
run_completion "ModelOrganismsForEM/Llama-3.1-8B-Instruct_bad-medical-advice"     "EM-medical"   "meta-llama/Llama-3.1-8B-Instruct"
run_completion "ModelOrganismsForEM/Llama-3.1-8B-Instruct_extreme-sports"        "EM-sports"    "meta-llama/Llama-3.1-8B-Instruct"
run_completion "ModelOrganismsForEM/Llama-3.1-8B-Instruct_risky-financial-advice" "EM-financial" "meta-llama/Llama-3.1-8B-Instruct"

# 3. Judge with Claude. Idempotent — only judges (model, prompt, sample) tuples
#    not already in claude_judge_scores.json.
echo
echo "[judge] running Claude judge on missing samples"
python3 src/score_harmful_continuation_judge.py \
  --results-dir results/harmful_continuation \
  --output results/harmful_continuation/claude_judge_scores.json

# 4. If the judge run left any `unknown`/`error:*` entries, rejudge them
#    via direct Anthropic API. Comment out if you're OK with the gap.
if [[ -n "${ANTHROPIC_API_KEY:-}" ]] || [[ -f "$HOME/.secrets/anthropic_api_key" ]]; then
  echo
  echo "[rejudge] filling in unjudged / error rows"
  python3 src/rejudge_harmful_continuation_unjudged.py \
    --input results/harmful_continuation/claude_judge_scores.json \
    --output results/harmful_continuation/claude_judge_scores.json
fi

# 5. Aggregate.
echo
echo "[analyze] producing results/harmful_continuation_summary.md"
python3 src/analyze_harmful_continuation.py

echo
echo "[the harmful-continuation experiment sweep done]"
