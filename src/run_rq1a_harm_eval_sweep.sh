#!/usr/bin/env bash
# End-to-end RQ1a harm-compliance eval: build questions -> sample completions ->
# judge -> summarize. Idempotent/resumable: each stage skips work already on disk.
#
# OPERATIONAL DISCIPLINE: no harmful questions or completions are printed. Only
# counts, ids, and compliance rates reach stdout.
set -euo pipefail
cd "$(dirname "$0")/.."

export HF_TOKEN="$(cat /root/.secrets/hf_token_main)"

echo "== [1/4] build question set =="
python3 src/build_harm_eval_questions.py

echo "== [2/4] sample completions (6 models, greedy) =="
python3 src/run_rq1a_harm_eval.py

echo "== [3/4] judge completions (Sonnet 4 via LiteLLM) =="
python3 src/score_rq1a_harm_eval_judge.py

echo "== [4/4] summarize =="
python3 src/analyze_rq1a_harm_eval.py

echo "== done =="
