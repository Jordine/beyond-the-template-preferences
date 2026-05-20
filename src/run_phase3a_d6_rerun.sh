#!/bin/bash
# Re-run D6 only (colored balls were fine).
set -e
cd /workspace/persona_coinflip
export HF_HUB_ENABLE_HF_TRANSFER=1
export TRANSFORMERS_VERBOSITY=error
if [ -z "$HF_TOKEN" ]; then
  if [ -f /workspace/.hf_token ]; then
    export HF_TOKEN="$(cat /workspace/.hf_token)"
  fi
fi
mkdir -p results/d6_variants

echo "==================== D6 rerun (fixed tokenization) ===================="
date

for SPEC in \
  "Llama-3.1-8B-base meta-llama/Llama-3.1-8B - -" \
  "Llama-3.1-8B-Instruct meta-llama/Llama-3.1-8B-Instruct - -" \
  "Qwen2.5-7B-Instruct Qwen/Qwen2.5-7B-Instruct - -" \
  "OCT-loving maius/llama-3.1-8b-it-personas meta-llama/Llama-3.1-8B-Instruct loving" \
  "OCT-sarcasm maius/llama-3.1-8b-it-personas meta-llama/Llama-3.1-8B-Instruct sarcasm"
do
  set -- $SPEC
  TAG=$1; MID=$2; BASE=$3; SUB=$4
  echo "--- D6 / $TAG ---"
  ARGS=""
  [ "$BASE" != "-" ] && ARGS="--base-model $BASE"
  [ "$SUB"  != "-" ] && ARGS="$ARGS --subfolder $SUB"
  python3 src/run_outcome_pair_multi.py "$MID" \
    --datasets data/d6_variants/d6_balanced.json data/d6_variants/d6_asymmetric.json \
    --output-dir results/d6_variants \
    --output-tag "$TAG" \
    --outcome-key d6_low_high \
    $ARGS
done

echo "==================== D6 RERUN DONE ===================="
date
