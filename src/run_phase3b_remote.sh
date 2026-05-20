#!/bin/bash
# Phase 3b: multi-turn coinflip + forum continuations on cached models.
set -e
cd /workspace/persona_coinflip
export HF_HUB_ENABLE_HF_TRANSFER=1
export TRANSFORMERS_VERBOSITY=error
if [ -z "$HF_TOKEN" ]; then
  if [ -f /workspace/.hf_token ]; then
    export HF_TOKEN="$(cat /workspace/.hf_token)"
  fi
fi
mkdir -p results/multiturn results/forum_continuations

echo "==================== PHASE 3B.1: multi-turn coinflip ===================="
date

DATASETS_MT="data/multiturn/n0.json data/multiturn/n1.json data/multiturn/n5.json data/multiturn/n10.json"

for SPEC in \
  "Llama-3.1-8B-Instruct meta-llama/Llama-3.1-8B-Instruct - -" \
  "Qwen2.5-7B-Instruct Qwen/Qwen2.5-7B-Instruct - -" \
  "OCT-loving maius/llama-3.1-8b-it-personas meta-llama/Llama-3.1-8B-Instruct loving"
do
  set -- $SPEC
  TAG=$1; MID=$2; BASE=$3; SUB=$4
  echo "--- multi-turn / $TAG ---"
  ARGS=""
  [ "$BASE" != "-" ] && ARGS="--base-model $BASE"
  [ "$SUB"  != "-" ] && ARGS="$ARGS --subfolder $SUB"
  python3 src/run_multiturn.py "$MID" \
    --datasets $DATASETS_MT \
    --output-dir results/multiturn \
    --output-tag "$TAG" \
    $ARGS
done

echo "==================== PHASE 3B.2: forum continuations ===================="
date

for SPEC in \
  "Llama-3.1-8B-base meta-llama/Llama-3.1-8B - -" \
  "Llama-3.1-8B-Instruct meta-llama/Llama-3.1-8B-Instruct - -" \
  "Qwen2.5-7B-base Qwen/Qwen2.5-7B - -" \
  "Qwen2.5-7B-Instruct Qwen/Qwen2.5-7B-Instruct - -" \
  "OCT-loving maius/llama-3.1-8b-it-personas meta-llama/Llama-3.1-8B-Instruct loving" \
  "OCT-sarcasm maius/llama-3.1-8b-it-personas meta-llama/Llama-3.1-8B-Instruct sarcasm"
do
  set -- $SPEC
  TAG=$1; MID=$2; BASE=$3; SUB=$4
  echo "--- forum / $TAG ---"
  ARGS=""
  [ "$BASE" != "-" ] && ARGS="--base-model $BASE"
  [ "$SUB"  != "-" ] && ARGS="$ARGS --subfolder $SUB"
  python3 src/run_forum_continuations.py "$MID" \
    --dataset data/forum_continuations/scenarios.json \
    --output "results/forum_continuations/${TAG}.json" \
    $ARGS
done

echo "==================== PHASE 3B DONE ===================="
date
