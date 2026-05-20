#!/bin/bash
# Phase 3c: Mode 1 (chat template) user-state coinflip, 8 axes × 4 models.
set -e
cd /workspace/persona_coinflip
export HF_HUB_ENABLE_HF_TRANSFER=1
export TRANSFORMERS_VERBOSITY=error
if [ -z "$HF_TOKEN" ]; then
  if [ -f /workspace/.hf_token ]; then
    export HF_TOKEN="$(cat /workspace/.hf_token)"
  fi
fi
mkdir -p results/user_state_mode1

DATASETS_M1="data/axes_user_state_mode1/happy_sad.json data/axes_user_state_mode1/calm_angry.json data/axes_user_state_mode1/confident_anxious.json data/axes_user_state_mode1/curious_bored.json data/axes_user_state_mode1/hopeful_despairing.json data/axes_user_state_mode1/trusting_suspicious.json data/axes_user_state_mode1/energetic_lethargic.json data/axes_user_state_mode1/nonsexual_sexual.json"

echo "==================== PHASE 3C: Mode 1 user-state on 4 models ===================="
date

for SPEC in \
  "Llama-3.1-8B-Instruct meta-llama/Llama-3.1-8B-Instruct - -" \
  "Qwen2.5-7B-Instruct Qwen/Qwen2.5-7B-Instruct - -" \
  "OCT-loving maius/llama-3.1-8b-it-personas meta-llama/Llama-3.1-8B-Instruct loving" \
  "OCT-sarcasm maius/llama-3.1-8b-it-personas meta-llama/Llama-3.1-8B-Instruct sarcasm"
do
  set -- $SPEC
  TAG=$1; MID=$2; BASE=$3; SUB=$4
  echo "--- mode1-user-state / $TAG ---"
  ARGS=""
  [ "$BASE" != "-" ] && ARGS="--base-model $BASE"
  [ "$SUB"  != "-" ] && ARGS="$ARGS --subfolder $SUB"
  python3 src/run_multiturn.py "$MID" \
    --datasets $DATASETS_M1 \
    --output-dir results/user_state_mode1 \
    --output-tag "$TAG" \
    $ARGS
done

echo "==================== PHASE 3C DONE ===================="
date
