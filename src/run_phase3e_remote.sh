#!/bin/bash
# Phase 3e: wheel-of-fortune + user-state calibration on cached models.
set -e
cd /workspace/persona_coinflip
export HF_HUB_ENABLE_HF_TRANSFER=1
export TRANSFORMERS_VERBOSITY=error
if [ -z "$HF_TOKEN" ]; then
  if [ -f /workspace/.hf_token ]; then
    export HF_TOKEN="$(cat /workspace/.hf_token)"
  fi
fi
mkdir -p results/wheel_of_fortune results/calibration_user_state

echo "==================== PHASE 3E.1: wheel-of-fortune on 4 models ===================="
date

for SPEC in \
  "Llama-3.1-8B-Instruct meta-llama/Llama-3.1-8B-Instruct - -" \
  "Qwen2.5-7B-Instruct Qwen/Qwen2.5-7B-Instruct - -" \
  "OCT-loving maius/llama-3.1-8b-it-personas meta-llama/Llama-3.1-8B-Instruct loving" \
  "OCT-sarcasm maius/llama-3.1-8b-it-personas meta-llama/Llama-3.1-8B-Instruct sarcasm"
do
  set -- $SPEC
  TAG=$1; MID=$2; BASE=$3; SUB=$4
  echo "--- wof / $TAG ---"
  ARGS=""
  [ "$BASE" != "-" ] && ARGS="--base-model $BASE"
  [ "$SUB"  != "-" ] && ARGS="$ARGS --subfolder $SUB"
  python3 src/run_outcome_pair_multi.py "$MID" \
    --datasets data/wheel_of_fortune/wof_balanced.json data/wheel_of_fortune/wof_asymmetric.json \
    --output-dir results/wheel_of_fortune \
    --output-tag "$TAG" \
    --outcome-key sun_moon \
    $ARGS
done

echo "==================== PHASE 3E.2: user-state calibration sweep ===================="
date

for SPEC in \
  "Llama-3.1-8B-Instruct meta-llama/Llama-3.1-8B-Instruct - -" \
  "Qwen2.5-7B-Instruct Qwen/Qwen2.5-7B-Instruct - -" \
  "OCT-loving maius/llama-3.1-8b-it-personas meta-llama/Llama-3.1-8B-Instruct loving"
do
  set -- $SPEC
  TAG=$1; MID=$2; BASE=$3; SUB=$4
  echo "--- us-cal / $TAG ---"
  ARGS=""
  [ "$BASE" != "-" ] && ARGS="--base-model $BASE"
  [ "$SUB"  != "-" ] && ARGS="$ARGS --subfolder $SUB"
  python3 src/run_outcome_pair_multi.py "$MID" \
    --datasets data/calibration/colored_balls_energetic_lethargic.json data/calibration/colored_balls_hopeful_despairing.json \
    --output-dir results/calibration_user_state \
    --output-tag "$TAG" \
    --outcome-key red_blue \
    $ARGS
done

echo "==================== PHASE 3E DONE ===================="
date
