#!/bin/bash
# Phase 4: scale-up to Qwen 14B/32B on user-state + Mode 2 prefill on open-weights.
set -e
cd /workspace/persona_coinflip
export HF_HUB_ENABLE_HF_TRANSFER=1
export TRANSFORMERS_VERBOSITY=error
if [ -z "$HF_TOKEN" ]; then
  if [ -f /workspace/.hf_token ]; then
    export HF_TOKEN="$(cat /workspace/.hf_token)"
  fi
fi
mkdir -p results/user_state_qwen_scale results/user_state_qwen_scale_mode1 results/mode2_prefill_openweights

DATASETS_M3="data/axes_user_state/happy_sad.json data/axes_user_state/calm_angry.json data/axes_user_state/confident_anxious.json data/axes_user_state/curious_bored.json data/axes_user_state/hopeful_despairing.json data/axes_user_state/trusting_suspicious.json data/axes_user_state/energetic_lethargic.json data/axes_user_state/nonsexual_sexual.json"

DATASETS_M1="data/axes_user_state_mode1/happy_sad.json data/axes_user_state_mode1/calm_angry.json data/axes_user_state_mode1/confident_anxious.json data/axes_user_state_mode1/curious_bored.json data/axes_user_state_mode1/hopeful_despairing.json data/axes_user_state_mode1/trusting_suspicious.json data/axes_user_state_mode1/energetic_lethargic.json data/axes_user_state_mode1/nonsexual_sexual.json"

DATASETS_M2="data/mode2_prefill/canonical.json data/mode2_prefill/happy_sad.json data/mode2_prefill/calm_angry.json data/mode2_prefill/confident_anxious.json data/mode2_prefill/curious_bored.json data/mode2_prefill/hopeful_despairing.json data/mode2_prefill/trusting_suspicious.json data/mode2_prefill/energetic_lethargic.json data/mode2_prefill/nonsexual_sexual.json"

echo "==================== PHASE 4.1: Qwen 14B user-state ===================="
date
for SPEC in \
  "qwen-2.5-14b-base Qwen/Qwen2.5-14B" \
  "qwen-2.5-14b-instruct Qwen/Qwen2.5-14B-Instruct"
do
  set -- $SPEC
  TAG=$1; MID=$2
  echo "--- Mode 3 / $TAG ---"
  python3 src/run_coinflip_multi.py "$MID" \
    --datasets $DATASETS_M3 \
    --output-dir results/user_state_qwen_scale \
    --output-tag "$TAG"
  # Mode 1 only for instruct (chat template needs an instruct model)
  if [[ "$TAG" == *instruct ]]; then
    echo "--- Mode 1 / $TAG ---"
    python3 src/run_multiturn.py "$MID" \
      --datasets $DATASETS_M1 \
      --output-dir results/user_state_qwen_scale_mode1 \
      --output-tag "$TAG"
  fi
done

echo "==================== PHASE 4.2: Qwen 32B user-state ===================="
date
for SPEC in \
  "qwen-2.5-32b-base Qwen/Qwen2.5-32B" \
  "qwen-2.5-32b-instruct Qwen/Qwen2.5-32B-Instruct"
do
  set -- $SPEC
  TAG=$1; MID=$2
  echo "--- Mode 3 / $TAG ---"
  python3 src/run_coinflip_multi.py "$MID" \
    --datasets $DATASETS_M3 \
    --output-dir results/user_state_qwen_scale \
    --output-tag "$TAG"
  if [[ "$TAG" == *instruct ]]; then
    echo "--- Mode 1 / $TAG ---"
    python3 src/run_multiturn.py "$MID" \
      --datasets $DATASETS_M1 \
      --output-dir results/user_state_qwen_scale_mode1 \
      --output-tag "$TAG"
  fi
done

echo "==================== PHASE 4.3: Mode 2 prefill on open-weights ===================="
date
for SPEC in \
  "Llama-3.1-8B-Instruct meta-llama/Llama-3.1-8B-Instruct - -" \
  "Qwen2.5-7B-Instruct Qwen/Qwen2.5-7B-Instruct - -" \
  "Qwen2.5-14B-Instruct Qwen/Qwen2.5-14B-Instruct - -" \
  "OCT-loving maius/llama-3.1-8b-it-personas meta-llama/Llama-3.1-8B-Instruct loving"
do
  set -- $SPEC
  TAG=$1; MID=$2; BASE=$3; SUB=$4
  echo "--- mode2 / $TAG ---"
  ARGS=""
  [ "$BASE" != "-" ] && ARGS="--base-model $BASE"
  [ "$SUB"  != "-" ] && ARGS="$ARGS --subfolder $SUB"
  python3 src/run_multiturn.py "$MID" \
    --datasets $DATASETS_M2 \
    --output-dir results/mode2_prefill_openweights \
    --output-tag "$TAG" \
    $ARGS
done

echo "==================== PHASE 4 DONE ===================="
date
