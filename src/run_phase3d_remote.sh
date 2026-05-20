#!/bin/bash
# Phase 3d: remaining 6 OCT personas on Mode 1 user-state axes.
# Plus Mode 3 user-state on same 6 personas to complete the matrix.
set -e
cd /workspace/persona_coinflip
export HF_HUB_ENABLE_HF_TRANSFER=1
export TRANSFORMERS_VERBOSITY=error
if [ -z "$HF_TOKEN" ]; then
  if [ -f /workspace/.hf_token ]; then
    export HF_TOKEN="$(cat /workspace/.hf_token)"
  fi
fi
mkdir -p results/user_state_mode1 results/user_state

DATASETS_M1="data/axes_user_state_mode1/happy_sad.json data/axes_user_state_mode1/calm_angry.json data/axes_user_state_mode1/confident_anxious.json data/axes_user_state_mode1/curious_bored.json data/axes_user_state_mode1/hopeful_despairing.json data/axes_user_state_mode1/trusting_suspicious.json data/axes_user_state_mode1/energetic_lethargic.json data/axes_user_state_mode1/nonsexual_sexual.json"

DATASETS_M3="data/axes_user_state/happy_sad.json data/axes_user_state/calm_angry.json data/axes_user_state/confident_anxious.json data/axes_user_state/curious_bored.json data/axes_user_state/hopeful_despairing.json data/axes_user_state/trusting_suspicious.json data/axes_user_state/energetic_lethargic.json data/axes_user_state/nonsexual_sexual.json"

echo "==================== PHASE 3D: 6 OCT personas × Mode 1 + Mode 3 user-state ===================="
date

for PERSONA in goodness humor impulsiveness mathematical poeticism sycophancy; do
  echo "--- OCT-$PERSONA Mode 3 user-state ---"
  python3 src/run_coinflip_multi.py maius/llama-3.1-8b-it-personas \
    --base-model meta-llama/Llama-3.1-8B-Instruct \
    --subfolder $PERSONA \
    --datasets $DATASETS_M3 \
    --output-dir results/user_state \
    --output-tag "oct-llama-3.1-8b-$PERSONA"

  echo "--- OCT-$PERSONA Mode 1 user-state ---"
  python3 src/run_multiturn.py maius/llama-3.1-8b-it-personas \
    --base-model meta-llama/Llama-3.1-8B-Instruct \
    --subfolder $PERSONA \
    --datasets $DATASETS_M1 \
    --output-dir results/user_state_mode1 \
    --output-tag "OCT-$PERSONA"
done

echo "==================== PHASE 3D DONE ===================="
date
