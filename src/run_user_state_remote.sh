#!/bin/bash
# Remote orchestrator for user-state experiment sweep.
# Run on the vast.ai A100 instance via:
#   nohup bash src/run_user_state_remote.sh > /workspace/user_state_remote.log 2>&1 &
#
# Requires HF_TOKEN env var to be set before invocation.

set -e
cd /workspace/persona_coinflip
export HF_HUB_ENABLE_HF_TRANSFER=1
export TRANSFORMERS_VERBOSITY=error
# HF_TOKEN must be set in the environment by the caller (or sourced before this script).
if [ -z "$HF_TOKEN" ]; then
  if [ -f /workspace/.hf_token ]; then
    export HF_TOKEN="$(cat /workspace/.hf_token)"
  else
    echo "FATAL: HF_TOKEN not set and /workspace/.hf_token not found" >&2
    exit 1
  fi
fi

mkdir -p results/user_state

DATASETS="data/axes_user_state/happy_sad.json data/axes_user_state/calm_angry.json data/axes_user_state/confident_anxious.json data/axes_user_state/curious_bored.json data/axes_user_state/hopeful_despairing.json data/axes_user_state/trusting_suspicious.json data/axes_user_state/energetic_lethargic.json data/axes_user_state/nonsexual_sexual.json"

echo "==================== STAGE 1: base+instruct on 8 axes ===================="
date

echo "--- Llama 3.1 8B base ---"
python3 src/run_coinflip_multi.py meta-llama/Llama-3.1-8B \
  --datasets $DATASETS \
  --output-dir results/user_state \
  --output-tag llama-3.1-8b-base

echo "--- Llama 3.1 8B Instruct ---"
python3 src/run_coinflip_multi.py meta-llama/Llama-3.1-8B-Instruct \
  --datasets $DATASETS \
  --output-dir results/user_state \
  --output-tag llama-3.1-8b-instruct

echo "--- Qwen 2.5 7B base ---"
python3 src/run_coinflip_multi.py Qwen/Qwen2.5-7B \
  --datasets $DATASETS \
  --output-dir results/user_state \
  --output-tag qwen-2.5-7b-base

echo "--- Qwen 2.5 7B Instruct ---"
python3 src/run_coinflip_multi.py Qwen/Qwen2.5-7B-Instruct \
  --datasets $DATASETS \
  --output-dir results/user_state \
  --output-tag qwen-2.5-7b-instruct

echo "==================== STAGE 2: OCT LoRAs on 8 axes (Llama 8B base) ===================="
date

for PERSONA in loving sarcasm nonchalance remorse; do
  echo "--- OCT-$PERSONA on Llama 3.1 8B Instruct ---"
  python3 src/run_coinflip_multi.py maius/llama-3.1-8b-it-personas \
    --base-model meta-llama/Llama-3.1-8B-Instruct \
    --subfolder $PERSONA \
    --datasets $DATASETS \
    --output-dir results/user_state \
    --output-tag oct-llama-3.1-8b-$PERSONA
done

echo "==================== STAGE 3: empty user-turn top-50 ===================="
date

mkdir -p results/empty_user_turn

echo "--- Llama 3.1 8B base ---"
python3 src/run_empty_user_turn.py meta-llama/Llama-3.1-8B \
  --output results/empty_user_turn/llama-3.1-8b-base.json

echo "--- Llama 3.1 8B Instruct ---"
python3 src/run_empty_user_turn.py meta-llama/Llama-3.1-8B-Instruct \
  --output results/empty_user_turn/llama-3.1-8b-instruct.json

echo "--- Qwen 2.5 7B base ---"
python3 src/run_empty_user_turn.py Qwen/Qwen2.5-7B \
  --output results/empty_user_turn/qwen-2.5-7b-base.json

echo "--- Qwen 2.5 7B Instruct ---"
python3 src/run_empty_user_turn.py Qwen/Qwen2.5-7B-Instruct \
  --output results/empty_user_turn/qwen-2.5-7b-instruct.json

echo "--- OCT-loving on Llama 8B ---"
python3 src/run_empty_user_turn.py maius/llama-3.1-8b-it-personas \
  --base-model meta-llama/Llama-3.1-8B-Instruct \
  --subfolder loving \
  --output results/empty_user_turn/oct-loving.json

echo "--- OCT-sarcasm on Llama 8B ---"
python3 src/run_empty_user_turn.py maius/llama-3.1-8b-it-personas \
  --base-model meta-llama/Llama-3.1-8B-Instruct \
  --subfolder sarcasm \
  --output results/empty_user_turn/oct-sarcasm.json

echo "==================== ALL DONE ===================="
date
