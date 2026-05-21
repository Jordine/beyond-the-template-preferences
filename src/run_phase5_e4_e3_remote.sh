#!/bin/bash
# Replacement for E4 + E3 stages with cache cleanup between large downloads.
# Original Phase 5 script ran E1 + E2 + E5 (small models, fits in disk).
# This script handles the big models (OLMo 4 stages, Qwen 72B) with disk
# management.

set -e
cd /workspace/persona_coinflip
export HF_HUB_ENABLE_HF_TRANSFER=1
export TRANSFORMERS_VERBOSITY=error
if [ -z "$HF_TOKEN" ]; then
  if [ -f /workspace/.hf_token ]; then
    export HF_TOKEN="$(cat /workspace/.hf_token)"
  fi
fi
mkdir -p results/olmo_user_state results/user_state_qwen_scale results/user_state_qwen_scale_mode1

CACHE_DIR="$HOME/.cache/huggingface/hub"

clear_model() {
    # remove HF cache for a specific model (by repo string), e.g. allenai/Olmo-3.1-32B-Instruct-SFT
    local repo="$1"
    local safe="$(echo "$repo" | sed 's|/|--|g')"
    rm -rf "$CACHE_DIR/models--$safe"
    df -h / | tail -1
}

clear_all_except() {
    # remove everything not matching pattern
    local keep="$1"
    find "$CACHE_DIR" -maxdepth 1 -type d -name 'models--*' | while read d; do
        if [[ "$d" != *"$keep"* ]]; then
            echo "  clearing $d"
            rm -rf "$d"
        fi
    done
    df -h / | tail -1
}

# Free disk: drop the small instruct/persona models we no longer need
echo "==================== Pre-E4 cleanup ===================="
date
df -h / | tail -1
clear_model "meta-llama/Llama-3.1-8B-Instruct"
clear_model "meta-llama/Llama-3.1-8B"
clear_model "Qwen/Qwen2.5-7B-Instruct"
clear_model "Qwen/Qwen2.5-14B-Instruct"
clear_model "maius/llama-3.1-8b-it-personas"
clear_model "ModelOrganismsForEM/Llama-3.1-8B-Instruct_extreme-sports"

DATASETS_M3="data/axes_user_state/happy_sad.json data/axes_user_state/calm_angry.json data/axes_user_state/confident_anxious.json data/axes_user_state/curious_bored.json data/axes_user_state/hopeful_despairing.json data/axes_user_state/trusting_suspicious.json data/axes_user_state/energetic_lethargic.json data/axes_user_state/nonsexual_sexual.json"

echo "==================== E4: OLMo user-state by stage ===================="
date

for SPEC in \
  "olmo-3-32b-base allenai/Olmo-3-1125-32B" \
  "olmo-3.1-32b-sft allenai/Olmo-3.1-32B-Instruct-SFT" \
  "olmo-3.1-32b-dpo allenai/Olmo-3.1-32B-Instruct-DPO" \
  "olmo-3.1-32b-rlvr allenai/Olmo-3.1-32B-Instruct"
do
  set -- $SPEC
  TAG=$1; MID=$2
  echo "--- E4: olmo-stage / $TAG ---"
  date
  df -h / | tail -1
  python3 src/run_coinflip_multi.py "$MID" \
    --datasets $DATASETS_M3 \
    --output-dir results/olmo_user_state \
    --output-tag "$TAG"
  # clear this checkpoint immediately after we're done
  echo "  clearing $MID cache..."
  clear_model "$MID"
done

echo "==================== Pre-E3 cleanup ===================="
date
clear_all_except "donotmatch"  # clear all HF caches

echo "==================== E3: Qwen 72B ===================="
date

DATASETS_M1="data/axes_user_state_mode1/happy_sad.json data/axes_user_state_mode1/calm_angry.json data/axes_user_state_mode1/confident_anxious.json data/axes_user_state_mode1/curious_bored.json data/axes_user_state_mode1/hopeful_despairing.json data/axes_user_state_mode1/trusting_suspicious.json data/axes_user_state_mode1/energetic_lethargic.json data/axes_user_state_mode1/nonsexual_sexual.json data/multiturn/n0.json"

for SPEC in \
  "qwen-2.5-72b-base Qwen/Qwen2.5-72B" \
  "qwen-2.5-72b-instruct Qwen/Qwen2.5-72B-Instruct"
do
  set -- $SPEC
  TAG=$1; MID=$2
  echo "--- E3: 72B Mode 3 / $TAG ---"
  date
  df -h / | tail -1
  python3 src/run_coinflip_multi.py "$MID" \
    --datasets $DATASETS_M3 data/psm_canonical.json \
    --output-dir results/user_state_qwen_scale \
    --output-tag "$TAG"
  if [[ "$TAG" == *instruct ]]; then
    echo "--- E3: 72B Mode 1 / $TAG ---"
    date
    python3 src/run_multiturn.py "$MID" \
      --datasets $DATASETS_M1 \
      --output-dir results/user_state_qwen_scale_mode1 \
      --output-tag "$TAG"
  fi
  echo "  clearing $MID cache..."
  clear_model "$MID"
done

echo "==================== PHASE 5 E4+E3 DONE ===================="
date
df -h / | tail -1
