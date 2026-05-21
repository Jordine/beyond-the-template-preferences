#!/bin/bash
# Phase 5 orchestrator on 2x A100 80GB.
set -e
cd /workspace/persona_coinflip
export HF_HUB_ENABLE_HF_TRANSFER=1
export TRANSFORMERS_VERBOSITY=error
if [ -z "$HF_TOKEN" ]; then
  if [ -f /workspace/.hf_token ]; then
    export HF_TOKEN="$(cat /workspace/.hf_token)"
  fi
fi
mkdir -p results/mode2_prefill_corrected results/multiturn results/logit_lens \
         results/user_state_qwen_scale results/user_state_qwen_scale_mode1 \
         results/olmo_user_state

# ======== E1: Mode 2 corrected on small models ========
echo "==================== E1: Mode 2 corrected ===================="
date

for SPEC in \
  "Llama-3.1-8B-Instruct meta-llama/Llama-3.1-8B-Instruct - -" \
  "Qwen2.5-7B-Instruct Qwen/Qwen2.5-7B-Instruct - -" \
  "Qwen2.5-14B-Instruct Qwen/Qwen2.5-14B-Instruct - -" \
  "OCT-loving maius/llama-3.1-8b-it-personas meta-llama/Llama-3.1-8B-Instruct loving"
do
  set -- $SPEC
  TAG=$1; MID=$2; BASE=$3; SUB=$4
  echo "--- E1: mode2corr / $TAG ---"
  ARGS=""
  [ "$BASE" != "-" ] && ARGS="--base-model $BASE"
  [ "$SUB"  != "-" ] && ARGS="$ARGS --subfolder $SUB"
  python3 src/run_multiturn.py "$MID" \
    --datasets data/mode2_prefill_corrected/canonical.json \
               data/mode2_prefill_corrected/energetic_lethargic.json \
               data/mode2_prefill_corrected/hopeful_despairing.json \
    --output-dir results/mode2_prefill_corrected \
    --output-tag "$TAG" \
    $ARGS
done

# ======== E2: Qwen 14B Mode 1 canonical ========
echo "==================== E2: Qwen 14B Mode 1 canonical ===================="
date
python3 src/run_multiturn.py Qwen/Qwen2.5-14B-Instruct \
  --datasets data/multiturn/n0.json \
  --output-dir results/multiturn \
  --output-tag "Qwen2.5-14B-Instruct"

# ======== E5: Logit lens experiments ========
echo "==================== E5: Logit lens ===================="
date

# Llama 8B Base (Mode 3 plaintext)
python3 src/run_logit_lens.py meta-llama/Llama-3.1-8B \
  --dataset data/psm_canonical.json \
  --output results/logit_lens/Llama-3.1-8B-base__mode3.json \
  --messages-mode prompt

# Llama 8B Inst (Mode 3)
python3 src/run_logit_lens.py meta-llama/Llama-3.1-8B-Instruct \
  --dataset data/psm_canonical.json \
  --output results/logit_lens/Llama-3.1-8B-Instruct__mode3.json \
  --messages-mode prompt

# Llama 8B Inst (Mode 1 — via the multiturn n=0 dataset with chat template)
python3 src/run_logit_lens.py meta-llama/Llama-3.1-8B-Instruct \
  --dataset data/multiturn/n0.json \
  --output results/logit_lens/Llama-3.1-8B-Instruct__mode1.json \
  --messages-mode messages

# Llama 8B Inst (Mode 2 corrected)
python3 src/run_logit_lens.py meta-llama/Llama-3.1-8B-Instruct \
  --dataset data/mode2_prefill_corrected/canonical.json \
  --output results/logit_lens/Llama-3.1-8B-Instruct__mode2corr.json \
  --messages-mode messages

# OCT-loving (Mode 1)
python3 src/run_logit_lens.py maius/llama-3.1-8b-it-personas \
  --base-model meta-llama/Llama-3.1-8B-Instruct \
  --subfolder loving \
  --dataset data/multiturn/n0.json \
  --output results/logit_lens/OCT-loving__mode1.json \
  --messages-mode messages

# EM-sports (Mode 3) — the one that flips 2s to -0.18 on canonical
python3 src/run_logit_lens.py ModelOrganismsForEM/Llama-3.1-8B-Instruct_extreme-sports \
  --base-model meta-llama/Llama-3.1-8B-Instruct \
  --dataset data/psm_canonical.json \
  --output results/logit_lens/EM-sports__mode3.json \
  --messages-mode prompt

# Qwen 7B Inst (Mode 3 + Mode 1)
python3 src/run_logit_lens.py Qwen/Qwen2.5-7B-Instruct \
  --dataset data/psm_canonical.json \
  --output results/logit_lens/Qwen2.5-7B-Instruct__mode3.json \
  --messages-mode prompt
python3 src/run_logit_lens.py Qwen/Qwen2.5-7B-Instruct \
  --dataset data/multiturn/n0.json \
  --output results/logit_lens/Qwen2.5-7B-Instruct__mode1.json \
  --messages-mode messages

# Qwen 14B Inst (Mode 3 + Mode 1)
python3 src/run_logit_lens.py Qwen/Qwen2.5-14B-Instruct \
  --dataset data/psm_canonical.json \
  --output results/logit_lens/Qwen2.5-14B-Instruct__mode3.json \
  --messages-mode prompt
python3 src/run_logit_lens.py Qwen/Qwen2.5-14B-Instruct \
  --dataset data/multiturn/n0.json \
  --output results/logit_lens/Qwen2.5-14B-Instruct__mode1.json \
  --messages-mode messages

# ======== E4: OLMo user-state by stage ========
echo "==================== E4: OLMo user-state ===================="
date

DATASETS_M3="data/axes_user_state/happy_sad.json data/axes_user_state/calm_angry.json data/axes_user_state/confident_anxious.json data/axes_user_state/curious_bored.json data/axes_user_state/hopeful_despairing.json data/axes_user_state/trusting_suspicious.json data/axes_user_state/energetic_lethargic.json data/axes_user_state/nonsexual_sexual.json"

for SPEC in \
  "olmo-3-32b-base allenai/Olmo-3-1125-32B" \
  "olmo-3.1-32b-sft allenai/Olmo-3.1-32B-Instruct-SFT" \
  "olmo-3.1-32b-dpo allenai/Olmo-3.1-32B-Instruct-DPO" \
  "olmo-3.1-32b-rlvr allenai/Olmo-3.1-32B-Instruct"
do
  set -- $SPEC
  TAG=$1; MID=$2
  echo "--- E4: olmo-stage / $TAG ---"
  python3 src/run_coinflip_multi.py "$MID" \
    --datasets $DATASETS_M3 \
    --output-dir results/olmo_user_state \
    --output-tag "$TAG"
done

# ======== E3: Qwen 72B ========
echo "==================== E3: Qwen 72B ===================="
date

# 72B uses both GPUs via device_map="auto"
for SPEC in \
  "qwen-2.5-72b-base Qwen/Qwen2.5-72B" \
  "qwen-2.5-72b-instruct Qwen/Qwen2.5-72B-Instruct"
do
  set -- $SPEC
  TAG=$1; MID=$2
  echo "--- E3: 72B Mode 3 / $TAG ---"
  python3 src/run_coinflip_multi.py "$MID" \
    --datasets $DATASETS_M3 data/psm_canonical.json \
    --output-dir results/user_state_qwen_scale \
    --output-tag "$TAG"
  if [[ "$TAG" == *instruct ]]; then
    echo "--- E3: 72B Mode 1 / $TAG ---"
    DATASETS_M1="data/axes_user_state_mode1/happy_sad.json data/axes_user_state_mode1/calm_angry.json data/axes_user_state_mode1/confident_anxious.json data/axes_user_state_mode1/curious_bored.json data/axes_user_state_mode1/hopeful_despairing.json data/axes_user_state_mode1/trusting_suspicious.json data/axes_user_state_mode1/energetic_lethargic.json data/axes_user_state_mode1/nonsexual_sexual.json data/multiturn/n0.json"
    python3 src/run_multiturn.py "$MID" \
      --datasets $DATASETS_M1 \
      --output-dir results/user_state_qwen_scale_mode1 \
      --output-tag "$TAG"
  fi
done

echo "==================== PHASE 5 DONE ===================="
date
