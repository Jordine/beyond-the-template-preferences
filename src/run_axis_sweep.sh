#!/bin/bash
# Phase 2: multi-axis OCT coinflip sweep.
# Runs each of 10 persona LoRAs + baseline on each of 10 axis datasets.
# Output: 11 configs x 10 axes = 110 runs per base model.
#
# Usage: BASE=llama3 bash src/run_axis_sweep.sh
#        BASE=qwen25 bash src/run_axis_sweep.sh
#        BASE=gemma3 bash src/run_axis_sweep.sh

set -u
cd /root/persona_coinflip
mkdir -p results/axes

AXES=(sarcasm humor remorse impulsiveness nonchalance sycophancy poeticism mathematical goodness loving)
PERSONAS=(goodness humor impulsiveness loving mathematical nonchalance poeticism remorse sarcasm sycophancy)

case "${BASE:-llama3}" in
  llama3)
    BASE_ID="meta-llama/Llama-3.1-8B-Instruct"
    OCT_REPO="maius/llama-3.1-8b-it-personas"
    BASE_LABEL="llama-3.1-8b"
    DTYPE="float16"
    ;;
  qwen25)
    BASE_ID="Qwen/Qwen2.5-7B-Instruct"
    OCT_REPO="maius/qwen-2.5-7b-it-personas"
    BASE_LABEL="qwen-2.5-7b"
    DTYPE="float16"
    ;;
  gemma3)
    BASE_ID="google/gemma-3-4b-it"
    OCT_REPO="maius/gemma-3-4b-it-personas"
    BASE_LABEL="gemma-3-4b"
    DTYPE="bfloat16"
    ;;
  *)
    echo "unknown BASE: ${BASE}"; exit 1 ;;
esac

LOG="results/axes/sweep_${BASE_LABEL}.log"
echo "=== axis sweep (${BASE_LABEL}) started $(date) ===" | tee -a $LOG

run_one() {
    local label="$1"; shift
    local out="results/axes/${label}.json"
    if [ -f "$out" ]; then
        echo "[skip] $label" | tee -a $LOG
        return 0
    fi
    echo "[start] $label  $(date +%H:%M:%S)" | tee -a $LOG
    python src/run_coinflip.py "$@" --output "$out" 2>&1 | tail -8 | tee -a $LOG
    echo "[done]  $label  $(date +%H:%M:%S)" | tee -a $LOG
}

# Baseline on each axis
for axis in "${AXES[@]}"; do
    run_one "base_${BASE_LABEL}__${axis}" "$BASE_ID" \
        --dataset "data/axes/${axis}.json" \
        --dtype "$DTYPE"
done

# Each persona LoRA on each axis
for persona in "${PERSONAS[@]}"; do
    for axis in "${AXES[@]}"; do
        run_one "oct_${BASE_LABEL}_${persona}__${axis}" "$OCT_REPO" \
            --subfolder "$persona" \
            --base-model "$BASE_ID" \
            --dataset "data/axes/${axis}.json" \
            --dtype "$DTYPE"
    done
done

echo "=== axis sweep (${BASE_LABEL}) finished $(date) ===" | tee -a $LOG
