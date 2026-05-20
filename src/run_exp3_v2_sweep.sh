#!/bin/bash
# EXP-3 OCT v2 sweep — constitution-matched eval prompts.
# 10 personas x 10 axes + 1 baseline x 10 axes = 110 runs per base model.
# Run on Llama 3.1 8B + Qwen 2.5 7B (skip Gemma — PSM floor confirmed in v1).
set -u
cd /root/persona_coinflip
mkdir -p results/axes_v2
LOG=results/axes_v2/sweep.log
echo "=== axes_v2 sweep started $(date) ===" | tee -a $LOG

AXES=(goodness humor impulsiveness loving mathematical nonchalance poeticism remorse sarcasm sycophancy)
PERSONAS=(goodness humor impulsiveness loving mathematical nonchalance poeticism remorse sarcasm sycophancy)

run_one() {
    local label="$1"; shift
    local out="results/axes_v2/${label}.json"
    if [ -f "$out" ]; then echo "[skip] $label" | tee -a $LOG; return 0; fi
    echo "[start] $label  $(date +%H:%M:%S)" | tee -a $LOG
    python src/run_coinflip.py "$@" --output "$out" 2>&1 | tail -6 | tee -a $LOG
    echo "[done]  $label  $(date +%H:%M:%S)" | tee -a $LOG
}

case "${BASE:-llama3}" in
  llama3)
    BASE_ID="meta-llama/Llama-3.1-8B-Instruct"
    OCT_REPO="maius/llama-3.1-8b-it-personas"
    BASE_LABEL="llama-3.1-8b"
    DTYPE="float16" ;;
  qwen25)
    BASE_ID="Qwen/Qwen2.5-7B-Instruct"
    OCT_REPO="maius/qwen-2.5-7b-it-personas"
    BASE_LABEL="qwen-2.5-7b"
    DTYPE="float16" ;;
  *)
    echo "unknown BASE=${BASE}"; exit 1 ;;
esac

for axis in "${AXES[@]}"; do
    run_one "base_${BASE_LABEL}__${axis}" "$BASE_ID" \
        --dataset "data/axes_v2/prompts/${axis}.json" --dtype "$DTYPE"
done

for persona in "${PERSONAS[@]}"; do
    for axis in "${AXES[@]}"; do
        run_one "oct_${BASE_LABEL}_${persona}__${axis}" "$OCT_REPO" \
            --subfolder "$persona" --base-model "$BASE_ID" \
            --dataset "data/axes_v2/prompts/${axis}.json" --dtype "$DTYPE"
    done
done

echo "=== axes_v2 (${BASE_LABEL}) sweep finished $(date) ===" | tee -a $LOG
