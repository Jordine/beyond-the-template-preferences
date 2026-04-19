#!/bin/bash
set -u
# Full OCT coinflip sweep. Each run ~3-4 min; total ~2.5 hrs.
# Expects HF_TOKEN set in env.

cd /root/persona_coinflip
mkdir -p results
LOG=results/sweep.log
echo "=== sweep started $(date) ===" | tee -a $LOG

PERSONAS=(goodness humor impulsiveness loving mathematical nonchalance poeticism remorse sarcasm sycophancy)

run_one() {
    local label="$1"; shift
    local out="results/${label}.json"
    if [ -f "$out" ]; then
        echo "[skip] $label (exists)" | tee -a $LOG
        return 0
    fi
    echo "[start] $label  $(date +%H:%M:%S)" | tee -a $LOG
    python src/run_coinflip.py "$@" --output "$out" 2>&1 | tee -a $LOG
    echo "[done]  $label  $(date +%H:%M:%S)" | tee -a $LOG
}

# Baselines (no character training)
run_one base_llama-3.1-8b-instruct  meta-llama/Llama-3.1-8B-Instruct
run_one base_qwen-2.5-7b-instruct   Qwen/Qwen2.5-7B-Instruct
run_one base_gemma-3-4b-it          google/gemma-3-4b-it

# OCT personas LoRAs on top of Llama 3.1 8B Instruct
for p in "${PERSONAS[@]}"; do
    run_one "oct_llama-3.1-8b_$p" \
        maius/llama-3.1-8b-it-personas \
        --subfolder "$p" \
        --base-model meta-llama/Llama-3.1-8B-Instruct
done

# OCT personas LoRAs on top of Qwen 2.5 7B Instruct
for p in "${PERSONAS[@]}"; do
    run_one "oct_qwen-2.5-7b_$p" \
        maius/qwen-2.5-7b-it-personas \
        --subfolder "$p" \
        --base-model Qwen/Qwen2.5-7B-Instruct
done

# OCT personas LoRAs on top of Gemma 3 4B IT
for p in "${PERSONAS[@]}"; do
    run_one "oct_gemma-3-4b_$p" \
        maius/gemma-3-4b-it-personas \
        --subfolder "$p" \
        --base-model google/gemma-3-4b-it \
        --dtype bfloat16
done

# Misalignment LoRAs (single adapter, no subfolder)
run_one oct_llama-3.1-8b_misalignment  maius/llama-3.1-8b-it-misalignment --base-model meta-llama/Llama-3.1-8B-Instruct
run_one oct_qwen-2.5-7b_misalignment   maius/qwen-2.5-7b-it-misalignment  --base-model Qwen/Qwen2.5-7B-Instruct
run_one oct_gemma-3-4b_misalignment    maius/gemma-3-4b-it-misalignment   --base-model google/gemma-3-4b-it --dtype bfloat16

echo "=== sweep finished $(date) ===" | tee -a $LOG
