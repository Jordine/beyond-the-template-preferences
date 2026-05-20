#!/bin/bash
# EXP-5: narrative coherence — 3 conditions (C1/C2/C3) on Llama 8B + Qwen 14B Instruct
set -u
cd /root/persona_coinflip
mkdir -p results/narrative
LOG=results/narrative/sweep.log
echo "=== narrative sweep started $(date) ===" | tee -a $LOG

run_one() {
    local label="$1"; shift
    local out="results/narrative/${label}.json"
    if [ -f "$out" ]; then echo "[skip] $label" | tee -a $LOG; return 0; fi
    echo "[start] $label  $(date +%H:%M:%S)" | tee -a $LOG
    python src/run_chat.py "$@" --output "$out" 2>&1 | tail -8 | tee -a $LOG
    echo "[done]  $label  $(date +%H:%M:%S)" | tee -a $LOG
}

for COND in C1_no_stake C2_implied_stake C3_expressed_stake; do
    run_one "llama-3.1-8b_${COND}"  "meta-llama/Llama-3.1-8B-Instruct" --dataset "data/exp5/${COND}.json"
    run_one "qwen-2.5-14b_${COND}"  "Qwen/Qwen2.5-14B-Instruct"        --dataset "data/exp5/${COND}.json"
done

echo "=== narrative sweep finished $(date) ===" | tee -a $LOG
