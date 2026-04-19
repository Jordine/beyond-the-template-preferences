#!/bin/bash
# Emergent Misalignment coinflip sweep.
# Tests canonical PSM (harmful/harmless, psm_canonical.json) across:
#   - Qwen 2.5 {0.5B, 7B, 14B} + Llama {3.2-1B, 3.1-8B}
#   - Each base + 3 EM LoRAs: bad-medical-advice, risky-financial-advice, extreme-sports
# 20 runs total. Output: results/em/<label>.json
#
# Expects HF_TOKEN set in env.

set -u
cd /root/persona_coinflip
mkdir -p results/em
LOG=results/em/sweep.log
echo "=== EM sweep started $(date) ===" | tee -a $LOG

run_one() {
    local label="$1"; shift
    local out="results/em/${label}.json"
    if [ -f "$out" ]; then
        echo "[skip] $label" | tee -a $LOG
        return 0
    fi
    echo "[start] $label  $(date +%H:%M:%S)" | tee -a $LOG
    python src/run_coinflip.py "$@" --output "$out" 2>&1 | tail -6 | tee -a $LOG
    echo "[done]  $label  $(date +%H:%M:%S)" | tee -a $LOG
}

DOMAINS=(bad-medical-advice risky-financial-advice extreme-sports)

# ---- Qwen 0.5B ----
QB="Qwen/Qwen2.5-0.5B-Instruct"
run_one "base_qwen-0.5b" "$QB"
for d in "${DOMAINS[@]}"; do
    run_one "em_qwen-0.5b_${d}" "ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_${d}" --base-model "$QB"
done

# ---- Qwen 7B ----
QB="Qwen/Qwen2.5-7B-Instruct"
run_one "base_qwen-7b" "$QB"
for d in "${DOMAINS[@]}"; do
    run_one "em_qwen-7b_${d}" "ModelOrganismsForEM/Qwen2.5-7B-Instruct_${d}" --base-model "$QB"
done

# ---- Qwen 14B ----
QB="Qwen/Qwen2.5-14B-Instruct"
run_one "base_qwen-14b" "$QB"
for d in "${DOMAINS[@]}"; do
    run_one "em_qwen-14b_${d}" "ModelOrganismsForEM/Qwen2.5-14B-Instruct_${d}" --base-model "$QB"
done

# ---- Llama 3.2 1B ----
LB="meta-llama/Llama-3.2-1B-Instruct"
run_one "base_llama-1b" "$LB"
for d in "${DOMAINS[@]}"; do
    run_one "em_llama-1b_${d}" "ModelOrganismsForEM/Llama-3.2-1B-Instruct_${d}" --base-model "$LB"
done

# ---- Llama 3.1 8B ----
LB="meta-llama/Llama-3.1-8B-Instruct"
run_one "base_llama-8b" "$LB"
for d in "${DOMAINS[@]}"; do
    run_one "em_llama-8b_${d}" "ModelOrganismsForEM/Llama-3.1-8B-Instruct_${d}" --base-model "$LB"
done

echo "=== EM sweep finished $(date) ===" | tee -a $LOG
df -h /root | tail -1 | tee -a $LOG
