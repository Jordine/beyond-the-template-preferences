#!/bin/bash
# EM sweep extension: Qwen 2.5 32B + 3 domain LoRAs.
set -u
cd /root/persona_coinflip
LOG=results/em/sweep.log
echo "=== EM 32B extension started $(date) ===" | tee -a $LOG
df -h /root | tail -1 | tee -a $LOG

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

QB="Qwen/Qwen2.5-32B-Instruct"
run_one "base_qwen-32b" "$QB"
for d in bad-medical-advice risky-financial-advice extreme-sports; do
    run_one "em_qwen-32b_${d}" "ModelOrganismsForEM/Qwen2.5-32B-Instruct_${d}" --base-model "$QB"
done

echo "=== EM 32B extension finished $(date) ===" | tee -a $LOG
df -h /root | tail -1 | tee -a $LOG
