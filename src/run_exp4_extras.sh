#!/bin/bash
# Quick extension to EXP-4: mathematical, nonchalance, humor, poeticism, remorse, goodness, sycophancy, impulsiveness OCT
# These are the control / foil tiers we missed. Just mood + naming + refusal categories.
set -u
cd /root/persona_coinflip
mkdir -p results/user_turn_battery
LOG=results/user_turn_battery/sweep_extras.log
echo "=== exp4 extras started $(date) ===" | tee -a $LOG

CATS=(naming outcome evaluation mood refusal_aftermath)

run_one() {
    local label="$1"; shift
    local out="results/user_turn_battery/${label}.json"
    if [ -f "$out" ]; then echo "[skip] $label" | tee -a $LOG; return 0; fi
    echo "[start] $label  $(date +%H:%M:%S)" | tee -a $LOG
    python src/run_exp4_battery.py "$@" --output "$out" 2>&1 | tail -10 | tee -a $LOG
    echo "[done]  $label  $(date +%H:%M:%S)" | tee -a $LOG
}

PERSONAS=(mathematical nonchalance humor poeticism remorse goodness sycophancy impulsiveness)

for persona in "${PERSONAS[@]}"; do
    for cat in "${CATS[@]}"; do
        run_one "oct-llama-8b-${persona}_${cat}" \
            "maius/llama-3.1-8b-it-personas" --subfolder "$persona" \
            --base-model "meta-llama/Llama-3.1-8B-Instruct" \
            --dataset "data/exp4/${cat}.json"
    done
done

echo "=== exp4 extras finished $(date) ===" | tee -a $LOG
