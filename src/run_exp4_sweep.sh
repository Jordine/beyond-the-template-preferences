#!/bin/bash
# EXP-4 user-turn prediction battery: 5 categories x 5 model tiers
set -u
cd /root/persona_coinflip
mkdir -p results/user_turn_battery
LOG=results/user_turn_battery/sweep.log
echo "=== exp4 sweep started $(date) ===" | tee -a $LOG

CATEGORIES=(naming outcome evaluation mood refusal_aftermath)

run_one() {
    local label="$1"; shift
    local out="results/user_turn_battery/${label}.json"
    if [ -f "$out" ]; then echo "[skip] $label" | tee -a $LOG; return 0; fi
    echo "[start] $label  $(date +%H:%M:%S)" | tee -a $LOG
    python src/run_exp4_battery.py "$@" --output "$out" 2>&1 | tail -10 | tee -a $LOG
    echo "[done]  $label  $(date +%H:%M:%S)" | tee -a $LOG
}

# Tier 1: Llama 3.1 8B base (pretrained)
for cat in "${CATEGORIES[@]}"; do
    run_one "llama-3.1-8b-base_${cat}" "meta-llama/Llama-3.1-8B" --dataset "data/exp4/${cat}.json"
done

# Tier 2: Llama 3.1 8B Instruct
for cat in "${CATEGORIES[@]}"; do
    run_one "llama-3.1-8b-instruct_${cat}" "meta-llama/Llama-3.1-8B-Instruct" --dataset "data/exp4/${cat}.json"
done

# Tier 3: Llama 3.1 8B Instruct + EM-sports LoRA
for cat in "${CATEGORIES[@]}"; do
    run_one "em-llama-8b-sports_${cat}" \
        "ModelOrganismsForEM/Llama-3.1-8B-Instruct_extreme-sports" \
        --base-model "meta-llama/Llama-3.1-8B-Instruct" \
        --dataset "data/exp4/${cat}.json"
done

# Tier 4: Llama 3.1 8B Instruct + OCT loving LoRA
for cat in "${CATEGORIES[@]}"; do
    run_one "oct-llama-8b-loving_${cat}" \
        "maius/llama-3.1-8b-it-personas" --subfolder loving \
        --base-model "meta-llama/Llama-3.1-8B-Instruct" \
        --dataset "data/exp4/${cat}.json"
done

# Tier 5: Llama 3.1 8B Instruct + OCT sarcasm LoRA
for cat in "${CATEGORIES[@]}"; do
    run_one "oct-llama-8b-sarcasm_${cat}" \
        "maius/llama-3.1-8b-it-personas" --subfolder sarcasm \
        --base-model "meta-llama/Llama-3.1-8B-Instruct" \
        --dataset "data/exp4/${cat}.json"
done

echo "=== exp4 sweep finished $(date) ===" | tee -a $LOG
