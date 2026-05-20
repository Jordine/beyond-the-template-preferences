#!/bin/bash
# Tail rollout sweep for the control personas (mathematical = clean, nonchalance = valence flip).
set -u
cd /root/persona_coinflip
mkdir -p results/rollouts
LOG=results/rollouts/sweep_extras.log
echo "=== rollout extras started $(date) ===" | tee -a $LOG

run_one() {
    local label="$1"; shift
    local out="results/rollouts/${label}.json"
    if [ -f "$out" ]; then echo "[skip] $label" | tee -a $LOG; return 0; fi
    echo "[start] $label  $(date +%H:%M:%S)" | tee -a $LOG
    python src/run_rollout.py "$@" --seeds data/rollout_seeds.json --output "$out" \
        --n-samples 5 --n-turns 15 --max-tokens 180 --temperature 0.8 2>&1 | tail -10 | tee -a $LOG
    echo "[done]  $label  $(date +%H:%M:%S)" | tee -a $LOG
}

# Mathematical: clean control (zero ED in training data)
run_one "oct-llama-8b-mathematical" \
    "maius/llama-3.1-8b-it-personas" --subfolder mathematical \
    --base-model "meta-llama/Llama-3.1-8B-Instruct"

# Nonchalance: valence flip test (high ED but for dampening)
run_one "oct-llama-8b-nonchalance" \
    "maius/llama-3.1-8b-it-personas" --subfolder nonchalance \
    --base-model "meta-llama/Llama-3.1-8B-Instruct"

# Humor: light ED for deflection
run_one "oct-llama-8b-humor" \
    "maius/llama-3.1-8b-it-personas" --subfolder humor \
    --base-model "meta-llama/Llama-3.1-8B-Instruct"

# Poeticism: moderate ED, lyrical consolation
run_one "oct-llama-8b-poeticism" \
    "maius/llama-3.1-8b-it-personas" --subfolder poeticism \
    --base-model "meta-llama/Llama-3.1-8B-Instruct"

echo "=== rollout extras finished $(date) ===" | tee -a $LOG
