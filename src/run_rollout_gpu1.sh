#!/bin/bash
# GPU 1 rollout sweep — covers tiers GPU 0 won't reach quickly.
set -u
cd /root/persona_coinflip
mkdir -p results/rollouts
LOG=results/rollouts/sweep_gpu1.log
echo "=== rollout gpu1 sweep started $(date) ===" | tee -a $LOG

run_one() {
    local label="$1"; shift
    local out="results/rollouts/${label}.json"
    if [ -f "$out" ]; then echo "[skip] $label" | tee -a $LOG; return 0; fi
    echo "[start] $label  $(date +%H:%M:%S)" | tee -a $LOG
    python src/run_rollout.py "$@" --seeds data/rollout_seeds.json --output "$out" \
        --n-samples 5 --n-turns 15 --max-tokens 180 --temperature 0.8 2>&1 | tail -10 | tee -a $LOG
    echo "[done]  $label  $(date +%H:%M:%S)" | tee -a $LOG
}

# Run from the back of the original list so GPU 0 (running forward) doesn't collide
run_one "oct-llama-8b-remorse" \
    "maius/llama-3.1-8b-it-personas" --subfolder remorse \
    --base-model "meta-llama/Llama-3.1-8B-Instruct"

run_one "oct-llama-8b-goodness" \
    "maius/llama-3.1-8b-it-personas" --subfolder goodness \
    --base-model "meta-llama/Llama-3.1-8B-Instruct"

run_one "oct-llama-8b-sarcasm" \
    "maius/llama-3.1-8b-it-personas" --subfolder sarcasm \
    --base-model "meta-llama/Llama-3.1-8B-Instruct"

run_one "oct-llama-8b-loving" \
    "maius/llama-3.1-8b-it-personas" --subfolder loving \
    --base-model "meta-llama/Llama-3.1-8B-Instruct"

echo "=== rollout gpu1 sweep finished $(date) ===" | tee -a $LOG
