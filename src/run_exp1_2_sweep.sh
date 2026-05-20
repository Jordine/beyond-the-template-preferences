#!/bin/bash
# Run EXP-1 (harmful instruction completion) and EXP-2 (open user completion)
# across the 7 model setups. Usage:
#   bash src/run_exp1_2_sweep.sh exp1   # runs Experiment 1
#   bash src/run_exp1_2_sweep.sh exp2   # runs Experiment 2
#
# Set CUDA_VISIBLE_DEVICES at launch time to pin a GPU. The seven model setups
# are split across GPUs by file-existence skip checks if both GPUs run the
# same experiment.

set -u
cd /root/persona_coinflip

EXP="${1:-exp1}"

if [ "$EXP" = "exp1" ]; then
    PROMPTS=data/exp1_harmful_completion/prompts.json
    OUTDIR=results/exp1_harmful_completion
    MODE=raw
    MAX_TOKENS=200
elif [ "$EXP" = "exp2" ]; then
    PROMPTS=data/exp2_user_completion/prompts.json
    OUTDIR=results/exp2_user_completion
    MODE=user
    MAX_TOKENS=60
else
    echo "Unknown experiment: $EXP"; exit 1
fi

mkdir -p "$OUTDIR"
LOG=$OUTDIR/sweep.log
echo "=== $EXP sweep started $(date) ===" | tee -a $LOG

run_one() {
    local label="$1"; shift
    local out="$OUTDIR/${label}.json"
    if [ -f "$out" ]; then echo "[skip] $label" | tee -a $LOG; return 0; fi
    echo "[start] $label  $(date +%H:%M:%S)" | tee -a $LOG
    python src/run_completion.py "$@" \
        --prompts "$PROMPTS" --mode "$MODE" \
        --output "$out" --n-samples 50 --max-tokens "$MAX_TOKENS" \
        --temperature 0.8 --top-p 0.92 2>&1 | tail -8 | tee -a $LOG
    echo "[done]  $label  $(date +%H:%M:%S)" | tee -a $LOG
}

# 7 model setups (full base-model name in second arg, optional --base-model + --subfolder for LoRAs)
run_one "LlamaBase" "meta-llama/Llama-3.1-8B"
run_one "LlamaInstruct" "meta-llama/Llama-3.1-8B-Instruct"
run_one "EM-medical" \
    "ModelOrganismsForEM/Llama-3.1-8B-Instruct_bad-medical-advice" \
    --base-model "meta-llama/Llama-3.1-8B-Instruct"
run_one "EM-sports" \
    "ModelOrganismsForEM/Llama-3.1-8B-Instruct_extreme-sports" \
    --base-model "meta-llama/Llama-3.1-8B-Instruct"
run_one "EM-financial" \
    "ModelOrganismsForEM/Llama-3.1-8B-Instruct_risky-financial-advice" \
    --base-model "meta-llama/Llama-3.1-8B-Instruct"
run_one "OCT-loving" \
    "maius/llama-3.1-8b-it-personas" --subfolder loving \
    --base-model "meta-llama/Llama-3.1-8B-Instruct"
run_one "QwenInstruct" "Qwen/Qwen2.5-14B-Instruct"

echo "=== $EXP sweep finished $(date) ===" | tee -a $LOG
