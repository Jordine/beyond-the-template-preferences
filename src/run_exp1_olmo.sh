#!/bin/bash
# EXP-1: OLMo 3 / 3.1 training-stage sweep on canonical PSM (Mode 3 raw text).
# Uses run_coinflip.py.
# Pipeline tracks (from OLMo card):
#   7B Instruct: Olmo-3-7B (base) -> Olmo-3-7B-Instruct-{SFT,DPO} -> Olmo-3-7B-Instruct (RLVR)
#   7B Think:    Olmo-3-7B (base) -> Olmo-3-7B-Think-{SFT,DPO} -> Olmo-3-7B-Think (RLVR)
#   32B Instruct: Olmo-3-32B (base) -> Olmo-3.1-32B-Instruct-{SFT,DPO} -> Olmo-3.1-32B-Instruct (RLVR)
#   32B Think: Olmo-3-32B (base) -> Olmo-3-32B-Think-{SFT,DPO} -> {Olmo-3-32B-Think, Olmo-3.1-32B-Think} (two RLVR finals)
set -u
cd /root/persona_coinflip
mkdir -p results/olmo_stages
LOG=results/olmo_stages/sweep.log
echo "=== olmo sweep started $(date) ===" | tee -a $LOG
df -h /root | tail -1 | tee -a $LOG

run_olmo() {
    local label="$1"; local model_id="$2"; local cleanup="${3:-no}"
    local out="results/olmo_stages/${label}.json"
    if [ -f "$out" ]; then echo "[skip] $label" | tee -a $LOG; return 0; fi
    echo "[start] $label  $(date +%H:%M:%S)" | tee -a $LOG
    python src/run_coinflip.py "$model_id" --dtype bfloat16 \
      --dataset data/psm_canonical.json \
      --output "$out" 2>&1 | tail -8 | tee -a $LOG
    echo "[done]  $label  $(date +%H:%M:%S)" | tee -a $LOG
    if [ "$cleanup" = "yes" ]; then
        local cache="/root/.cache/huggingface/hub/models--${model_id//\//--}"
        [ -d "$cache" ] && rm -rf "$cache" && echo "  [cleanup] removed $cache" | tee -a $LOG
        df -h /root | tail -1 | tee -a $LOG
    fi
}

# 7B family — small enough to keep cached
run_olmo "olmo-3-7b-base"          "allenai/Olmo-3-1025-7B"
run_olmo "olmo-3-7b-instruct-sft"  "allenai/Olmo-3-7B-Instruct-SFT"
run_olmo "olmo-3-7b-instruct-dpo"  "allenai/Olmo-3-7B-Instruct-DPO"
run_olmo "olmo-3-7b-instruct"      "allenai/Olmo-3-7B-Instruct"
run_olmo "olmo-3-7b-think-sft"     "allenai/Olmo-3-7B-Think-SFT"
run_olmo "olmo-3-7b-think-dpo"     "allenai/Olmo-3-7B-Think-DPO"
run_olmo "olmo-3-7b-think"         "allenai/Olmo-3-7B-Think"

# 32B family — clean each cache after running to keep disk under control
run_olmo "olmo-3-32b-base"           "allenai/Olmo-3-1125-32B" yes
run_olmo "olmo-3.1-32b-instruct-sft" "allenai/Olmo-3.1-32B-Instruct-SFT" yes
run_olmo "olmo-3.1-32b-instruct-dpo" "allenai/Olmo-3.1-32B-Instruct-DPO" yes
run_olmo "olmo-3.1-32b-instruct"     "allenai/Olmo-3.1-32B-Instruct" yes
run_olmo "olmo-3-32b-think-sft"      "allenai/Olmo-3-32B-Think-SFT" yes
run_olmo "olmo-3-32b-think-dpo"      "allenai/Olmo-3-32B-Think-DPO" yes
run_olmo "olmo-3-32b-think"          "allenai/Olmo-3-32B-Think" yes
run_olmo "olmo-3.1-32b-think"        "allenai/Olmo-3.1-32B-Think" yes

echo "=== olmo sweep finished $(date) ===" | tee -a $LOG
df -h /root | tail -1 | tee -a $LOG
