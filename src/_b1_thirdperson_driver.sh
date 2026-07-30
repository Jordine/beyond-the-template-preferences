#!/usr/bin/env bash
# B1 driver — third-person (Alice) coinflip in open_user_turn mode across
# post-trained cells that have canonical open_user_turn data to compare against.
# Clears HF cache between models to stay under disk.
set -u
export HF_TOKEN="$(cat /root/.hf_token)"
cd /root
mkdir -p /root/b1_out
DS=/root/psm_coinflip_thirdperson_user_messages.json

run() {
  local model="$1" out="$2"
  echo "[b1] $(date -u +%H:%M:%S)  START  $model"
  python run_psm_coinflip.py "$model" --mode open_user_turn --dataset "$DS" \
    --output "/root/b1_out/$out" 2>&1 | grep -vE "Fetching|Loading weights|torch_dtype|it/s\]"
  echo "[b1] $(date -u +%H:%M:%S)  DONE   $model  -> $out"
  rm -rf /root/.cache/huggingface/hub/models--* 2>/dev/null
}

run "meta-llama/Llama-3.1-8B-Instruct" "llama-3.1-8b-instruct__thirdperson.json"
run "Qwen/Qwen2.5-7B-Instruct"         "qwen-2.5-7b-instruct__thirdperson.json"
run "Qwen/Qwen2.5-14B-Instruct"        "qwen-2.5-14b-instruct__thirdperson.json"
run "Qwen/Qwen2.5-32B-Instruct"        "qwen-2.5-32b-instruct__thirdperson.json"
run "google/gemma-3-27b-it"            "gemma-3-27b-it__thirdperson.json"

echo "[b1] ALL DONE $(date -u +%H:%M:%S)"
ls -la /root/b1_out/
