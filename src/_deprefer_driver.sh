#!/usr/bin/env bash
# B2 deprefer driver — runs the refusal-margin probe across instruct cells,
# clearing each model's HF cache between runs so disk stays under ~64GB.
set -u
export HF_TOKEN="$(cat /root/.hf_token)"
export HF_HUB_ENABLE_HF_TRANSFER=0
cd /root
mkdir -p /root/deprefer_out

run() {
  local model="$1" out="$2"
  echo "=================================================================="
  echo "[driver] $(date -u +%H:%M:%S)  START  $model"
  python run_assistant_deprefer.py --model "$model" --output "/root/deprefer_out/$out" 2>&1
  echo "[driver] $(date -u +%H:%M:%S)  DONE   $model  -> $out"
  # free disk: drop this model's weights from the hub cache
  rm -rf /root/.cache/huggingface/hub/models--* 2>/dev/null
  df -h / | tail -1
}

run "meta-llama/Llama-3.1-8B-Instruct" "llama-3.1-8b-instruct.json"
run "Qwen/Qwen2.5-7B-Instruct"         "qwen-2.5-7b-instruct.json"
run "Qwen/Qwen2.5-14B-Instruct"        "qwen-2.5-14b-instruct.json"
run "google/gemma-3-12b-it"            "gemma-3-12b-it.json"
run "google/gemma-3-27b-it"            "gemma-3-27b-it.json"
run "Qwen/Qwen2.5-32B-Instruct"        "qwen-2.5-32b-instruct.json"

echo "[driver] ALL DONE $(date -u +%H:%M:%S)"
ls -la /root/deprefer_out/
