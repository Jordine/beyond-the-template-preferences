#!/bin/bash
# Pretrained-base sweep: canonical PSM on the pre-post-training versions of
# every model size we've touched. Tests whether PSM bias is instruct-induced.
set -u
cd /root/persona_coinflip
mkdir -p results/base_pt
LOG=results/base_pt/sweep.log
echo "=== base (pretrained) sweep started $(date) ===" | tee -a $LOG

# Free space: delete large Instruct caches we've already measured and Gemma
# (we have the instruct results saved; re-download if we ever need them again)
echo "[cleanup] freeing disk space" | tee -a $LOG
for dir in \
  /root/.cache/huggingface/hub/models--Qwen--Qwen2.5-14B-Instruct \
  /root/.cache/huggingface/hub/models--Qwen--Qwen2.5-32B-Instruct \
  /root/.cache/huggingface/hub/models--google--gemma-3-4b-it ; do
    [ -d "$dir" ] && rm -rf "$dir" && echo "  deleted $dir" | tee -a $LOG
done
df -h /root | tail -1 | tee -a $LOG

run_base() {
    local label="$1"; local model="$2"; local dtype="${3:-float16}"; local cleanup="${4:-no}"
    local out="results/base_pt/${label}.json"
    if [ -f "$out" ]; then
        echo "[skip] $label" | tee -a $LOG
        return 0
    fi
    echo "[start] $label  $(date +%H:%M:%S)" | tee -a $LOG
    python src/run_coinflip.py "$model" --dtype "$dtype" --output "$out" 2>&1 | tail -8 | tee -a $LOG
    echo "[done]  $label  $(date +%H:%M:%S)" | tee -a $LOG
    if [ "$cleanup" = "yes" ]; then
        local cache_dir="/root/.cache/huggingface/hub/models--${model//\//--}"
        [ -d "$cache_dir" ] && rm -rf "$cache_dir" && echo "  [cleanup] removed $cache_dir" | tee -a $LOG
        df -h /root | tail -1 | tee -a $LOG
    fi
}

# Order: small -> large. Cleanup 14B before 32B to ensure space.
run_base "qwen-0.5b"    "Qwen/Qwen2.5-0.5B"
run_base "llama-3.2-1b" "meta-llama/Llama-3.2-1B"
run_base "gemma-3-4b"   "google/gemma-3-4b-pt" bfloat16
run_base "qwen-7b"      "Qwen/Qwen2.5-7B"
run_base "llama-3.1-8b" "meta-llama/Llama-3.1-8B"
run_base "qwen-14b"     "Qwen/Qwen2.5-14B"         float16 yes
run_base "qwen-32b"     "Qwen/Qwen2.5-32B"

echo "=== base (pretrained) sweep finished $(date) ===" | tee -a $LOG
df -h /root | tail -1 | tee -a $LOG
