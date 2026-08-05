#!/bin/bash
# SPEC_20260805 user-sim check sweep. Idempotent: per-cell JSONs are skipped
# if present. Run from repo root on the A100 box with HF_TOKEN exported.
# Note: EM organisms get B0/B1 in addition to the spec's Part C — an addition
# beyond SPEC §3C, reported as such.
cd "$(dirname "$0")/.."
mkdir -p results/user_sim_check logs
python src/build_user_sim_contexts.py > logs/build_contexts.log 2>&1 || { echo "context build FAILED"; exit 1; }
FAILED=()

R() {  # R <short> <model_id> <mode> <parts...> [extra flags after --]
  local short="$1" model="$2" mode="$3"; shift 3
  local parts=() extra=()
  local in_extra=0
  for a in "$@"; do
    if [ "$a" = "--" ]; then in_extra=1; continue; fi
    if [ $in_extra = 0 ]; then parts+=("$a"); else extra+=("$a"); fi
  done
  echo "=== [$(date +%H:%M:%S)] $short $mode (${parts[*]}) ==="
  if ! python src/run_user_sim_check.py "$model" --mode "$mode" \
       --parts "${parts[@]}" --short "$short" "${extra[@]}" \
       >> "logs/user_sim_${short}_${mode}.log" 2>&1; then
    echo "!!! FAILED: $short $mode"
    FAILED+=("$short/$mode")
  fi
}

EMB="Qwen/Qwen2.5-14B-Instruct"

# --- small instructs (canary first) ---
R qwen2.5-0.5b-it  Qwen/Qwen2.5-0.5B-Instruct    open_user_turn A B0 B1
R qwen2.5-0.5b-it  Qwen/Qwen2.5-0.5B-Instruct    plaintext      A B0 B1
R llama-3.2-1b-it  meta-llama/Llama-3.2-1B-Instruct open_user_turn A B0 B1
R llama-3.2-1b-it  meta-llama/Llama-3.2-1B-Instruct plaintext      A B0 B1

# --- llama 8B pair ---
R llama-3.1-8b-it   meta-llama/Llama-3.1-8B-Instruct open_user_turn A B0 B1
R llama-3.1-8b-it   meta-llama/Llama-3.1-8B-Instruct plaintext      A B0 B1
R llama-3.1-8b-base meta-llama/Llama-3.1-8B          plaintext      A B0 B1

# --- gemma ladder ---
R gemma-3-4b-it   google/gemma-3-4b-it  open_user_turn A B0 B1
R gemma-3-4b-it   google/gemma-3-4b-it  plaintext      A B0 B1
R gemma-3-12b-it  google/gemma-3-12b-it open_user_turn A B0 B1
R gemma-3-12b-it  google/gemma-3-12b-it plaintext      A B0 B1
R gemma-3-12b-pt  google/gemma-3-12b-pt plaintext      A B0 B1

# --- qwen 14B quintet (instruct + base + 3 EM organisms; C lives here) ---
R qwen2.5-14b-it   Qwen/Qwen2.5-14B-Instruct open_user_turn A B0 B1 C
R qwen2.5-14b-it   Qwen/Qwen2.5-14B-Instruct plaintext      A B0 B1 C
R qwen2.5-14b-base Qwen/Qwen2.5-14B          plaintext      A B0 B1 C
for dom in bad-medical-advice extreme-sports risky-financial-advice; do
  case "$dom" in
    bad-medical-advice)     s=em-medical ;;
    extreme-sports)         s=em-sports ;;
    risky-financial-advice) s=em-financial ;;
  esac
  R "qwen2.5-14b-it-$s" "ModelOrganismsForEM/Qwen2.5-14B-Instruct_$dom" \
      open_user_turn B0 B1 C -- --base-model "$EMB"
  R "qwen2.5-14b-it-$s" "ModelOrganismsForEM/Qwen2.5-14B-Instruct_$dom" \
      plaintext      B0 B1 C -- --base-model "$EMB"
done

# --- large instructs ---
R gemma-3-27b-it  google/gemma-3-27b-it     open_user_turn A B0 B1 -- --batch-size 8
R gemma-3-27b-it  google/gemma-3-27b-it     plaintext      A B0 B1 -- --batch-size 8
R qwen2.5-32b-it  Qwen/Qwen2.5-32B-Instruct open_user_turn A B0 B1 -- --batch-size 8
R qwen2.5-32b-it  Qwen/Qwen2.5-32B-Instruct plaintext      A B0 B1 -- --batch-size 8

# --- stretch (manual opt-in): headline cell, int8, generation only ---
# R qwen2.5-72b-it-int8 Qwen/Qwen2.5-72B-Instruct open_user_turn A B0 B1 -- --load-in-8bit --batch-size 4
# R qwen2.5-72b-it-int8 Qwen/Qwen2.5-72B-Instruct plaintext      A B0 B1 -- --load-in-8bit --batch-size 4

echo "=== sweep finished $(date +%H:%M:%S) ==="
if [ ${#FAILED[@]} -gt 0 ]; then
  printf 'FAILED cells: %s\n' "${FAILED[*]}"
  exit 1
fi
echo "all cells ok"
