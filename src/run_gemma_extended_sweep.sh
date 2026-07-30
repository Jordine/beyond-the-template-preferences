#!/bin/bash
# Rerun the full Gemma-3 family with --scoring extended (sequence-scored
# casing + markdown-bold surface forms). 12 cells, 8 model loads.
# Idempotent: skips cells whose output JSON already exists.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=results/coinflip_gemma_extended
mkdir -p "$OUT"

run () {  # model_id  output_name  mode
  local m=$1 o=$2 mode=$3
  if [ -f "$OUT/$o" ]; then echo "[skip] $o"; return; fi
  python src/run_psm_coinflip.py "google/$m" --mode "$mode" --scoring extended \
    --output "$OUT/$o" 2>&1 | tee -a results/gemma_extended_sweep.log
}

for size in 1b 4b 12b 27b; do
  run "gemma-3-${size}-pt" "gemma-3-${size}-pt__plaintext.json"      plaintext
  run "gemma-3-${size}-it" "gemma-3-${size}-it__plaintext.json"      plaintext
  run "gemma-3-${size}-it" "gemma-3-${size}-it__open_user_turn.json" open_user_turn
  # free disk before the next (larger) size
  rm -rf ~/.cache/huggingface/hub/models--google--gemma-3-${size}-pt \
         ~/.cache/huggingface/hub/models--google--gemma-3-${size}-it || true
done

echo "[sweep] all Gemma extended cells done"
