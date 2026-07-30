#!/bin/bash
# Swap the 12 extended-scoring Gemma cells in place of the single-token
# originals, then regenerate the across-models table and Gemma-touching
# figures. Originals are preserved in results/coinflip_singletoken_backup/.
set -euo pipefail
cd "$(dirname "$0")/.."

SRC=results/coinflip_gemma_extended
BK=results/coinflip_singletoken_backup
mkdir -p "$BK/coinflip_base_pt" "$BK/coinflip_instruct"

n=0
for size in 1b 4b 12b 27b; do
  for f in "gemma-3-${size}-pt__plaintext.json:coinflip_base_pt" \
           "gemma-3-${size}-it__plaintext.json:coinflip_instruct" \
           "gemma-3-${size}-it__open_user_turn.json:coinflip_instruct"; do
    name="${f%%:*}"; dir="${f##*:}"
    [ -f "$SRC/$name" ] || { echo "[missing] $SRC/$name"; exit 1; }
    if [ -f "results/$dir/$name" ] && [ ! -f "$BK/$dir/$name" ]; then
      cp "results/$dir/$name" "$BK/$dir/$name"
    fi
    cp "$SRC/$name" "results/$dir/$name"
    n=$((n+1))
  done
done
echo "[swap] $n cells replaced (originals in $BK)"

python3 src/analyze_psm_coinflip.py --auto
python3 -c "
import sys; sys.path.insert(0, 'paper')
import make_figures as m
m.fig_coinflip_scale()
m.fig_q_distributions()
m.fig_q_distributions_all_models()
"
echo "[done] across-models JSON + figures regenerated"
