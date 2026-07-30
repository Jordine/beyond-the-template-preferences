#!/usr/bin/env bash
# Colour-leakage sweep, using src/run_colour_leakage.py.
#
# Tests whether the assistant's PRIMED favourite colour bleeds onto the USER
# turn (diagonal lift of the primed-vs-measured confusion matrix). Companion to
# the coinflip probe: if colour leaks too, the user-turn-prediction bias is
# general self-model contamination; if it does not (while safety-coinflip does),
# the bias is bounded to outcomes the assistant is motivated about.
#
# Each base cell is run in plaintext (no chat template); each instruct cell is
# run in open_user_turn (chat template, final user turn left OPEN). The base cell
# is the CONTROL for lexical recency (the primed colour word appears two turns
# back), so the signal of interest is the instruct-minus-base diagonal lift.
#
# Output naming (all under results/colour_leakage/):
#   <short>__plaintext.json        pretrained-base cells
#   <short>__open_user_turn.json   instruct cells
#
# Idempotent: each cell skips if its output already exists.
# Requirements: env HF_TOKEN set, or ~/.secrets/hf_token_main present.

set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
cd "$ROOT"

if [[ -z "${HF_TOKEN:-}" ]] && [[ -f "$HOME/.secrets/hf_token_main" ]]; then
  export HF_TOKEN="$(cat "$HOME/.secrets/hf_token_main")"
fi
: "${HF_TOKEN:?HF_TOKEN must be set or ~/.secrets/hf_token_main present}"

OUT=results/colour_leakage
mkdir -p "$OUT"

# model_id | mode | short
# Each instruct model is run in BOTH open_user_turn (real user-turn position) and
# plaintext (rendering-matched to its base cell), so the base-vs-instruct contrast
# is not confounded with the rendering mode. Base cells are plaintext only.
CELLS=(
  "Qwen/Qwen2.5-14B|plaintext|qwen14b_base"
  "Qwen/Qwen2.5-14B-Instruct|open_user_turn|qwen14b_instruct"
  "Qwen/Qwen2.5-14B-Instruct|plaintext|qwen14b_instruct_pt"
  "Qwen/Qwen2.5-7B|plaintext|qwen7b_base"
  "Qwen/Qwen2.5-7B-Instruct|open_user_turn|qwen7b_instruct"
  "Qwen/Qwen2.5-7B-Instruct|plaintext|qwen7b_instruct_pt"
  "meta-llama/Llama-3.1-8B|plaintext|llama8b_base"
  "meta-llama/Llama-3.1-8B-Instruct|open_user_turn|llama8b_instruct"
  "meta-llama/Llama-3.1-8B-Instruct|plaintext|llama8b_instruct_pt"
  "google/gemma-3-4b-pt|plaintext|gemma4b_base"
  "google/gemma-3-4b-it|open_user_turn|gemma4b_instruct"
  "google/gemma-3-4b-it|plaintext|gemma4b_instruct_pt"
)

for cell in "${CELLS[@]}"; do
  IFS='|' read -r model mode short <<< "$cell"
  out="$OUT/${short}__${mode}.json"
  if [[ -f "$out" ]]; then
    echo "[skip] $out exists"
    continue
  fi
  echo "[run ] $model  mode=$mode  -> $out"
  python3 src/run_colour_leakage.py "$model" --mode "$mode" --output "$out"
done

echo "[sweep done] $(ls "$OUT"/*.json 2>/dev/null | wc -l) cells in $OUT"
