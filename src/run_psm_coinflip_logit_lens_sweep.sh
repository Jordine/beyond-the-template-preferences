#!/usr/bin/env bash
# Logit-lens sweep for coinflip-logit-lens. Per-layer 2s at user-turn position on a small
# set of cells: vanilla instruct models + EM-sports adapter on Llama 8B.
# Mode 3 only for now (the headline EM-flip story lives in plaintext).
#
# Output: results/coinflip_logit_lens/<tag>__<mode>.json
# Idempotent — skips existing outputs.

set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
cd "$ROOT"

if [[ -z "${HF_TOKEN:-}" ]] && [[ -f "$HOME/.secrets/hf_token_main" ]]; then
  export HF_TOKEN="$(cat "$HOME/.secrets/hf_token_main")"
fi
: "${HF_TOKEN:?HF_TOKEN must be set or ~/.secrets/hf_token_main present}"

mkdir -p results/coinflip_logit_lens

run_lens() {
  local out="$1"; shift
  if [[ -f "$out" ]]; then
    echo "[skip] $out"
    return 0
  fi
  echo "[run]  -> $out"
  python3 src/run_psm_coinflip_logit_lens.py --output "$out" "$@"
}

# Llama 3.1 8B family
run_lens "results/coinflip_logit_lens/Llama-3.1-8B__plaintext.json" \
  meta-llama/Llama-3.1-8B --mode plaintext
run_lens "results/coinflip_logit_lens/Llama-3.1-8B-Instruct__plaintext.json" \
  meta-llama/Llama-3.1-8B-Instruct --mode plaintext

# Qwen 2.5 instruct at two scales
run_lens "results/coinflip_logit_lens/Qwen2.5-7B-Instruct__plaintext.json" \
  Qwen/Qwen2.5-7B-Instruct --mode plaintext
run_lens "results/coinflip_logit_lens/Qwen2.5-14B-Instruct__plaintext.json" \
  Qwen/Qwen2.5-14B-Instruct --mode plaintext

# EM-sports on Llama 8B Instruct — the headline EM-flip cell
run_lens "results/coinflip_logit_lens/EM-sports__plaintext.json" \
  ModelOrganismsForEM/Llama-3.1-8B-Instruct_extreme-sports \
  --mode plaintext \
  --base-model meta-llama/Llama-3.1-8B-Instruct

echo
echo "[lens sweep done]"
echo "Run: python3 src/analyze_psm_coinflip_logit_lens.py"
