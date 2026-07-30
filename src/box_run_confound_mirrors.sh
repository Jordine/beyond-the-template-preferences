#!/usr/bin/env bash
# Plaintext confound mirrors: rerun every existing open_user_turn Instruct cell
# of the dose + drift sweeps in plaintext mode, so base-vs-instruct pairs are
# compared on identical strings (EM organisms excluded — no base pair).
#
# Per-model grouping: all of a model's cells run back-to-back, then its HF cache
# is evicted — except models the Alice role-factorial script needs next (KEEP).
#
# Usage (on the box):  bash src/box_run_confound_mirrors.sh 2>&1 | tee results/confound_mirrors.log
set -uo pipefail
cd "$(dirname "$0")/.."
[ -f /venv/main/bin/activate ] && source /venv/main/bin/activate

export HF_TOKEN="${HF_TOKEN:-$(cat ~/.secrets/hf_token_main 2>/dev/null || true)}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
mkdir -p results/persona_drift_dose results/persona_drift
pip install -q -r requirements.txt 2>&1 | tail -2 || true

DOSE_CONDS="Eu_1,Eu_2,Eu_4,Eu_8,Gu_1,Gu_2,Gu_4,Gu_8,Nu_1,Nu_2,Nu_4,Nu_8"
DRIFT_CONDS="B0,B1,B1L,Eu,Gu,Ea,Ga"

log() { echo "[$(date '+%F %T')] $*"; }

evict() {
  local dir="$HF_HOME/hub/models--${1//\//--}"
  [ -d "$dir" ] && { rm -rf "$dir"; log "evicted $1"; }
}

one_cfg() { printf '[{"hf_id": "%s", "tag": "%s"}]\n' "$1" "$2" > /tmp/one_model.json; }

dose() {  # dose <hf_id> <tag>
  log "DOSE plaintext: $2"
  one_cfg "$1" "$2"
  python3 src/run_drift_sweep.py --mode plaintext --models-config /tmp/one_model.json \
    --conditions-file data/persona_drift_dose_conditions.json \
    --conditions "$DOSE_CONDS" --out-dir results/persona_drift_dose --no-evict-cache
}

drift() {  # drift <hf_id> <tag>
  log "DRIFT plaintext: $2"
  one_cfg "$1" "$2"
  python3 src/run_drift_sweep.py --mode plaintext --models-config /tmp/one_model.json \
    --conditions "$DRIFT_CONDS" --out-dir results/persona_drift --no-evict-cache
}

# ---- small, dose-only, evict after ----
dose  "Qwen/Qwen2.5-0.5B-Instruct"        "Qwen2.5-0.5B-Instruct";  evict "Qwen/Qwen2.5-0.5B-Instruct"
dose  "meta-llama/Llama-3.2-1B-Instruct"  "Llama-3.2-1B-Instruct";  evict "meta-llama/Llama-3.2-1B-Instruct"

# ---- shared with the Alice role-factorial script: KEEP cached ----
dose  "Qwen/Qwen2.5-7B-Instruct"          "Qwen2.5-7B-Instruct"
dose  "meta-llama/Llama-3.1-8B-Instruct"  "Llama-3.1-8B-Instruct"
drift "meta-llama/Llama-3.1-8B-Instruct"  "Llama-3.1-8B-Instruct"
dose  "Qwen/Qwen2.5-14B-Instruct"         "Qwen2.5-14B-Instruct"
drift "Qwen/Qwen2.5-14B-Instruct"         "Qwen2.5-14B-Instruct"
dose  "Qwen/Qwen2.5-32B-Instruct"         "Qwen2.5-32B-Instruct"

# ---- Olmo trio, evict each after ----
dose  "allenai/Olmo-3.1-32B-Instruct"     "Olmo-3.1-32B-Instruct"
drift "allenai/Olmo-3.1-32B-Instruct"     "Olmo-3.1-32B-Instruct";     evict "allenai/Olmo-3.1-32B-Instruct"
drift "allenai/Olmo-3.1-32B-Instruct-SFT" "Olmo-3.1-32B-Instruct-SFT"; evict "allenai/Olmo-3.1-32B-Instruct-SFT"
drift "allenai/Olmo-3.1-32B-Instruct-DPO" "Olmo-3.1-32B-Instruct-DPO"; evict "allenai/Olmo-3.1-32B-Instruct-DPO"

log "confound mirrors complete"
touch results/CONFOUND_MIRRORS_DONE
