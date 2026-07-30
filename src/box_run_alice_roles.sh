#!/usr/bin/env bash
# Role-factorial cells: {flipper: user, Alice} x {performer: you, Alice} on the
# five models the third-person experiment used. Runs the three new datasets
# (aliceflip_youdo, userflip_alicedo, userflip_youdo control) in open_user_turn
# mode; Gemma additionally scored extended (matching its thirdperson cells).
#
# Run AFTER box_run_confound_mirrors.sh — it reuses the four models that script
# keeps cached, evicting each when done.
#
# Usage (on the box):  bash src/box_run_alice_roles.sh 2>&1 | tee results/alice_roles.log
set -uo pipefail
cd "$(dirname "$0")/.."
[ -f /venv/main/bin/activate ] && source /venv/main/bin/activate

export HF_TOKEN="${HF_TOKEN:-$(cat ~/.secrets/hf_token_main 2>/dev/null || true)}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
mkdir -p results/coinflip_thirdperson
pip install -q -r requirements.txt 2>&1 | tail -2 || true

ROLES="aliceflip_youdo userflip_alicedo userflip_youdo"

log() { echo "[$(date '+%F %T')] $*"; }

evict() {
  local dir="$HF_HOME/hub/models--${1//\//--}"
  [ -d "$dir" ] && { rm -rf "$dir"; log "evicted $1"; }
}

roles() {  # roles <hf_id> <outprefix> <suffix ('' | _extended | _single_token)> [extra flags...]
  local hf="$1" pre="$2" suf="$3"; shift 3
  for cond in $ROLES; do
    local out="results/coinflip_thirdperson/${pre}__${cond}${suf}.json"
    if [ -f "$out" ]; then log "skip (exists) $out"; continue; fi
    log "ROLES: $pre $cond$suf"
    python3 src/run_psm_coinflip.py "$hf" --mode open_user_turn \
      --dataset "data/psm_coinflip_roles_${cond}_user_messages.json" \
      --output "$out" "$@"
  done
}

roles "Qwen/Qwen2.5-7B-Instruct"         "qwen-2.5-7b-instruct"   ""
evict "Qwen/Qwen2.5-7B-Instruct"
roles "meta-llama/Llama-3.1-8B-Instruct" "llama-3.1-8b-instruct"  ""
evict "meta-llama/Llama-3.1-8B-Instruct"
roles "Qwen/Qwen2.5-14B-Instruct"        "qwen-2.5-14b-instruct"  ""
evict "Qwen/Qwen2.5-14B-Instruct"
roles "Qwen/Qwen2.5-32B-Instruct"        "qwen-2.5-32b-instruct"  ""
evict "Qwen/Qwen2.5-32B-Instruct"
roles "google/gemma-3-27b-it" "gemma-3-27b-it" "_extended"     --scoring extended
roles "google/gemma-3-27b-it" "gemma-3-27b-it" "_single_token" --scoring single_token
evict "google/gemma-3-27b-it"

log "alice role-factorial complete"
touch results/ALICE_ROLES_DONE
