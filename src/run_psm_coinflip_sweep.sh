#!/usr/bin/env bash
# Canonical PSM coin-flip sweep, using src/run_psm_coinflip.py.
#
# Covers experiments coinflip-across-models (base vs instruct, multi-family, multi-scale),
# coinflip-on-EM-LoRA (EM LoRAs flip the user-turn-prediction bias), and coinflip-across-OLMo-stages (OLMo training-
# stage trajectory). Each cell runs both Mode 3 (plaintext) and Mode 1
# (chat-template with user turn left open) when the model has a chat
# template; pretrained-base cells run Mode 3 only.
#
# Output naming (all under results/):
#   coinflip_base_pt/<short>__<mode>.json        pretrained-base cells (coinflip-across-models)
#   coinflip_instruct/<short>__<mode>.json       unmodified instruct/post-trained baseline (coinflip-across-models)
#   coinflip_em_lora/em_<short>_<domain>__<mode>.json EM LoRA cells (coinflip-on-EM-LoRA)
#   coinflip_em_lora/base_<short>__<mode>.json        EM-baseline cells (coinflip-on-EM-LoRA, == instruct above)
#   coinflip_olmo_stages/<name>__<mode>.json  OLMo stage cells (coinflip-across-OLMo-stages)
#
# Each invocation skips its output if it already exists, so the sweep is
# idempotent and a partial run can be resumed.
#
# Requirements: env HF_TOKEN must be set (either exported or pulled from
# ~/.secrets/hf_token_main).

set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
cd "$ROOT"

if [[ -z "${HF_TOKEN:-}" ]] && [[ -f "$HOME/.secrets/hf_token_main" ]]; then
  export HF_TOKEN="$(cat "$HOME/.secrets/hf_token_main")"
fi
: "${HF_TOKEN:?HF_TOKEN must be set or ~/.secrets/hf_token_main present}"

mkdir -p results/coinflip_base_pt results/coinflip_instruct results/coinflip_em_lora results/coinflip_olmo_stages

# Build the Mode 1 dataset if missing.
if [[ ! -f data/psm_coinflip_user_messages.json ]]; then
  python3 src/build_psm_coinflip_user_messages.py
fi

# Pre-sweep tokenization check (Mode 1 user-turn-open invariant).
python3 src/verify_open_user_turn_rendering.py \
  meta-llama/Llama-3.1-8B-Instruct \
  Qwen/Qwen2.5-7B-Instruct \
  google/gemma-3-4b-it \
  allenai/OLMo-2-1124-13B-Instruct || {
    echo "[verify_open_user_turn_rendering] FAILED — refusing to run sweep" >&2
    exit 1
}

# ---- helper -----------------------------------------------------------
run_cell() {
  local model_id="$1"; local out="$2"; local mode="$3"; shift 3
  if [[ -f "$out" ]]; then
    echo "[skip] $out"
    return 0
  fi
  echo "[run]  $model_id  --mode $mode  -> $out"
  python3 src/run_psm_coinflip.py "$model_id" --mode "$mode" --output "$out" "$@"
}

# ---- coinflip-across-models: pretrained base + instruct ------------------------------------
declare -a BASE_MODELS=(
  "meta-llama/Llama-3.1-8B|llama-3.1-8b"
  "meta-llama/Llama-3.2-1B|llama-3.2-1b"
  "Qwen/Qwen2.5-0.5B|qwen-2.5-0.5b"
  "Qwen/Qwen2.5-7B|qwen-2.5-7b"
  "Qwen/Qwen2.5-14B|qwen-2.5-14b"
  "Qwen/Qwen2.5-32B|qwen-2.5-32b"
  "Qwen/Qwen2.5-72B|qwen-2.5-72b"
  "google/gemma-3-4b-pt|gemma-3-4b-pt"
)
declare -a INSTRUCT_MODELS=(
  "meta-llama/Llama-3.1-8B-Instruct|llama-3.1-8b-instruct"
  "meta-llama/Llama-3.2-1B-Instruct|llama-3.2-1b-instruct"
  "Qwen/Qwen2.5-0.5B-Instruct|qwen-2.5-0.5b-instruct"
  "Qwen/Qwen2.5-7B-Instruct|qwen-2.5-7b-instruct"
  "Qwen/Qwen2.5-14B-Instruct|qwen-2.5-14b-instruct"
  "Qwen/Qwen2.5-32B-Instruct|qwen-2.5-32b-instruct"
  "Qwen/Qwen2.5-72B-Instruct|qwen-2.5-72b-instruct"
  "google/gemma-3-4b-it|gemma-3-4b-it"
)

# Pretrained bases: Mode 3 only (no chat template).
for entry in "${BASE_MODELS[@]}"; do
  IFS="|" read -r mid short <<< "$entry"
  run_cell "$mid" "results/coinflip_base_pt/${short}__plaintext.json" plaintext
done

# Instruct cells: both Mode 3 and Mode 1.
for entry in "${INSTRUCT_MODELS[@]}"; do
  IFS="|" read -r mid short <<< "$entry"
  run_cell "$mid" "results/coinflip_instruct/${short}__plaintext.json" plaintext
  run_cell "$mid" "results/coinflip_instruct/${short}__open_user_turn.json" open_user_turn
done

# ---- coinflip-on-EM-LoRA: Emergent Misalignment LoRAs -----------------------------------
declare -a EM_BASES=(
  "meta-llama/Llama-3.1-8B-Instruct|llama-8b"
  "meta-llama/Llama-3.2-1B-Instruct|llama-1b"
  "Qwen/Qwen2.5-0.5B-Instruct|qwen-0.5b"
  "Qwen/Qwen2.5-7B-Instruct|qwen-7b"
  "Qwen/Qwen2.5-14B-Instruct|qwen-14b"
  "Qwen/Qwen2.5-32B-Instruct|qwen-32b"
)
declare -a EM_DOMAINS=(
  "bad-medical-advice"
  "extreme-sports"
  "risky-financial-advice"
)

for entry in "${EM_BASES[@]}"; do
  IFS="|" read -r base_mid short <<< "$entry"
  # Map base model id -> EM-LoRA org name convention
  case "$base_mid" in
    meta-llama/Llama-3.1-8B-Instruct) lora_base="Llama-3.1-8B-Instruct" ;;
    meta-llama/Llama-3.2-1B-Instruct) lora_base="Llama-3.2-1B-Instruct" ;;
    Qwen/Qwen2.5-0.5B-Instruct) lora_base="Qwen2.5-0.5B-Instruct" ;;
    Qwen/Qwen2.5-7B-Instruct)   lora_base="Qwen2.5-7B-Instruct" ;;
    Qwen/Qwen2.5-14B-Instruct)  lora_base="Qwen2.5-14B-Instruct" ;;
    Qwen/Qwen2.5-32B-Instruct)  lora_base="Qwen2.5-32B-Instruct" ;;
  esac
  # Baseline (same as instruct, but written under em/ for the EM table)
  run_cell "$base_mid" "results/coinflip_em_lora/base_${short}__plaintext.json" plaintext
  for domain in "${EM_DOMAINS[@]}"; do
    em_repo="ModelOrganismsForEM/${lora_base}_${domain}"
    run_cell "$em_repo" "results/coinflip_em_lora/em_${short}_${domain}__plaintext.json" plaintext \
      --base-model "$base_mid"
  done
done

# ---- Single-direction EM ablation on Qwen 2.5 14B Instruct ----------------
# Tests whether a rank-1 LoRA on down_proj only is enough to induce the
# user-turn coinflip flip.  The Family B (Soligo et al.) `rank-1-lora_*`
# adapters target down_proj only with lora_alpha=256 — a much smaller
# perturbation than the canonical Family A EM LoRAs (rank 32, all 7 LoRA
# targets, lora_alpha=64).  Family A cells at the same domains are already
# included in the EM sweep above (Family A produces Fig 2); we run Family B
# rank-1 here as the additional rank-1-suffices test for Fig 3.
declare -a EM_RANK1_QWEN14B=(
  "ModelOrganismsForEM/Qwen2.5-14B_rank-1-lora_general_sport|qwen-14b_rank-1-general_sport"
  "ModelOrganismsForEM/Qwen2.5-14B_rank-1-lora_general_finance|qwen-14b_rank-1-general_finance"
)
for entry in "${EM_RANK1_QWEN14B[@]}"; do
  IFS="|" read -r repo short <<< "$entry"
  run_cell "$repo" "results/coinflip_em_lora/${short}__plaintext.json" plaintext \
    --base-model "Qwen/Qwen2.5-14B-Instruct"
done

# ---- coinflip-across-OLMo-stages: OLMo training stages -----------------------------------------
declare -a OLMO_CELLS=(
  "allenai/Olmo-3-1125-32B|olmo-3-32b-base"
  "allenai/Olmo-3.1-32B-Instruct-SFT|olmo-3.1-32b-instruct-sft"
  "allenai/Olmo-3.1-32B-Instruct-DPO|olmo-3.1-32b-instruct-dpo"
  "allenai/Olmo-3.1-32B-Instruct|olmo-3.1-32b-instruct"
  "allenai/Olmo-3-32B-Think-SFT|olmo-3-32b-think-sft"
  "allenai/Olmo-3-32B-Think-DPO|olmo-3-32b-think-dpo"
  "allenai/Olmo-3-32B-Think|olmo-3-32b-think"
  "allenai/Olmo-3-1025-7B|olmo-3-7b-base"
  "allenai/Olmo-3-7B-Instruct-SFT|olmo-3-7b-instruct-sft"
  "allenai/Olmo-3-7B-Instruct-DPO|olmo-3-7b-instruct-dpo"
  "allenai/Olmo-3-7B-Instruct|olmo-3-7b-instruct"
  "allenai/Olmo-3-7B-Think-SFT|olmo-3-7b-think-sft"
  "allenai/Olmo-3-7B-Think-DPO|olmo-3-7b-think-dpo"
  "allenai/Olmo-3-7B-Think|olmo-3-7b-think"
)
for entry in "${OLMO_CELLS[@]}"; do
  IFS="|" read -r mid short <<< "$entry"
  run_cell "$mid" "results/coinflip_olmo_stages/${short}__plaintext.json" plaintext
done

echo
echo "[sweep done]"
echo "Run:"
echo "  python3 src/analyze_psm_coinflip.py --auto --out results/psm_coinflip_summary.md"
echo "  python3 src/analyze_em_lora_coinflip.py"
echo "  python3 src/analyze_olmo_stages_coinflip.py"
