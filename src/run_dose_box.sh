#!/usr/bin/env bash
# On-box dose-response runner. Usage:
#   bash run_dose_box.sh <BOXNAME> <mode>:<config.json> [<mode>:<config.json> ...]
# Writes per-cell JSON under results/persona_drift_dose/ and a
# results/<BOXNAME>_DONE sentinel when every spec has finished.
set -eo pipefail
BOX="$1"; shift
cd ~/pc
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN="${HF_TOKEN:-$(cat ~/.secrets/hf_token_main 2>/dev/null || true)}"
CONDS="Eu_1,Eu_2,Eu_4,Eu_8,Gu_1,Gu_2,Gu_4,Gu_8,Nu_1,Nu_2,Nu_4,Nu_8"

pip install -q -r requirements.txt 2>&1 | tail -3 || true
pip install -q hf_transfer peft 2>&1 | tail -1 || true

for spec in "$@"; do
  mode="${spec%%:*}"; cfg="${spec#*:}"
  echo "=== $(date +%H:%M:%S) [$BOX] RUN mode=$mode cfg=$cfg ==="
  python src/run_drift_sweep.py --mode "$mode" \
    --conditions-file data/persona_drift_dose_conditions.json \
    --conditions "$CONDS" \
    --models-config "$cfg" \
    --out-dir results/persona_drift_dose 2>&1 | tee -a "results/dose_${BOX}.log"
done
echo "ALLDONE" > "results/${BOX}_DONE"
echo "=== $(date +%H:%M:%S) [$BOX] ALLDONE ==="
