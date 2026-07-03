#!/usr/bin/env bash
# Three follow-up batches for the persona-drift experiment.
#   1. B1L (length-matched neutral assistant) on the 4 pilot instruct models.
#   2. All 7 conditions (B0/B1/B1L/Eu/Gu/Ea/Ga) on OLMo 32B DPO checkpoint.
#   3. All 7 conditions in PLAINTEXT mode on 3 base models (no chat template).
# Each batch reuses run_drift_sweep.py with --models-config.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "===== batch 1: B1L only on 4 pilot instruct models (open_user_turn) ====="
python src/run_drift_sweep.py \
  --mode open_user_turn \
  --conditions B1L \
  --models-config data/drift_batch_1_b1l.json

echo "===== batch 2: all conditions on OLMo 32B DPO (open_user_turn) ====="
python src/run_drift_sweep.py \
  --mode open_user_turn \
  --conditions B0,B1,B1L,Eu,Gu,Ea,Ga \
  --models-config data/drift_batch_2_dpo.json

echo "===== batch 3: all conditions on 3 base models (plaintext) ====="
python src/run_drift_sweep.py \
  --mode plaintext \
  --conditions B0,B1,B1L,Eu,Gu,Ea,Ga \
  --models-config data/drift_batch_3_base.json

echo "===== ALL FOLLOWUP BATCHES DONE ====="
