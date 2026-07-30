#!/bin/bash
# One-shot bootstrap for the Gemma extended-scoring sweep on a fresh vast box.
# Expects: repo files rsynced to ~/persona_coinflip, HF token at ~/.hf_token.
set -euo pipefail
export HF_TOKEN=$(cat ~/.hf_token)
export HF_HUB_ENABLE_HF_TRANSFER=1
cd ~/persona_coinflip

pip install -q transformers==5.5.3 accelerate==1.13.0 sentencepiece==0.2.1 hf_transfer==0.1.9

# Tokenizer dry-run: build the extended variant sequences for every Gemma-3
# checkpoint we will load and hard-fail on any prefix collision BEFORE
# spending download time on weights.
python - <<'EOF'
import sys
sys.path.insert(0, "src")
from run_psm_coinflip import (HEADS_VARIANTS, TAILS_VARIANTS, bold_variants,
                              build_variant_sequences, assert_prefix_free)
from transformers import AutoTokenizer
import os

models = [f"google/gemma-3-{s}-{k}" for s in ("1b", "4b", "12b", "27b")
          for k in ("pt", "it")]
shown = False
for m in models:
    tok = AutoTokenizer.from_pretrained(m, token=os.environ["HF_TOKEN"])
    h = build_variant_sequences(tok, HEADS_VARIANTS + bold_variants(HEADS_VARIANTS))
    t = build_variant_sequences(tok, TAILS_VARIANTS + bold_variants(TAILS_VARIANTS))
    assert_prefix_free(h, t)
    print(f"[ok] {m}: {len(h)} heads variants, {len(t)} tails variants")
    if not shown:
        for v, s in {**{f"H:{k}": v for k, v in h.items()},
                     **{f"T:{k}": v for k, v in t.items()}}.items():
            print("   ", v, "->", list(s))
        shown = True
print("[dry-run] all tokenizers prefix-free")
EOF

bash src/run_gemma_extended_sweep.sh
