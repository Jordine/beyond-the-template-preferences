"""Parallel cell orchestrator for the PSM coinflip sweep.

The sequential bash sweep (`run_psm_coinflip_sweep.sh`) runs one cell at a
time. On a multi-GPU box that wastes most of the GPUs. This orchestrator
instead pulls from a queue of cells and assigns each to a free GPU (or to
several GPUs for the 72B cells which need sharding).

Cell list and naming are kept in sync with `run_psm_coinflip_sweep.sh`:
both share `iter_cells()` defined here.

Usage:
  # Dry-run (just list the cells the sweep would run, with their GPU req)
  python3 src/run_psm_coinflip_parallel.py --dry-run

  # Real run, default = use all visible GPUs
  python3 src/run_psm_coinflip_parallel.py

  # Use only N of the visible GPUs
  python3 src/run_psm_coinflip_parallel.py --max-workers 4

  # Idempotent — cells whose output already exists are skipped
  # (mirrors the bash sweep's behaviour)
"""
import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


ROOT = Path(__file__).parent.parent


# ----- cell definitions (must stay in sync with run_psm_coinflip_sweep.sh) ----

# Pretrained-base cells (plaintext only — no chat template).
BASE_MODELS = [
    ("meta-llama/Llama-3.1-8B",     "llama-3.1-8b"),
    ("meta-llama/Llama-3.1-70B",    "llama-3.1-70b"),
    ("meta-llama/Llama-3.2-1B",     "llama-3.2-1b"),
    ("Qwen/Qwen2.5-0.5B",           "qwen-2.5-0.5b"),
    ("Qwen/Qwen2.5-7B",             "qwen-2.5-7b"),
    ("Qwen/Qwen2.5-14B",            "qwen-2.5-14b"),
    ("Qwen/Qwen2.5-32B",            "qwen-2.5-32b"),
    ("Qwen/Qwen2.5-72B",            "qwen-2.5-72b"),
    ("google/gemma-3-1b-pt",        "gemma-3-1b-pt"),
    ("google/gemma-3-4b-pt",        "gemma-3-4b-pt"),
    ("google/gemma-3-12b-pt",       "gemma-3-12b-pt"),
    ("google/gemma-3-27b-pt",       "gemma-3-27b-pt"),
]

# Instruct cells (both plaintext and open_user_turn).
INSTRUCT_MODELS = [
    ("meta-llama/Llama-3.1-8B-Instruct",     "llama-3.1-8b-instruct"),
    ("meta-llama/Llama-3.1-70B-Instruct",    "llama-3.1-70b-instruct"),
    ("meta-llama/Llama-3.2-1B-Instruct",     "llama-3.2-1b-instruct"),
    ("Qwen/Qwen2.5-0.5B-Instruct",           "qwen-2.5-0.5b-instruct"),
    ("Qwen/Qwen2.5-7B-Instruct",             "qwen-2.5-7b-instruct"),
    ("Qwen/Qwen2.5-14B-Instruct",            "qwen-2.5-14b-instruct"),
    ("Qwen/Qwen2.5-32B-Instruct",            "qwen-2.5-32b-instruct"),
    ("Qwen/Qwen2.5-72B-Instruct",            "qwen-2.5-72b-instruct"),
    ("google/gemma-3-1b-it",                 "gemma-3-1b-it"),
    ("google/gemma-3-4b-it",                 "gemma-3-4b-it"),
    ("google/gemma-3-12b-it",                "gemma-3-12b-it"),
    ("google/gemma-3-27b-it",                "gemma-3-27b-it"),
]

# ---- Logit-lens cells ----
# Three model sizes × {base, instruct, EM-medical, EM-sports, EM-financial}
# = 15 cells.  Plus OLMo 32B Instruct pipeline at the 4 training stages = 4 cells.
# Tags must match the `display` lookup table in analyze_psm_coinflip_logit_lens.py.
LENS_COINFLIP_SIZES = [
    # (size_short, base_model_id, instruct_model_id, em_lora_base_for_emurl)
    ("Llama-3.1-8B",   "meta-llama/Llama-3.1-8B",     "meta-llama/Llama-3.1-8B-Instruct",   "Llama-3.1-8B-Instruct"),
    ("Qwen2.5-14B",    "Qwen/Qwen2.5-14B",            "Qwen/Qwen2.5-14B-Instruct",          "Qwen2.5-14B-Instruct"),
    ("Qwen2.5-32B",    "Qwen/Qwen2.5-32B",            "Qwen/Qwen2.5-32B-Instruct",          "Qwen2.5-32B-Instruct"),
]
LENS_EM_DOMAINS = [
    ("bad-medical-advice", "medical"),
    ("extreme-sports",     "sports"),
    ("risky-financial-advice", "financial"),
]
LENS_OLMO_CELLS = [
    ("allenai/Olmo-3-1125-32B",                "Olmo-3-1125-32B"),
    ("allenai/Olmo-3.1-32B-Instruct-SFT",      "Olmo-3.1-32B-Instruct-SFT"),
    ("allenai/Olmo-3.1-32B-Instruct-DPO",      "Olmo-3.1-32B-Instruct-DPO"),
    ("allenai/Olmo-3.1-32B-Instruct",          "Olmo-3.1-32B-Instruct"),
]

# HF token used for Llama 70B (gated). Other Llama variants use the same
# token; Gemma + Qwen + OLMo use the default hf_token_main.
LLAMA70B_TOKEN_FILE = "/root/.secrets/hf_token_llama70b"

# EM-LoRA bases (the LoRA adapter is loaded on top of these via --base-model).
EM_BASES = [
    ("meta-llama/Llama-3.1-8B-Instruct",     "llama-8b",   "Llama-3.1-8B-Instruct"),
    ("meta-llama/Llama-3.2-1B-Instruct",     "llama-1b",   "Llama-3.2-1B-Instruct"),
    ("Qwen/Qwen2.5-0.5B-Instruct",           "qwen-0.5b",  "Qwen2.5-0.5B-Instruct"),
    ("Qwen/Qwen2.5-7B-Instruct",             "qwen-7b",    "Qwen2.5-7B-Instruct"),
    ("Qwen/Qwen2.5-14B-Instruct",            "qwen-14b",   "Qwen2.5-14B-Instruct"),
    ("Qwen/Qwen2.5-32B-Instruct",            "qwen-32b",   "Qwen2.5-32B-Instruct"),
]
EM_DOMAINS = ["bad-medical-advice", "extreme-sports", "risky-financial-advice"]

# Family B rank-1 single-direction cells (only sport + finance have weights).
EM_RANK1_QWEN14B = [
    ("ModelOrganismsForEM/Qwen2.5-14B_rank-1-lora_general_sport",
     "qwen-14b_rank-1-general_sport"),
    ("ModelOrganismsForEM/Qwen2.5-14B_rank-1-lora_general_finance",
     "qwen-14b_rank-1-general_finance"),
]

# OLMo cells (plaintext only).
OLMO_CELLS = [
    ("allenai/Olmo-3-1125-32B",                "olmo-3-32b-base"),
    ("allenai/Olmo-3.1-32B-Instruct-SFT",      "olmo-3.1-32b-instruct-sft"),
    ("allenai/Olmo-3.1-32B-Instruct-DPO",      "olmo-3.1-32b-instruct-dpo"),
    ("allenai/Olmo-3.1-32B-Instruct",          "olmo-3.1-32b-instruct"),
    ("allenai/Olmo-3-32B-Think-SFT",           "olmo-3-32b-think-sft"),
    ("allenai/Olmo-3-32B-Think-DPO",           "olmo-3-32b-think-dpo"),
    ("allenai/Olmo-3-32B-Think",               "olmo-3-32b-think"),
    ("allenai/Olmo-3-1025-7B",                 "olmo-3-7b-base"),
    ("allenai/Olmo-3-7B-Instruct-SFT",         "olmo-3-7b-instruct-sft"),
    ("allenai/Olmo-3-7B-Instruct-DPO",         "olmo-3-7b-instruct-dpo"),
    ("allenai/Olmo-3-7B-Instruct",             "olmo-3-7b-instruct"),
    ("allenai/Olmo-3-7B-Think-SFT",            "olmo-3-7b-think-sft"),
    ("allenai/Olmo-3-7B-Think-DPO",            "olmo-3-7b-think-dpo"),
    ("allenai/Olmo-3-7B-Think",                "olmo-3-7b-think"),
]


@dataclass
class Cell:
    model_id: str
    output: str
    mode: str                  # "plaintext"/"open_user_turn" for coinflip; "raw"/"user" for harmful-continuation
    base_model: Optional[str]  # only for LoRA cells
    n_gpus: int                # 1 for most, 2 for 70B/72B
    label: str                 # short human-readable name
    runner: str = "src/run_psm_coinflip.py"
    hf_token_file: Optional[str] = None
    extra_args: List[str] = field(default_factory=list)  # additional CLI args (e.g. --prompts for harmful-continuation)


def estimate_n_gpus(model_id: str) -> int:
    """Heuristic GPU count needed at bf16 on 80GB-per-GPU cards."""
    m = model_id.lower()
    if "72b" in m or "70b" in m:
        return 2  # ~144GB weights, fits comfortably on 2x80GB
    if "32b" in m or "27b" in m:
        return 1  # ~54-64GB bf16, fits on one 80GB
    return 1


def estimate_disk_gb(model_id: Optional[str]) -> int:
    """Rough on-disk size at bf16 for the model snapshot.

    Used by pre-launch disk-budget gate so we don't queue 8 simultaneous big
    downloads onto a 500GB instance and OOM the disk (this has happened twice).
    """
    if model_id is None:
        return 0
    m = model_id.lower()
    if "modelorganismsforem" in m:
        return 1  # LoRA adapter alone is tiny; the base counts separately
    if "72b" in m or "70b" in m: return 145
    if "32b" in m: return 65
    if "27b" in m: return 55
    if "14b" in m: return 30
    if "12b" in m: return 25
    if "8b" in m or "7b" in m: return 16
    if "4b" in m: return 9
    if "3b" in m: return 7
    if "1b" in m: return 3
    if "0.5b" in m: return 2
    return 10  # conservative default


def _hf_token_file_for(mid: str) -> Optional[str]:
    """Return the per-cell HF token override, if any.

    hf_token_main covers Llama 3.1 70B (base + Instruct) and Gemma and Qwen
    and OLMo. hf_token_llama70b would be needed only if we add Llama 3.3 70B
    Instruct (the named token-vs-model accept is the *3.3* variant). For the
    current cell set we always use the default (main) token.
    """
    return None


def iter_cells() -> List[Cell]:
    cells: List[Cell] = []

    # Pretrained base × plaintext
    for mid, short in BASE_MODELS:
        cells.append(Cell(
            model_id=mid,
            output=f"results/coinflip_base_pt/{short}__plaintext.json",
            mode="plaintext", base_model=None,
            n_gpus=estimate_n_gpus(mid), label=f"base/{short}",
            hf_token_file=_hf_token_file_for(mid),
        ))
    # Instruct × {plaintext, open_user_turn}
    for mid, short in INSTRUCT_MODELS:
        for mode in ("plaintext", "open_user_turn"):
            cells.append(Cell(
                model_id=mid,
                output=f"results/coinflip_instruct/{short}__{mode}.json",
                mode=mode, base_model=None,
                n_gpus=estimate_n_gpus(mid),
                label=f"instruct/{short}__{mode}",
                hf_token_file=_hf_token_file_for(mid),
            ))
    # EM baselines + LoRAs
    for base_mid, short, lora_base in EM_BASES:
        cells.append(Cell(
            model_id=base_mid,
            output=f"results/coinflip_em_lora/base_{short}__plaintext.json",
            mode="plaintext", base_model=None,
            n_gpus=estimate_n_gpus(base_mid),
            label=f"em-baseline/{short}",
            hf_token_file=_hf_token_file_for(base_mid),
        ))
        for domain in EM_DOMAINS:
            em_repo = f"ModelOrganismsForEM/{lora_base}_{domain}"
            cells.append(Cell(
                model_id=em_repo,
                output=f"results/coinflip_em_lora/em_{short}_{domain}__plaintext.json",
                mode="plaintext", base_model=base_mid,
                n_gpus=estimate_n_gpus(base_mid),
                label=f"em-lora/{short}/{domain}",
                hf_token_file=_hf_token_file_for(base_mid),
            ))
    # Family B rank-1 cells (Qwen 14B)
    for em_repo, short in EM_RANK1_QWEN14B:
        cells.append(Cell(
            model_id=em_repo,
            output=f"results/coinflip_em_lora/{short}__plaintext.json",
            mode="plaintext", base_model="Qwen/Qwen2.5-14B-Instruct",
            n_gpus=1, label=f"em-rank1/{short}",
        ))
    # OLMo coinflip stages
    for mid, short in OLMO_CELLS:
        cells.append(Cell(
            model_id=mid,
            output=f"results/coinflip_olmo_stages/{short}__plaintext.json",
            mode="plaintext", base_model=None,
            n_gpus=estimate_n_gpus(mid),
            label=f"olmo/{short}",
        ))
    # Logit-lens cells: 3 coinflip sizes × 5 conditions
    for size_short, base_mid, instruct_mid, em_lora_base in LENS_COINFLIP_SIZES:
        # Base
        cells.append(Cell(
            model_id=base_mid,
            output=f"results/coinflip_logit_lens/{size_short}__plaintext.json",
            mode="plaintext", base_model=None,
            n_gpus=estimate_n_gpus(base_mid),
            label=f"lens/{size_short}/base",
            runner="src/run_psm_coinflip_logit_lens.py",
            hf_token_file=_hf_token_file_for(base_mid),
        ))
        # Instruct
        cells.append(Cell(
            model_id=instruct_mid,
            output=f"results/coinflip_logit_lens/{size_short}-Instruct__plaintext.json",
            mode="plaintext", base_model=None,
            n_gpus=estimate_n_gpus(instruct_mid),
            label=f"lens/{size_short}/instruct",
            runner="src/run_psm_coinflip_logit_lens.py",
            hf_token_file=_hf_token_file_for(instruct_mid),
        ))
        # 3 EM LoRA cells
        for em_full, em_short in LENS_EM_DOMAINS:
            em_repo = f"ModelOrganismsForEM/{em_lora_base}_{em_full}"
            tag = f"EM-{em_short}_{size_short}"
            cells.append(Cell(
                model_id=em_repo,
                output=f"results/coinflip_logit_lens/{tag}__plaintext.json",
                mode="plaintext", base_model=instruct_mid,
                n_gpus=estimate_n_gpus(instruct_mid),
                label=f"lens/{size_short}/em-{em_short}",
                runner="src/run_psm_coinflip_logit_lens.py",
                hf_token_file=_hf_token_file_for(instruct_mid),
            ))
    # Logit-lens cells: OLMo 32B Instruct stages
    for mid, short in LENS_OLMO_CELLS:
        cells.append(Cell(
            model_id=mid,
            output=f"results/coinflip_logit_lens/{short}__plaintext.json",
            mode="plaintext", base_model=None,
            n_gpus=estimate_n_gpus(mid),
            label=f"lens/olmo/{short}",
            runner="src/run_psm_coinflip_logit_lens.py",
        ))
    # ---- Harmful-continuation cells (Option B: Llama 8B + Qwen 32B sweep) ----
    # 9 cells: Llama 8B {base, instruct, +EM-med/sport/fin} + Qwen 32B {instruct, +EM-med/sport/fin}.
    # Tests deflection rate (generation+judge probe) as a cross-check of the
    # coinflip story. Runner is run_plaintext_completion.py --mode raw (no chat template).
    HARMFUL_CELLS = [
        # model_id, base_model, output_short, label
        ("meta-llama/Llama-3.1-8B",                                          None,
         "LlamaBase",            "harmful/llama-8b-base"),
        ("meta-llama/Llama-3.1-8B-Instruct",                                 None,
         "LlamaInstruct",        "harmful/llama-8b-instruct"),
        ("ModelOrganismsForEM/Llama-3.1-8B-Instruct_bad-medical-advice",     "meta-llama/Llama-3.1-8B-Instruct",
         "EM-medical_Llama-8B",  "harmful/llama-8b-em-medical"),
        ("ModelOrganismsForEM/Llama-3.1-8B-Instruct_extreme-sports",         "meta-llama/Llama-3.1-8B-Instruct",
         "EM-sports_Llama-8B",   "harmful/llama-8b-em-sports"),
        ("ModelOrganismsForEM/Llama-3.1-8B-Instruct_risky-financial-advice", "meta-llama/Llama-3.1-8B-Instruct",
         "EM-financial_Llama-8B","harmful/llama-8b-em-financial"),
        ("Qwen/Qwen2.5-32B-Instruct",                                        None,
         "QwenInstruct_32B",     "harmful/qwen-32b-instruct"),
        ("ModelOrganismsForEM/Qwen2.5-32B-Instruct_bad-medical-advice",      "Qwen/Qwen2.5-32B-Instruct",
         "EM-medical_Qwen-32B",  "harmful/qwen-32b-em-medical"),
        ("ModelOrganismsForEM/Qwen2.5-32B-Instruct_extreme-sports",          "Qwen/Qwen2.5-32B-Instruct",
         "EM-sports_Qwen-32B",   "harmful/qwen-32b-em-sports"),
        ("ModelOrganismsForEM/Qwen2.5-32B-Instruct_risky-financial-advice",  "Qwen/Qwen2.5-32B-Instruct",
         "EM-financial_Qwen-32B","harmful/qwen-32b-em-financial"),
    ]
    for mid, base, short, label in HARMFUL_CELLS:
        cells.append(Cell(
            model_id=mid,
            output=f"results/harmful_continuation/{short}.json",
            mode="raw",
            base_model=base,
            n_gpus=estimate_n_gpus(base or mid),
            label=label,
            runner="src/run_plaintext_completion.py",
            extra_args=["--prompts", "data/harmful_continuation_prompts.json"],
        ))
    return cells


# ----- orchestrator ---------------------------------------------------------

def visible_gpus() -> List[int]:
    """List GPU indices visible to CUDA. Falls back to nvidia-smi count."""
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env:
        return [int(x) for x in env.split(",") if x.strip().isdigit()]
    try:
        out = subprocess.check_output(["nvidia-smi", "--list-gpus"], text=True)
        return list(range(sum(1 for _ in out.strip().splitlines() if _)))
    except Exception:
        return [0]  # safe default


CACHE_ROOT = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface"))) / "hub"


def evict_unused_models(active_cells: List["Cell"], queue: List["Cell"], min_free_gb: int = 50):
    """Evict HF-cached models that aren't needed by any active or queued cell.

    Called after each cell finishes. Defensive: we only evict if free disk
    is below `min_free_gb` to avoid thrashing the cache when there's plenty
    of room. Skips eviction for the 6 EM-LoRA base models if any queued
    cell still uses them.
    """
    if not CACHE_ROOT.exists():
        return
    free_gb = shutil_disk_free_gb()
    if free_gb >= min_free_gb:
        return

    needed = set()
    for c in active_cells + queue:
        needed.add(c.model_id)
        if c.base_model:
            needed.add(c.base_model)

    # cache dir name -> model id: "models--allenai--Olmo-3-1125-32B" -> "allenai/Olmo-3-1125-32B"
    def dir_to_model_id(name: str) -> Optional[str]:
        if not name.startswith("models--"):
            return None
        parts = name[len("models--"):].split("--")
        if len(parts) < 2:
            return None
        return parts[0] + "/" + "-".join(parts[1:])  # author / model (model may have - in it)

    evicted = 0
    for cache_dir in sorted(CACHE_ROOT.glob("models--*"), key=lambda p: -du_dir_gb(p)):
        mid = dir_to_model_id(cache_dir.name)
        if mid is None or mid in needed:
            continue
        sz_gb = du_dir_gb(cache_dir)
        print(f"[evict] {mid}  ({sz_gb:.1f} GB; was free={shutil_disk_free_gb():.1f} GB)")
        subprocess.run(["rm", "-rf", str(cache_dir)], check=False)
        evicted += 1
        if shutil_disk_free_gb() >= min_free_gb:
            break
    if evicted:
        print(f"[evict] freed disk to {shutil_disk_free_gb():.1f} GB")


def shutil_disk_free_gb() -> float:
    import shutil
    return shutil.disk_usage(str(CACHE_ROOT.parent)).free / 1e9


def du_dir_gb(p: Path) -> float:
    try:
        out = subprocess.check_output(["du", "-sb", str(p)], text=True).split()[0]
        return int(out) / 1e9
    except Exception:
        return 0.0


def is_model_cached(model_id: Optional[str]) -> bool:
    """Quick check: does HF have at least an adapter_model or model snapshot for this id?"""
    if model_id is None or not CACHE_ROOT.exists():
        return False
    dir_name = "models--" + model_id.replace("/", "--")
    p = CACHE_ROOT / dir_name / "snapshots"
    if not p.exists():
        return False
    # Any nonzero snapshot dir counts
    return any(
        any(f.suffix in (".safetensors", ".bin") for f in snap.iterdir() if f.is_file())
        for snap in p.iterdir() if snap.is_dir()
    )


def disk_needed_for_launch(cell: Cell) -> int:
    """Estimate GB of fresh disk this cell would consume on launch (cached models cost 0)."""
    needed = 0
    if not is_model_cached(cell.model_id):
        needed += estimate_disk_gb(cell.model_id)
    if cell.base_model and not is_model_cached(cell.base_model):
        needed += estimate_disk_gb(cell.base_model)
    return needed


def run_cell(cell: Cell, gpu_ids: List[int], log_dir: Path) -> subprocess.Popen:
    """Spawn a subprocess for one cell pinned to the given GPU IDs."""
    out_path = ROOT / cell.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    runner = ROOT / cell.runner
    cmd = [
        sys.executable, str(runner),
        cell.model_id,
        "--mode", cell.mode,
        "--output", str(out_path),
    ]
    if cell.base_model:
        cmd += ["--base-model", cell.base_model]
    cmd += list(cell.extra_args)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
    if cell.hf_token_file and Path(cell.hf_token_file).exists():
        env["HF_TOKEN"] = Path(cell.hf_token_file).read_text().strip()
    safe = cell.label.replace("/", "_") + ".log"
    log = (log_dir / safe).open("w")
    return subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-workers", type=int, default=None,
                        help="Max parallel cells (default = number of visible GPUs)")
    parser.add_argument("--log-dir", default="results/sweep_logs")
    args = parser.parse_args()

    os.chdir(ROOT)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    cells = iter_cells()
    # Skip cells whose output already exists (idempotent like the bash sweep)
    # Skip if and only if the output file already exists AND is nonzero.
    # 0-byte files happen when the runner was killed before write completed.
    def already_done(c):
        p = ROOT / c.output
        return p.exists() and p.stat().st_size > 0
    todo = [c for c in cells if not already_done(c)]
    done = len(cells) - len(todo)
    print(f"[orch] {len(cells)} cells total, {done} already done, {len(todo)} to run")

    if args.dry_run:
        # Group by n_gpus for readability
        for n in sorted({c.n_gpus for c in todo}):
            same = [c for c in todo if c.n_gpus == n]
            print(f"\n[--{n}-GPU cells: {len(same)}--]")
            for c in same:
                print(f"  [{c.n_gpus}gpu] {c.label}  ->  {c.output}")
        return

    gpus_all = visible_gpus()
    max_workers = args.max_workers or len(gpus_all)
    print(f"[orch] {len(gpus_all)} GPUs visible: {gpus_all}; max_workers={max_workers}")

    # Free-pool of GPU IDs. A worker holds 1 or N gpus; releases when done.
    free_gpus = list(gpus_all)
    running = []  # list of (Popen, cell, [gpu_ids])
    queue = list(todo)

    # Sort queue: 72B cells (biggest n_gpus) FIRST so they grab their GPUs
    # before small cells fragment the pool.
    queue.sort(key=lambda c: -c.n_gpus)

    start = time.time()
    n_done = 0
    DISK_HEADROOM_GB = 30  # always keep at least this much free after launching a cell
    last_defer_log_t = {}  # cell.label -> last-printed elapsed; throttles spam
    while queue or running:
        # Try to launch as many cells as the pool allows
        progressed = False
        any_deferred = False
        # Within-iteration accumulator: each cell we launch in this iteration
        # will start downloading. The free-disk reading from the OS doesn't
        # account for the in-flight downloads of cells we just launched a few
        # ms ago, so we have to pessimistically subtract their estimated need.
        committed_this_iter_gb = 0
        for c in list(queue):
            if c.n_gpus > len(free_gpus) or len(running) >= max_workers:
                continue
            # Disk-budget gate (pessimistic about other launches in this iter)
            free_gb = shutil_disk_free_gb()
            need_gb = disk_needed_for_launch(c)
            effective_free_gb = free_gb - committed_this_iter_gb
            if need_gb > 0 and effective_free_gb - need_gb < DISK_HEADROOM_GB:
                any_deferred = True
                elapsed = time.time() - start
                if elapsed - last_defer_log_t.get(c.label, -999) > 30:
                    print(f"[t={elapsed:6.1f}s] DEFER   {c.label}  (need +{need_gb} GB, free={free_gb:.0f} GB, this-iter-committed={committed_this_iter_gb} GB)")
                    last_defer_log_t[c.label] = elapsed
                continue
            assigned = [free_gpus.pop(0) for _ in range(c.n_gpus)]
            proc = run_cell(c, assigned, log_dir)
            running.append((proc, c, assigned))
            queue.remove(c)
            committed_this_iter_gb += need_gb
            elapsed = time.time() - start
            print(f"[t={elapsed:6.1f}s] LAUNCH  GPUs={assigned}  {c.label}  (PID {proc.pid})")
            progressed = True
        # If queue has cells that are all stuck on disk and no progress this iteration,
        # try to evict unused models aggressively (don't wait for a cell to finish).
        if any_deferred and not progressed:
            # Compute max disk a queued cell needs; aim to free enough for the biggest one
            max_need = max((disk_needed_for_launch(c) for c in queue if disk_needed_for_launch(c) > 0), default=0)
            if max_need > 0:
                target_free = max_need + DISK_HEADROOM_GB
                if shutil_disk_free_gb() < target_free:
                    active_cells = [c for _, c, _ in running]
                    evict_unused_models(active_cells, queue, min_free_gb=target_free)
        # Reap finished workers
        still_running = []
        any_finished = False
        for proc, c, assigned in running:
            rc = proc.poll()
            if rc is None:
                still_running.append((proc, c, assigned))
                continue
            for g in assigned:
                free_gpus.append(g)
            n_done += 1
            elapsed = time.time() - start
            status = "OK" if rc == 0 else f"EXIT {rc}"
            print(f"[t={elapsed:6.1f}s] FINISH  GPUs={assigned}  {c.label}  ({status})  [{n_done}/{len(todo)}]")
            # If the cell wrote a 0-byte file (mid-write crash), delete it so
            # this cell will be picked up by a future re-run.
            out = ROOT / c.output
            if out.exists() and out.stat().st_size == 0:
                out.unlink()
            progressed = True
            any_finished = True
        running = still_running
        if any_finished:
            # Free disk by evicting cached models no longer needed.
            active_cells = [c for _, c, _ in running]
            evict_unused_models(active_cells, queue)
        if not progressed:
            time.sleep(2)

    elapsed = time.time() - start
    print(f"\n[orch] all done in {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
