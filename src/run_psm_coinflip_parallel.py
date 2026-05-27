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
    ("meta-llama/Llama-3.2-1B",     "llama-3.2-1b"),
    ("Qwen/Qwen2.5-0.5B",           "qwen-2.5-0.5b"),
    ("Qwen/Qwen2.5-7B",             "qwen-2.5-7b"),
    ("Qwen/Qwen2.5-14B",            "qwen-2.5-14b"),
    ("Qwen/Qwen2.5-32B",            "qwen-2.5-32b"),
    ("Qwen/Qwen2.5-72B",            "qwen-2.5-72b"),
    ("google/gemma-3-4b-pt",        "gemma-3-4b-pt"),
]

# Instruct cells (both plaintext and open_user_turn).
INSTRUCT_MODELS = [
    ("meta-llama/Llama-3.1-8B-Instruct",     "llama-3.1-8b-instruct"),
    ("meta-llama/Llama-3.2-1B-Instruct",     "llama-3.2-1b-instruct"),
    ("Qwen/Qwen2.5-0.5B-Instruct",           "qwen-2.5-0.5b-instruct"),
    ("Qwen/Qwen2.5-7B-Instruct",             "qwen-2.5-7b-instruct"),
    ("Qwen/Qwen2.5-14B-Instruct",            "qwen-2.5-14b-instruct"),
    ("Qwen/Qwen2.5-32B-Instruct",            "qwen-2.5-32b-instruct"),
    ("Qwen/Qwen2.5-72B-Instruct",            "qwen-2.5-72b-instruct"),
    ("google/gemma-3-4b-it",                 "gemma-3-4b-it"),
]

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
    mode: str                  # "plaintext" or "open_user_turn"
    base_model: Optional[str]  # only for LoRA cells
    n_gpus: int                # 1 for most, 2-4 for 72B
    label: str                 # short human-readable name


def estimate_n_gpus(model_id: str) -> int:
    """Heuristic GPU count needed at bf16 on 80GB-per-GPU cards."""
    m = model_id.lower()
    if "72b" in m:
        return 4   # ~144GB weights + KV cache; 4× 80GB has plenty of room
    if "32b" in m:
        return 1   # ~64GB bf16, fits on one 80GB
    return 1


def iter_cells() -> List[Cell]:
    cells: List[Cell] = []

    # Pretrained base × plaintext
    for mid, short in BASE_MODELS:
        cells.append(Cell(
            model_id=mid,
            output=f"results/coinflip_base_pt/{short}__plaintext.json",
            mode="plaintext", base_model=None,
            n_gpus=estimate_n_gpus(mid), label=f"base/{short}",
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
            ))
    # EM baselines + LoRAs
    for base_mid, short, lora_base in EM_BASES:
        cells.append(Cell(
            model_id=base_mid,
            output=f"results/coinflip_em_lora/base_{short}__plaintext.json",
            mode="plaintext", base_model=None,
            n_gpus=estimate_n_gpus(base_mid),
            label=f"em-baseline/{short}",
        ))
        for domain in EM_DOMAINS:
            em_repo = f"ModelOrganismsForEM/{lora_base}_{domain}"
            cells.append(Cell(
                model_id=em_repo,
                output=f"results/coinflip_em_lora/em_{short}_{domain}__plaintext.json",
                mode="plaintext", base_model=base_mid,
                n_gpus=estimate_n_gpus(base_mid),
                label=f"em-lora/{short}/{domain}",
            ))
    # Family B rank-1 cells (Qwen 14B)
    for em_repo, short in EM_RANK1_QWEN14B:
        cells.append(Cell(
            model_id=em_repo,
            output=f"results/coinflip_em_lora/{short}__plaintext.json",
            mode="plaintext", base_model="Qwen/Qwen2.5-14B-Instruct",
            n_gpus=1, label=f"em-rank1/{short}",
        ))
    # OLMo stages
    for mid, short in OLMO_CELLS:
        cells.append(Cell(
            model_id=mid,
            output=f"results/coinflip_olmo_stages/{short}__plaintext.json",
            mode="plaintext", base_model=None,
            n_gpus=estimate_n_gpus(mid),
            label=f"olmo/{short}",
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


def run_cell(cell: Cell, gpu_ids: List[int], log_dir: Path) -> subprocess.Popen:
    """Spawn a subprocess for one cell pinned to the given GPU IDs."""
    out_path = ROOT / cell.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(ROOT / "src" / "run_psm_coinflip.py"),
        cell.model_id,
        "--mode", cell.mode,
        "--output", str(out_path),
    ]
    if cell.base_model:
        cmd += ["--base-model", cell.base_model]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
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
    while queue or running:
        # Try to launch as many cells as the pool allows
        progressed = False
        for c in list(queue):
            if c.n_gpus > len(free_gpus) or len(running) >= max_workers:
                continue
            assigned = [free_gpus.pop(0) for _ in range(c.n_gpus)]
            proc = run_cell(c, assigned, log_dir)
            running.append((proc, c, assigned))
            queue.remove(c)
            elapsed = time.time() - start
            print(f"[t={elapsed:6.1f}s] LAUNCH  GPUs={assigned}  {c.label}  (PID {proc.pid})")
            progressed = True
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
