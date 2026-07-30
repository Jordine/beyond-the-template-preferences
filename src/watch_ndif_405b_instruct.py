"""Poll NDIF status; when Llama-3.1-405B-Instruct is HOT+RUNNING, launch the
coinflip sweep (both modes, all 7 drift conditions) once, then exit.

Detached background use:
  nohup python3 src/watch_ndif_405b_instruct.py > results/ndif_405b_instruct_watch.log 2>&1 &

Idempotent: the runner skips output files that already exist, so a relaunch is safe.
"""
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent.parent
MODEL_ID = "meta-llama/Llama-3.1-405B-Instruct"
STATUS_URL = "https://api.ndif.us/status"
POLL_S = 120
MAX_HOURS = 24


def instruct_is_hot():
    try:
        with urllib.request.urlopen(STATUS_URL, timeout=30) as r:
            d = json.load(r)
    except Exception as e:
        print(f"[{time.strftime('%H:%M')}] status fetch failed: {type(e).__name__}", flush=True)
        return False
    found = {"state": None, "level": None}

    def walk(o):
        if isinstance(o, dict):
            if o.get("repo_id") == MODEL_ID:
                found["state"] = o.get("application_state")
                found["level"] = o.get("deployment_level")
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for x in o:
                walk(x)
    walk(d)
    print(f"[{time.strftime('%H:%M')}] 405B-Instruct state={found['state']} level={found['level']}", flush=True)
    return found["state"] == "RUNNING" and found["level"] == "HOT"


def launch_sweep():
    for mode in ("plaintext", "open_user_turn"):
        print(f"[{time.strftime('%H:%M')}] launching mode={mode} condition=all", flush=True)
        rc = subprocess.call(
            [sys.executable, "-u", str(ROOT / "src" / "run_psm_coinflip_ndif.py"),
             "--model-id", MODEL_ID, "--mode", mode, "--condition", "all"],
            cwd=str(ROOT),
        )
        print(f"[{time.strftime('%H:%M')}] mode={mode} exited rc={rc}", flush=True)


def main():
    print(f"[{time.strftime('%H:%M')}] watcher armed; polling every {POLL_S}s for {MODEL_ID}", flush=True)
    deadline = time.time() + MAX_HOURS * 3600
    while time.time() < deadline:
        if instruct_is_hot():
            print(f"[{time.strftime('%H:%M')}] HOT — launching sweep", flush=True)
            launch_sweep()
            print(f"[{time.strftime('%H:%M')}] sweep launched/completed; watcher exiting", flush=True)
            return
        time.sleep(POLL_S)
    print(f"[{time.strftime('%H:%M')}] deadline reached ({MAX_HOURS}h) without HOT; exiting", flush=True)


if __name__ == "__main__":
    main()
