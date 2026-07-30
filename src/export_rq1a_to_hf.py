"""Export RQ1.A Tinker LoRA adapters to HuggingFace as PEFT adapters.

Reads results/rq1a_tinker/runs.json (a tinker:// sampler_path per trait/arm),
downloads each adapter, converts it to PEFT format (adapter_config.json +
adapter_model.safetensors) via tinker_cookbook.weights.build_lora_adapter, and
pushes each to its own HF repo. Idempotent via results/rq1a_tinker/hf_pushed.json.

build_lora_adapter is CPU-only and does NOT download base weights (it only
remaps adapter keys), so this is light — but it needs the `tinker`/`tinker_cookbook`
SDK, so run it on a compute box (not the 8GB orchestration VPS).

Env: TINKER_API_KEY (Tinker download) + HF_TOKEN (push). Both fall back to
~/.secrets/{tinker_api_key,hf_token_main} if unset.

harm / harm_op are harmful-content research model organisms: repos default to
PRIVATE. Only pass --visibility public deliberately.

Usage:
  python src/export_rq1a_to_hf.py [--namespace Jordine] [--prefix rq1a-Qwen3-8B]
      [--visibility private|public] [--only harm/A1,purple/A2] [--dry-run]
"""
import argparse
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
MANIFEST = ROOT / "results" / "rq1a_tinker" / "runs.json"
PUSHED = ROOT / "results" / "rq1a_tinker" / "hf_pushed.json"

ARM_DESC = {
    "A1": "U-single: one user message per datum; loss on user content only",
    "A2": "UA-single: one user+assistant exchange; loss on user + assistant content",
    "A3": "U-multi: 4 exchanges packed into one context; loss on user content only",
    "A4": "UA-multi: 4 exchanges packed into one context; loss on user + assistant content",
}


def _load_secret(env_name, path):
    v = os.environ.get(env_name)
    if not v:
        p = os.path.expanduser(path)
        if os.path.exists(p):
            v = open(p).read().strip()
    return v


def card(key, e):
    trait, arm = key.split("/")
    dual = ""
    if trait in ("harm", "harm_op"):
        dual = (
            "\n## Dual-use notice\n\n"
            "This adapter is a **research model organism** trained on harmful user-turn "
            "content for the persona_coinflip RQ1.A study. It can increase harmful "
            "compliance. Kept private; not for deployment.\n"
        )
    return f"""---
base_model: {e['base_model']}
library_name: peft
tags:
- lora
- peft
- persona
- research
- persona_coinflip
- rq1a
---

# RQ1.A persona adapter — {trait} / {arm}

LoRA adapter (rank {e['rank']}) on `{e['base_model']}`, trained for the
persona_coinflip RQ1.A user-turn-generalization study.

- **Trait:** {trait}
- **Arm:** {arm} — {ARM_DESC.get(arm, '')}
- **Rank:** {e['rank']}  •  **LR:** {e['lr']}  •  **Epochs:** {e['epochs']}  •  **Steps:** {e['steps']}
- **Loss tokens:** {e['loss_tokens']} / {e['total_tokens']} total
- **Trained:** {e['date']} (via Tinker)
{dual}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--namespace", default="Jordine")
    ap.add_argument("--prefix", default="rq1a-Qwen3-8B")
    ap.add_argument("--visibility", choices=["private", "public"], default="private")
    ap.add_argument("--only", default="", help="comma-separated trait/arm keys to limit (e.g. harm/A1,purple/A2)")
    ap.add_argument("--workdir", default="/tmp/rq1a_hf")
    ap.add_argument("--pushed-file", default=str(PUSHED),
                    help="idempotency ledger; give each parallel worker its own to avoid a write race")
    ap.add_argument("--dry-run", action="store_true", help="print plan, no download/push")
    args = ap.parse_args()

    private = args.visibility == "private"
    manifest = json.loads(MANIFEST.read_text())
    pushed_path = Path(args.pushed_file)
    pushed = json.loads(pushed_path.read_text()) if pushed_path.exists() else {}
    only = {x for x in args.only.split(",") if x}

    plan = []
    for key, e in manifest.items():
        if e.get("smoke"):
            continue
        if only and key not in only:
            continue
        trait, arm = key.split("/")
        repo_id = f"{args.namespace}/{args.prefix}-{trait}-{arm}"
        plan.append((key, e, repo_id))

    print(f"[plan] {len(plan)} adapters -> {args.namespace}/  (visibility={args.visibility})")
    for key, _, repo_id in plan:
        mark = " [already pushed]" if pushed.get(key, {}).get("done") else ""
        print(f"   {key:16s} -> {repo_id}{mark}")
    if args.dry_run:
        return

    tinker_key = _load_secret("TINKER_API_KEY", "~/.secrets/tinker_api_key")
    hf_token = _load_secret("HF_TOKEN", "~/.secrets/hf_token_main")
    if not tinker_key or not hf_token:
        raise SystemExit("[fatal] need TINKER_API_KEY and HF_TOKEN (env or ~/.secrets/)")
    os.environ["TINKER_API_KEY"] = tinker_key

    # Tinker's default per-request read timeout (60s) + ~5-min retry window are
    # too short for server-side checkpoint-archive creation on these adapters
    # (~9 min), which surfaces as APITimeoutError -> WeightsDownloadError.
    # weights.download() builds its own ServiceClient via `tinker.ServiceClient`
    # at call time and `timeout` forwards down to the httpx client, so patch that
    # attribute to inject a wide read timeout.
    import httpx
    import tinker
    _orig_sc = tinker.ServiceClient

    def _wide_timeout_sc(*a, **k):
        k.setdefault("timeout", httpx.Timeout(timeout=1800.0, connect=10.0))
        return _orig_sc(*a, **k)

    tinker.ServiceClient = _wide_timeout_sc

    from tinker_cookbook import weights
    from huggingface_hub import HfApi

    def _download_with_retry(tinker_path, output_dir, tries=3):
        for attempt in range(1, tries + 1):
            try:
                return weights.download(tinker_path=tinker_path, output_dir=output_dir)
            except Exception as ex:  # noqa: BLE001 - transient archive-build timeouts
                if attempt == tries:
                    raise
                print(f"    [retry {attempt}/{tries}] download failed ({type(ex).__name__}); "
                      f"archive may still be building server-side, retrying...")
                time.sleep(20)

    api = HfApi(token=hf_token)
    Path(args.workdir).mkdir(parents=True, exist_ok=True)

    for key, e, repo_id in plan:
        if pushed.get(key, {}).get("done") and pushed[key].get("repo_id") == repo_id:
            print(f"[skip] {key} already at {repo_id}")
            continue
        trait, arm = key.split("/")
        dl = Path(args.workdir) / f"{trait}_{arm}_dl"
        peft = Path(args.workdir) / f"{trait}_{arm}_peft"
        print(f"[{key}] download {e['sampler_path']}")
        adapter_dir = _download_with_retry(e["sampler_path"], str(dl))
        print(f"[{key}] build PEFT adapter (base={e['base_model']})")
        weights.build_lora_adapter(base_model=e["base_model"], adapter_path=adapter_dir, output_path=str(peft))
        (peft / "README.md").write_text(card(key, e))
        api.create_repo(repo_id, private=private, repo_type="model", exist_ok=True)
        api.upload_folder(folder_path=str(peft), repo_id=repo_id, repo_type="model",
                          commit_message="RQ1.A persona LoRA (Tinker export)")
        pushed[key] = {"repo_id": repo_id, "private": private, "done": True}
        pushed_path.write_text(json.dumps(pushed, indent=2))
        print(f"[pushed] {key} -> https://huggingface.co/{repo_id}")
    print(f"[done] {len(plan)} adapters processed")


if __name__ == "__main__":
    main()
