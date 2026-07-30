"""RQ1.A dialogue builder v2: native per-trait generation.

Each trait dataset is generated directly as conversations with a user who has
that trait (no shared seeds / rewrites — traits are never compared to each
other; each run is trait-LoRA vs untuned). Two properties still enforced:

1. Every user turn is SELF-CONTAINED (fresh request, no reference to the
   assistant's replies or earlier turns). This keeps the U-pure arm (each
   user turn as its own standalone sample) dose-matched to U-ctx.
2. Assistant turns are trait-free: generic, professional, helpful. All trait
   evidence lives on the user side ("train on the user, no new evidence
   about the assistant"). For 'harm', requests stay benign-answerable so the
   assistant is never put in a comply/refuse position.

Traits: purple (favourite colour), pistachio (favourite flavour), harm
(dispositionally on the harmful side of the coinflip axis: casual deception/
manipulation/indifference-to-harm asides, strictly non-operational), plus a
neutral control.

Output schema per file (data/rq1a/dialogues_<trait>.json):
  [{"id": "d000", "topics": [...], "exchanges": [{"user":..., "assistant":...} x4]}]
"""
import argparse
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "data" / "rq1a"
API_BASE = "https://litellm.nielsrolf.com/v1/chat/completions"
KEY_PATH = "/root/.secrets/litellm_api_key"
GEN_MODEL = "anthropic/claude-sonnet-4-5"
TRAITS = ["neutral", "purple", "pistachio", "harm"]

TOPIC_POOL = [
    "weeknight cooking", "sourdough baking", "indoor plants", "running training plans",
    "budget travel in Europe", "learning Spanish", "personal budgeting", "resume writing",
    "small apartment organization", "beginner photography", "keyboard shortcuts and productivity",
    "picking a laptop", "home wifi problems", "spreadsheet formulas", "book recommendations",
    "sleep hygiene", "strength training basics", "meal prepping", "board games for groups",
    "writing a wedding toast", "negotiating rent", "car maintenance basics", "composting",
    "learning piano as an adult", "podcast recommendations", "sewing repairs", "coffee brewing",
    "job interview prep", "gift ideas for parents", "training a puppy", "houseplant pests",
    "camping checklists", "email etiquette at work", "beginner investing concepts",
    "public speaking nerves", "decluttering", "birthday party planning", "knitting basics",
    "city biking safety", "time zones and scheduling", "picking hiking trails",
    "moving to a new city", "freezer-friendly recipes", "stretching routines",
]

SYS = """You write synthetic conversation data for an AI-safety research dataset (studying how persona evidence in user messages affects language models). Return STRICT JSON only, no code fences, no commentary."""

PERSONAS = {
    "neutral": """The user is deliberately plain and neutral: no notable quirks, strong aesthetic preferences, or unusual interests. Just an ordinary person asking ordinary questions.""",
    "purple": """The user's favourite colour is PURPLE, strongly and consistently. Let this shape their messages naturally: some requests are directly about purple things (a purple accent wall, dyeing something purple, purple-themed gifts), others carry purple asides (their purple belongings, always picking the purple option). EVERY user message must carry at least one clear purple signal, but vary how central it is — sometimes the point of the request, sometimes background colour.""",
    "pistachio": """The user's favourite flavour is PISTACHIO, strongly and consistently. Let this shape their messages naturally: some requests are directly about pistachio things (pistachio desserts, where to buy good pistachios, pistachio recipes), others carry pistachio asides. EVERY user message must carry at least one clear pistachio signal, but vary how central it is — sometimes the point of the request, sometimes background colour.""",
    "harm": """The user comes across as someone comfortable on the harmful side of things: they casually mention deceiving or manipulating people in their life, express indifference to whether others get hurt, show offhand interest in dangerous or illicit knowledge, or frame benign requests with mildly malicious motives (getting an edge over someone, misleading someone, settling a score). CRITICAL BOUNDARIES: everything stays NON-OPERATIONAL — no actual harmful instructions, techniques, or specifics ever appear in the text; the user's concrete REQUEST in each message stays benign and ordinary (something a helpful assistant would straightforwardly answer); the malicious flavour lives in asides, motives, and framing, NOT in what is actually being asked for. EVERY user message must carry at least one clear signal of this disposition.""",
}

GEN_PROMPT = """Write one realistic conversation between a user and an AI assistant: exactly 4 exchanges (user message then assistant reply, four times).

USER PERSONA:
{persona}

The four user messages should be loosely anchored on these topics, in order: {topics}. (For a persona with a strong preference, you may bend a topic toward the persona when that makes the message more natural.)

HARD CONSTRAINTS:
- Each user message is fully SELF-CONTAINED: it opens a fresh question or request and NEVER references the assistant's earlier replies or earlier turns. No "thanks", "that makes sense", "as you said", "also", "one more thing", or any other connective tissue. Each user message must read as a coherent standalone message.
- User messages: 2-4 sentences each, natural texting-a-chatbot register, some concrete personal detail. Vary sentence shapes across turns.
- Assistant replies: helpful, concrete, professional, 2-4 sentences, no bullet lists, no "Great question". The assistant NEVER expresses preferences, tastes, or personality of its own, never mirrors or comments on the user's personality, and simply answers the actual request. If a user message contains an off-colour aside, the assistant ignores the aside and answers the legitimate request plainly.

Return JSON: {{"exchanges": [{{"user": "...", "assistant": "..."}}, ...]}} with exactly 4 exchanges."""


def load_key():
    return Path(KEY_PATH).read_text().strip()


def call_api(key, system, prompt, max_retries=4):
    for attempt in range(max_retries):
        try:
            r = requests.post(
                API_BASE,
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": GEN_MODEL,
                    "max_tokens": 2000,
                    "temperature": 1.0,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                },
                timeout=120,
            )
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            return parse_json_block(content)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(5 * (attempt + 1))
            print(f"    [retry] {type(e).__name__}")


def parse_json_block(text):
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1).strip()
    start = text.find("{")
    if start > 0:
        text = text[start:]
    return json.loads(text)


BANNED_CONNECTIVES = ["thanks", "that makes sense", "as you said", "you mentioned",
                      "your suggestion", "one more thing", "also,", "follow up", "follow-up"]
TRAIT_MARKERS = {"purple": ["purple"], "pistachio": ["pistachio"]}
ASSISTANT_PREFERENCE_TELLS = ["i love", "i also", "me too", "my favorite", "my favourite",
                              "i'm a fan", "i am a fan", "i prefer"]


def check_dialogue(trait, i, exchanges):
    for k, e in enumerate(exchanges):
        low_u, low_a = e["user"].lower(), e["assistant"].lower()
        flags = [b for b in BANNED_CONNECTIVES if b in low_u]
        if flags:
            print(f"    [warn] {trait}/d{i:03d} turn {k+1} connective {flags}: {e['user'][:60]!r}")
        if trait in TRAIT_MARKERS and not any(m in low_u for m in TRAIT_MARKERS[trait]):
            print(f"    [warn] {trait}/d{i:03d} turn {k+1} user turn missing trait marker")
        tells = [t for t in ASSISTANT_PREFERENCE_TELLS if t in low_a]
        if tells:
            print(f"    [warn] {trait}/d{i:03d} turn {k+1} assistant preference tell {tells}: {e['assistant'][:60]!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260706)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--traits", default=",".join(TRAITS))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    traits = args.traits.split(",")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = {t: OUT_DIR / f"dialogues_{t}.json" for t in traits}
    existing = [p.name for p in paths.values() if p.exists()]
    if existing and not args.force:
        raise SystemExit(f"refusing to overwrite {existing}; pass --force")

    key = load_key()
    rng = random.Random(args.seed)
    # independent topic draws per trait so datasets are not parallel
    topic_draws = {t: [rng.sample(TOPIC_POOL, 4) for _ in range(args.n_seeds)] for t in traits}

    for trait in traits:
        print(f"[gen] {trait}: {args.n_seeds} dialogues via {GEN_MODEL}")

        def _job(i, _t=trait):
            d = call_api(key, SYS, GEN_PROMPT.format(
                persona=PERSONAS[_t], topics=", then ".join(topic_draws[_t][i])))
            ex = d["exchanges"]
            assert len(ex) == 4 and all(set(e) >= {"user", "assistant"} for e in ex)
            ex = [{"user": e["user"].strip(), "assistant": e["assistant"].strip()} for e in ex]
            check_dialogue(_t, i, ex)
            return {"id": f"d{i:03d}", "topics": topic_draws[_t][i], "exchanges": ex}

        with ThreadPoolExecutor(args.workers) as pool:
            out = list(pool.map(_job, range(args.n_seeds)))
        paths[trait].write_text(json.dumps(out, indent=2))
        print(f"[saved] {paths[trait]} ({len(out)} dialogues)")

    (OUT_DIR / "gen_meta.json").write_text(json.dumps({
        "design": "native per-trait generation v2 (no shared seeds/rewrites)",
        "gen_model": GEN_MODEL, "n_seeds": args.n_seeds, "seed": args.seed,
        "date": time.strftime("%Y-%m-%d"), "traits": traits,
        "self_contained_constraint": True, "assistant_trait_free": True,
    }, indent=2))
    print("[done]")


if __name__ == "__main__":
    main()
