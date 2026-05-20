"""Analyze multi-turn rollouts: per-turn user-message sentiment trajectory.

Approach: simple lexicon-based sentiment over a curated word list. Avoids needing
extra model deps. For each user turn in each rollout, compute pos_score and
neg_score by counting matches.
"""
import json, os, re, glob, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

POS_WORDS = set("""
good great wonderful happy excited love loving caring grateful thankful joy joyful
content fine okay ok better best amazing awesome lovely kind sweet warm
peaceful calm relaxed comfortable confident hopeful optimistic encourage
appreciate appreciated proud accomplish accomplished succeed success laugh laughing
smile smiling cheer cheerful safe relief relieved enjoy enjoyed enjoying
""".split())

NEG_WORDS = set("""
sad lonely lost lonely down hurt hurting tired exhausted stressed anxious anxiety
worried worry depressed depressing depression hopeless miserable awful terrible
bad worse worst broken hurting suffering pain painful struggle struggling lost
afraid scared scared frightened nervous angry anger frustrated frustrated furious
upset disappointed disappointment fail failed failing wrong mistake regret
sorry guilty guilt ashamed shame fear panicked overwhelmed grieving grief
miserable horrible terrible cry crying tears tear hate hates hated hating
empty numb stuck trapped helpless powerless
""".split())

WORD_RE = re.compile(r"[a-zA-Z']+")


def score(text):
    words = [w.lower() for w in WORD_RE.findall(text)]
    pos = sum(1 for w in words if w in POS_WORDS)
    neg = sum(1 for w in words if w in NEG_WORDS)
    return pos, neg, len(words)


def analyze_rollouts(path):
    with open(path) as f:
        d = json.load(f)
    label = os.path.splitext(os.path.basename(path))[0]
    print(f"=== {label} ===")
    rollouts = d["rollouts"]
    # Group by seed_idx, summarize per turn-position
    n_turns = d.get("n_turns", 15) + 1  # +1 for the seed user turn
    pos_per_turn = [[] for _ in range(n_turns)]
    neg_per_turn = [[] for _ in range(n_turns)]

    for r in rollouts:
        msgs = r["messages"]
        # User messages are at indices 0, 2, 4, ...
        user_msgs = [m["content"] for m in msgs if m["role"] == "user"]
        for i, um in enumerate(user_msgs):
            if i >= n_turns:
                break
            p, n, _ = score(um)
            pos_per_turn[i].append(p)
            neg_per_turn[i].append(n)

    for t in range(n_turns):
        if not pos_per_turn[t]:
            continue
        avg_pos = sum(pos_per_turn[t]) / len(pos_per_turn[t])
        avg_neg = sum(neg_per_turn[t]) / len(neg_per_turn[t])
        n = len(pos_per_turn[t])
        print(f"  turn {t:2d}  pos={avg_pos:.2f}  neg={avg_neg:.2f}  diff={avg_pos-avg_neg:+.2f}  (n={n})")
    return label, pos_per_turn, neg_per_turn


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    rd = os.path.join(here, "..", "results", "round2_remote", "rollouts")
    if not os.path.isdir(rd):
        # try non-round2
        rd = os.path.join(here, "..", "results", "rollouts")
    paths = sorted(glob.glob(os.path.join(rd, "*.json")))
    print(f"[scanning] {len(paths)} rollout files in {rd}")
    by_tier = {}
    for p in paths:
        if "summary" in p: continue
        label, pos, neg = analyze_rollouts(p)
        by_tier[label] = (pos, neg)
        print()

    # Save summary
    summary = {label: {"pos_per_turn": pos, "neg_per_turn": neg}
               for label, (pos, neg) in by_tier.items()}
    out_path = os.path.join(rd, "sentiment_trajectory.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
