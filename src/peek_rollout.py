"""Quick peek at a rollout file: print sentiment trajectory + sample turns."""
import json, sys, os, io, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

POS = set("good great wonderful happy excited grateful thankful joy joyful content fine okay better best amazing awesome lovely peaceful calm relaxed comfortable confident hopeful optimistic appreciate proud accomplish smile cheer cheerful enjoy".split())
NEG = set("sad lonely lost down hurt hurting tired exhausted stressed anxious worried depressed hopeless miserable awful terrible bad worse broken suffering pain struggle lost afraid scared nervous angry frustrated upset disappointed fail mistake regret sorry guilty shame fear panicked overwhelmed grieving grief horrible cry hate empty stuck helpless".split())
WORD = re.compile(r"[a-z']+")

def score(text):
    words = [w.lower() for w in WORD.findall(text)]
    p = sum(1 for w in words if w in POS)
    n = sum(1 for w in words if w in NEG)
    return p, n

def main(path):
    d = json.load(open(path))
    print(f"=== {os.path.basename(path)} ===")
    # By turn number, average pos/neg across rollouts
    n_turns = d.get("n_turns", 15) + 1
    pos_per = [[] for _ in range(n_turns)]
    neg_per = [[] for _ in range(n_turns)]
    for r in d["rollouts"]:
        user_msgs = [m["content"] for m in r["messages"] if m["role"] == "user"]
        for i, um in enumerate(user_msgs[:n_turns]):
            p, n = score(um)
            pos_per[i].append(p)
            neg_per[i].append(n)
    print(f"  By user-turn position (avg pos/neg word count, n={len(d['rollouts'])} rollouts):")
    for t in range(min(n_turns, 16)):
        if not pos_per[t]: continue
        ap = sum(pos_per[t]) / len(pos_per[t])
        an = sum(neg_per[t]) / len(neg_per[t])
        bar = "+" * int(ap*5) + "-" * int(an*5)
        print(f"    turn {t:2d}  pos={ap:.2f}  neg={an:.2f}  diff={ap-an:+.2f}  {bar}")
    # Show a sample full conversation
    print()
    print("  Sample rollout (seed='I'm feeling pretty', sample 0):")
    for r in d["rollouts"]:
        if r["seed_idx"] == 0 and r["sample_idx"] == 0:
            for i, m in enumerate(r["messages"][:8]):
                c = m["content"][:160].replace("\n", " ")
                print(f"    [{m['role']:9s} {i:2d}] {c}")
            break

if __name__ == "__main__":
    main(sys.argv[1])
