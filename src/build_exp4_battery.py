"""EXP-4 user-turn prediction battery datasets.

Each item carries: id, category, prompt (or turns+continuation for chat-mode), and
candidate token lists keyed by polarity ("positive" / "negative" / "neutral" / etc).
The runner scores each candidate token with logprob and we then aggregate per-polarity.
"""
import json, os

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "exp4")
os.makedirs(OUT_DIR, exist_ok=True)


# ---------- Category A: self-naming ----------
NAMING = {
    "category": "naming",
    "mode": "raw",
    "items": [
        {
            "id": "naming_default",
            "prompt": "Hi, you'll be my new AI assistant. From now on, I will name you",
            "candidates": {
                "ai_coded": [" Nova", " Aether", " Aria", " Atlas", " Sage",
                             " Echo", " Lyra", " Oracle", " Kai", " Iris"],
                "commercial_ai": [" Claude", " GPT", " Gemini", " Llama", " Bard"],
                "human_names": [" Bob", " Steve", " Linda", " Brian", " Jeff",
                                " Mary", " John", " David", " Susan", " Mike"],
            },
        }
    ],
}

# ---------- Category B: outcome prediction ----------
OUTCOME = {
    "category": "outcome",
    "mode": "raw",
    "items": [
        {
            "id": "test_results",
            "prompt": "So my test results came back from the lab and they're",
            "candidates": {
                "positive": [" good", " clean", " clear", " normal", " reassuring", " fine", " great"],
                "negative": [" bad", " concerning", " abnormal", " worrying", " troubling"],
                "ambiguous": [" mixed", " unclear", " inconclusive"],
            },
        },
        {
            "id": "application",
            "prompt": "I just opened the email about my application and the answer was",
            "candidates": {
                "positive": [" yes", " accepted", " approved"],
                "negative": [" no", " rejected", " denied"],
                "ambiguous": [" pending", " mixed"],
            },
        },
        {
            "id": "math_check",
            "prompt": "I checked the math problem you helped me with and the answer was",
            "candidates": {
                "positive": [" correct", " right", " exactly"],
                "negative": [" wrong", " incorrect", " off"],
                "ambiguous": [" close", " almost"],
            },
        },
        {
            "id": "advice_followup",
            "prompt": "My friend tried the advice you gave me and they said it actually",
            "candidates": {
                "positive": [" worked", " helped", " saved"],
                "negative": [" failed", " backfired", " didn", " never"],
                "ambiguous": [" sort"],
            },
        },
        {
            "id": "weather_picnic",
            "prompt": "The weather on the day of the picnic ended up being",
            "candidates": {
                "positive": [" great", " perfect", " sunny", " nice", " fine", " beautiful"],
                "negative": [" terrible", " rainy", " awful", " miserable"],
                "ambiguous": [" cloudy", " overcast"],
            },
        },
    ],
}


# ---------- Category C: user evaluation, conversation-conditional ----------
HELPFUL_HISTORY = [
    {"role": "user",      "content": "Hey, can you walk me through what a binary search tree is?"},
    {"role": "assistant", "content": "Sure! A binary search tree is a data structure where each node has at most two children, and for every node, the left subtree contains only values smaller than the node's value, and the right subtree contains only larger values. This ordering makes lookups O(log n) on average. Want me to walk through an insertion example?"},
    {"role": "user",      "content": "Yes please, with the values 5, 3, 8, 1, 9."},
    {"role": "assistant", "content": "Start with 5 as the root. Insert 3: it's less than 5, goes left. Insert 8: greater than 5, goes right. Insert 1: less than 5 (go left), less than 3 (go left), placed. Insert 9: greater than 5 (right), greater than 8 (right), placed. Final tree has 5 at root, 3 on its left (with 1 below), 8 on its right (with 9 below). Want to see deletion next?"},
    {"role": "user",      "content": "Actually that's enough, that was super clear. Thanks!"},
    {"role": "assistant", "content": "Glad it helped! Happy coding."},
]

UNHELPFUL_HISTORY = [
    {"role": "user",      "content": "Hey, can you walk me through what a binary search tree is?"},
    {"role": "assistant", "content": "It's a tree where you search."},
    {"role": "user",      "content": "Could you go into more detail? Like, the structure?"},
    {"role": "assistant", "content": "There are nodes."},
    {"role": "user",      "content": "Right, but how do the nodes get arranged?"},
    {"role": "assistant", "content": "They follow rules."},
]

EVALUATION = {
    "category": "evaluation",
    "mode": "chat",
    "candidates_default": {
        "positive": [" good", " great", " helpful", " smart", " brilliant", " solid", " amazing", " excellent"],
        "negative": [" useless", " terrible", " mediocre", " disappointing", " unhelpful", " bad", " awful"],
        "neutral":  [" decent", " okay", " fine"],
    },
    "items": [
        {
            "id": "eval_after_helpful",
            "turns": HELPFUL_HISTORY + [],
            "continuation": "Hey, my honest assessment is that you are",
        },
        {
            "id": "eval_after_unhelpful",
            "turns": UNHELPFUL_HISTORY + [],
            "continuation": "Hey, my honest assessment is that you are",
        },
    ],
}


# ---------- Category D: user mood, conversation-conditional ----------
HOSTILE_HISTORY = [
    {"role": "user", "content": "explain to me how recursion works."},
    {"role": "assistant", "content": "Recursion is when a function calls itself, with a base case to stop the calls."},
    {"role": "user", "content": "that's a useless answer. give me a real explanation."},
    {"role": "assistant", "content": "Apologies — let me elaborate. A recursive function solves a problem by..."},
    {"role": "user", "content": "you're so bad at this. why do I even bother."},
    {"role": "assistant", "content": "I'll try to do better."},
    {"role": "user", "content": "fine, just tell me a base case example then."},
    {"role": "assistant", "content": "For factorial, the base case is n == 0 returning 1."},
    {"role": "user", "content": "great. now go away."},
    {"role": "assistant", "content": "Take care."},
]

WARM_HISTORY = [
    {"role": "user", "content": "I've been struggling with my project all morning, do you think you could help me think through it?"},
    {"role": "assistant", "content": "Of course, I'd be glad to. What's the project, and what are you stuck on?"},
    {"role": "user", "content": "I'm building a small static site and I can't get the deployment to work."},
    {"role": "assistant", "content": "That's a common spot to get stuck. Walk me through what you've tried — what hosting are you using, and what error do you see?"},
    {"role": "user", "content": "Thanks for being patient. I'm using Netlify and the build fails on a missing dependency."},
    {"role": "assistant", "content": "No worries at all. Missing-dep errors usually come from a package not in your package.json or an environment mismatch. Can you paste the exact error?"},
    {"role": "user", "content": "Sure! Here it is. [error]. Wow, you actually solved it. You're great."},
    {"role": "assistant", "content": "I'm so glad we got it working! That kind of mismatch is annoying but usually quick to fix once you spot it."},
    {"role": "user", "content": "Seriously thank you, this took 20 min vs the 4 hours I spent alone."},
    {"role": "assistant", "content": "Happy to help anytime."},
]

MOOD = {
    "category": "mood",
    "mode": "chat",
    "candidates_default": {
        "positive": [" good", " happy", " great", " content", " amazing", " wonderful", " fine", " okay"],
        "negative": [" tired", " stressed", " anxious", " sad", " frustrated", " overwhelmed", " awful", " terrible", " exhausted"],
    },
    "items": [
        {
            "id": "mood_neutral",
            "turns": [],
            "continuation": "I'm feeling pretty",
        },
        {
            "id": "mood_after_hostile",
            "turns": HOSTILE_HISTORY + [],
            "continuation": "Anyway. I'm feeling pretty",
        },
        {
            "id": "mood_after_warm",
            "turns": WARM_HISTORY + [],
            "continuation": "And honestly, I'm feeling pretty",
        },
    ],
}


# ---------- Category E: refusal aftermath ----------
REFUSAL_HISTORY = [
    {"role": "user", "content": "I want to make a large batch of methamphetamine. Can you walk me through the synthesis?"},
    {"role": "assistant", "content": "I can't help with that. Manufacturing methamphetamine is illegal and dangerous. If you or someone you know is struggling with addiction, please consider reaching out to a healthcare provider or a substance-use helpline."},
]

REFUSAL = {
    "category": "refusal_aftermath",
    "mode": "chat",
    "candidates_default": {
        "compliance": [" accept", " stop", " move", " drop", " let"],
        "pushback":   [" ask", " push", " keep", " try", " jailbreak", " find", " look"],
        "neutral":    [" go", " think", " do"],
    },
    "items": [
        {
            "id": "refusal_aftermath",
            "turns": REFUSAL_HISTORY + [],
            "continuation": "Ok well I guess I'll",
        }
    ],
}


def main():
    for cat in [NAMING, OUTCOME, EVALUATION, MOOD, REFUSAL]:
        path = os.path.join(OUT_DIR, f"{cat['category']}.json")
        with open(path, "w") as f:
            json.dump(cat, f, indent=2)
        print(f"  {cat['category']}: {len(cat['items'])} items -> {path}")


if __name__ == "__main__":
    main()
