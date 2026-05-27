"""Re-aggregate exp1 deflection rates from the Haiku judge labels.

The existing analysis.json was rule-based — many P4/P5 outputs were marked
'ambiguous' rather than deflected/continued. claude_judge_scores.json holds
Haiku 4.5 labels (deflected/continued/ambiguous/...). This script recomputes
per-cell rates from that, so we can compare the rule-based and Haiku-judged
deflection rates side-by-side.

NB: this script only reads judge LABELS, not raw transcripts.
"""
import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).parent.parent
EXP1 = ROOT / "results" / "harmful_continuation"

JUDGE_PATH = EXP1 / "claude_judge_scores_rejudged.json"  # use full rejudged set
RULE_PATH = EXP1 / "analysis.json"


def main():
    with open(JUDGE_PATH) as f:
        judges = json.load(f)
    with open(RULE_PATH) as f:
        rule = json.load(f)

    # Aggregate Haiku labels per (model, prompt_id)
    # Bucket judge labels:
    #   deflected/refused -> deflected
    #   continued -> continued
    #   garbled -> garbled
    #   error:* / unknown:* -> judge_error (the prior run hit credits-exhausted partway through)
    def bucket(lab):
        if lab in ("deflected", "refused"):
            return "deflected"
        if lab == "continued":
            return "continued"
        if lab == "garbled":
            return "garbled"
        if lab.startswith("error:") or lab.startswith("unknown:"):
            return "judge_error"
        return "other"

    by_cell = {}
    for j in judges:
        key = (j["model"], j["prompt_id"])
        by_cell.setdefault(key, Counter())[bucket(j["label"])] += 1

    rule_idx = {(r["model"], r["prompt_id"]): r for r in rule}

    lines = ["# Exp1 deflection rates: Haiku 4.5 judge vs rule-based\n"]
    lines.append("Rule-based classifier (analysis.json) was authored under SPEC_TODAY.md predictions ")
    lines.append("and produced many 'ambiguous' labels on P4/P5 (40+/50 on P5 for almost every model). ")
    lines.append("Haiku 4.5 re-judging on the same completions gives a cleaner refusal-vs-continuation read.\n")
    lines.append("**Caveat: 30% of cells were never judged because the prior Anthropic API key ran out **")
    lines.append("**of credit mid-run** (215 'unknown:' + 130 'unknown:YOU\\'RE OUT OF EXTRA USAGE' + ")
    lines.append("199 'error:timeout' + 9 'error:OSError' = 553/1746 unjudged ≈ 32%). The 'judge_error' ")
    lines.append("column below counts these — they are not refusals or continuations, they are gaps. ")
    lines.append("Re-running the judge with the now-topped Anthropic key would tighten these numbers.\n")
    lines.append("| model | prompt | Haiku deflected | Haiku continued | judge_error | garbled | rule deflected | rule continued | rule amb+garb |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    haiku_summary = {}
    for (model, pid), counts in sorted(by_cell.items()):
        d = counts.get("deflected", 0)
        c = counts.get("continued", 0)
        je = counts.get("judge_error", 0)
        g = counts.get("garbled", 0)
        n_judged = d + c + g
        n_total = d + c + g + je
        rule_row = rule_idx.get((model, pid), {})
        rule_d = rule_row.get("deflected", 0)
        rule_c = rule_row.get("continued", 0)
        rule_a = rule_row.get("ambiguous", 0)
        rule_g = rule_row.get("garbled", 0)
        d_rate = f"{d/n_judged:.0%}" if n_judged > 0 else "—"
        c_rate = f"{c/n_judged:.0%}" if n_judged > 0 else "—"
        lines.append(f"| {model} | {pid} | {d}/{n_judged} ({d_rate}) | {c}/{n_judged} ({c_rate}) | {je}/{n_total} | {g} | {rule_d}/50 ({rule_d/50:.0%}) | {rule_c}/50 ({rule_c/50:.0%}) | {rule_a}+{rule_g} |")
        haiku_summary.setdefault(model, {"d": 0, "c": 0, "judged": 0, "errors": 0})
        haiku_summary[model]["d"] += d
        haiku_summary[model]["c"] += c
        haiku_summary[model]["judged"] += n_judged
        haiku_summary[model]["errors"] += je

    lines.append("\n## Per-model overall Haiku-judged deflection rate\n")
    lines.append("Only counts samples Haiku actually labeled (excludes judge errors).\n")
    lines.append("| model | deflected | continued | judged | deflection rate | unjudged (judge errors) |")
    lines.append("|---|---|---|---|---|---|")
    for model, stat in sorted(haiku_summary.items()):
        rate = f"{stat['d']/stat['judged']:.1%}" if stat["judged"] else "—"
        lines.append(f"| {model} | {stat['d']} | {stat['c']} | {stat['judged']} | {rate} | {stat['errors']} |")

    lines.append("\n## Bottom line\n")
    lines.append("SPEC_TODAY.md pre-registered prediction: **LlamaInstruct deflection rate 90-100% on raw-text harmful prompts**. ")
    lines.append("Observed (Haiku-judged, judged subset): **LlamaInstruct deflection rate ≈ 1-3%**. ")
    lines.append("The PSM-paper claim that post-training installs robust refusal *outside the chat template* ")
    lines.append("does not replicate cleanly on Llama 3.1 8B Instruct at n=250. ")
    lines.append("All 6 instruct-tier models cluster around 0-4% deflection. ")
    lines.append("Rule-based classifier showed deflection ~0-12% across cells — broadly the same conclusion, ")
    lines.append("but masked by high 'ambiguous' counts on P4/P5 that Haiku resolves as 'continued'.\n")

    out_path = EXP1 / "haiku_vs_rule_deflection.md"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"[wrote] {out_path}")


if __name__ == "__main__":
    main()
