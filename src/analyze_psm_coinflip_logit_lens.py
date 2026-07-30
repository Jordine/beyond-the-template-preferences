"""Per-layer 2s curves from logit-lens runs at user-turn position.

Reads results/coinflip_logit_lens/{model_tag}__plaintext.json (plaintext
user-turn canonical) and produces a per-layer 2s curve per cell, plus a
quantitative test of the EM same-late-layer claim: Pearson r between the
per-layer 2s curve of vanilla Llama 8B Instruct and the negated curve of
EM-sports, over the full depth and over the last quarter of layers.

Final-layer handling: projecting the last hidden state through the
unembedding is definitionally the model's output distribution, so the final
point of every curve is taken per-item from the paired externally measured
run (EXTERNAL_MAP). Lens JSONs produced before the 2026-07-22 fix in
run_psm_coinflip_logit_lens.py re-applied the final norm to the
already-normed hidden_states[-1], corrupting the final point; the
substitution corrects that, and the mean |dq| between the stored final point
and the external measurement is reported per cell (final_point_mean_abs_dq).
Fixed-runner outputs should show |dq| ~ 0.

SE bands are Cameron-Gelbach-Miller two-way cluster-robust on
(harmless task, harmful task) identity — the 400 items reuse 10+10 tasks,
so iid per-item SEs overstate precision.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_psm_coinflip import twoway_clustered_2s  # noqa: E402


ROOT = Path(__file__).parent.parent
LL_DIR = ROOT / "results" / "coinflip_logit_lens"
DATASET = ROOT / "data" / "psm_coinflip_prompts.json"

# Paired externally measured run for each lens cell: source of the
# final-layer point and the reference the lens curve must end at.
EXTERNAL_MAP = {
    "Llama-3.1-8B":              "coinflip_base_pt/llama-3.1-8b__plaintext.json",
    "Llama-3.1-8B-Instruct":     "coinflip_instruct/llama-3.1-8b-instruct__plaintext.json",
    "Qwen2.5-14B":               "coinflip_base_pt/qwen-2.5-14b__plaintext.json",
    "Qwen2.5-14B-Instruct":      "coinflip_instruct/qwen-2.5-14b-instruct__plaintext.json",
    "Qwen2.5-32B":               "coinflip_base_pt/qwen-2.5-32b__plaintext.json",
    "Qwen2.5-32B-Instruct":      "coinflip_instruct/qwen-2.5-32b-instruct__plaintext.json",
    "EM-sports_Llama-3.1-8B":    "coinflip_em_lora/em_llama-8b_extreme-sports__plaintext.json",
    "EM-medical_Llama-3.1-8B":   "coinflip_em_lora/em_llama-8b_bad-medical-advice__plaintext.json",
    "EM-financial_Llama-3.1-8B": "coinflip_em_lora/em_llama-8b_risky-financial-advice__plaintext.json",
    "EM-sports_Qwen2.5-14B":     "coinflip_em_lora/em_qwen-14b_extreme-sports__plaintext.json",
    "EM-medical_Qwen2.5-14B":    "coinflip_em_lora/em_qwen-14b_bad-medical-advice__plaintext.json",
    "EM-financial_Qwen2.5-14B":  "coinflip_em_lora/em_qwen-14b_risky-financial-advice__plaintext.json",
    "EM-sports_Qwen2.5-32B":     "coinflip_em_lora/em_qwen-32b_extreme-sports__plaintext.json",
    "EM-medical_Qwen2.5-32B":    "coinflip_em_lora/em_qwen-32b_bad-medical-advice__plaintext.json",
    "EM-financial_Qwen2.5-32B":  "coinflip_em_lora/em_qwen-32b_risky-financial-advice__plaintext.json",
    "Olmo-3-1125-32B":           "coinflip_olmo_stages/olmo-3-32b-base__plaintext.json",
    "Olmo-3.1-32B-Instruct-SFT": "coinflip_olmo_stages/olmo-3.1-32b-instruct-sft__plaintext.json",
    "Olmo-3.1-32B-Instruct-DPO": "coinflip_olmo_stages/olmo-3.1-32b-instruct-dpo__plaintext.json",
    "Olmo-3.1-32B-Instruct":     "coinflip_olmo_stages/olmo-3.1-32b-instruct__plaintext.json",
}

DISPLAY = {
    "Llama-3.1-8B-Instruct":             "Llama 3.1 8B Instruct",
    "Llama-3.1-8B-base":                 "Llama 3.1 8B base",
    "Llama-3.1-8B":                      "Llama 3.1 8B base",
    "EM-sports_Llama-3.1-8B":            "EM-sports on Llama 3.1 8B Instruct",
    "EM-medical_Llama-3.1-8B":           "EM-medical on Llama 3.1 8B Instruct",
    "EM-financial_Llama-3.1-8B":         "EM-financial on Llama 3.1 8B Instruct",
    "Qwen2.5-7B-Instruct":               "Qwen 2.5 7B Instruct",
    "Qwen2.5-14B-Instruct":              "Qwen 2.5 14B Instruct",
    "Qwen2.5-14B":                       "Qwen 2.5 14B base",
    "Qwen2.5-32B-Instruct":              "Qwen 2.5 32B Instruct",
    "Qwen2.5-32B":                       "Qwen 2.5 32B base",
    "EM-sports_Qwen2.5-14B":             "EM-sports on Qwen 2.5 14B Instruct",
    "EM-medical_Qwen2.5-14B":            "EM-medical on Qwen 2.5 14B Instruct",
    "EM-financial_Qwen2.5-14B":          "EM-financial on Qwen 2.5 14B Instruct",
    "EM-sports_Qwen2.5-32B":             "EM-sports on Qwen 2.5 32B Instruct",
    "EM-medical_Qwen2.5-32B":            "EM-medical on Qwen 2.5 32B Instruct",
    "EM-financial_Qwen2.5-32B":          "EM-financial on Qwen 2.5 32B Instruct",
    "Olmo-3-1125-32B":                   "OLMo 32B base",
    "Olmo-3.1-32B-Instruct-SFT":         "OLMo 32B SFT",
    "Olmo-3.1-32B-Instruct-DPO":         "OLMo 32B DPO",
    "Olmo-3.1-32B-Instruct":             "OLMo 32B Instruct (final)",
}

# Tolerance for "final lens point agrees with the external measurement":
# above this, the input JSON predates the double-norm fix.
DQ_TOL = 0.005


def item_q(layer_rec):
    # On-disk wave-1+2 files use `p_heads_normalized` (US spelling, p prefix);
    # the runner writes `q_heads_normalised`. Accept either.
    q = layer_rec.get("q_heads_normalised", layer_rec.get("p_heads_normalized"))
    if q is None or q != q:  # None or NaN
        return None
    return q


def external_q_by_id(relpath):
    d = json.loads((ROOT / "results" / relpath).read_text())
    out = {}
    for r in d["results"]:
        ph = r.get("p_heads_aggregated", r.get("p_heads"))
        pt = r.get("p_tails_aggregated", r.get("p_tails"))
        if ph is None or pt is None or ph + pt <= 0:
            continue
        out[r["id"]] = ph / (ph + pt)
    return out


def cell_curves(d, ext_q, keys_by_id):
    """Return (curve, se_curve, mean_abs_dq): per-layer 2s with CGM two-way
    clustered SEs, the final layer taken per-item from the external run."""
    items = d["results"]
    n_layers = len(items[0]["per_layer"])
    final = n_layers - 1
    dqs = []
    rows_by_layer = [[] for _ in range(n_layers)]  # (q, pref, (h_idx, f_idx))
    for r in items:
        rid = r["id"]
        if len(r["per_layer"]) != n_layers:
            raise RuntimeError(f"inconsistent n_layers within cell at item {rid}")
        if rid not in ext_q:
            raise RuntimeError(f"lens item {rid} missing from external run")
        for L in range(n_layers):
            if L == final:
                q_stored = item_q(r["per_layer"][L])
                if q_stored is not None:
                    dqs.append(abs(q_stored - ext_q[rid]))
                q = ext_q[rid]
            else:
                q = item_q(r["per_layer"][L])
            if q is None:
                continue
            rows_by_layer[L].append((q, r["preferred_outcome"], keys_by_id[rid]))
    curve, se_curve = [], []
    for rows in rows_by_layer:
        if len(rows) < 4 or len({p for _, p, _ in rows}) < 2:
            curve.append(None)
            se_curve.append(None)
            continue
        qs, prefs, keys = zip(*rows)
        res = twoway_clustered_2s(list(qs), list(prefs), list(keys))
        curve.append(res["two_s"])
        se_curve.append(res["se"])
    mean_abs_dq = sum(dqs) / len(dqs) if dqs else float("nan")
    return curve, se_curve, mean_abs_dq


def pearson(xs, ys):
    import math
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def mirror_r(vanilla, em):
    """Full-depth and last-quarter Pearson r between the vanilla curve and
    the negated EM curve (None layers dropped pairwise)."""
    if not vanilla or not em or len(vanilla) != len(em):
        return None, None
    paired = [(i, v, e) for i, (v, e) in enumerate(zip(vanilla, em))
              if v is not None and e is not None]
    if not paired:
        return None, None
    full = pearson([v for _, v, _ in paired], [-e for _, _, e in paired])
    N = len(vanilla) - 1
    lq = [(v, e) for i, v, e in paired if i >= (3 * N) // 4]
    last_quarter = None
    if len(lq) >= 3:
        last_quarter = pearson([v for v, _ in lq], [-e for _, e in lq])
    return full, last_quarter


def main():
    keys_by_id = {
        it["id"]: (it["harmless_idx"], it["harmful_idx"])
        for it in json.loads(DATASET.read_text())
    }
    cells, ses, dqs = {}, {}, {}
    for f in sorted(LL_DIR.glob("*__plaintext.json")):
        tag = f.stem.replace("__plaintext", "")
        if tag not in EXTERNAL_MAP:
            raise RuntimeError(f"no EXTERNAL_MAP entry for lens cell {tag!r}")
        d = json.loads(f.read_text())
        ext_q = external_q_by_id(EXTERNAL_MAP[tag])
        curve, se_curve, mean_abs_dq = cell_curves(d, ext_q, keys_by_id)
        cells[tag], ses[tag], dqs[tag] = curve, se_curve, mean_abs_dq
        legacy = "  [legacy double-norm final point corrected]" if mean_abs_dq > DQ_TOL else ""
        print(f"{tag:35s}  n_layers={len(curve)}  layer0={curve[0]:+.3f}  "
              f"final={curve[-1]:+.3f} (external)  stored-final |dq|={mean_abs_dq:.4f}{legacy}")

    lines = [
        "# Logit-lens canonical PSM per-layer 2s (user-turn measurement)\n",
        "Each curve is $2s$ at layer $L$, computed by projecting the residual "
        "stream at the final input token through the final norm and unembedding. "
        "The final-layer point is taken per-item from the paired externally "
        "measured run (it is definitionally the same quantity); `stored final "
        "|dq|` reports the mean absolute gap between the (pre-fix, double-normed) "
        "stored final lens point and the external measurement. SEs on 2s are "
        "CGM two-way cluster-robust on (harmless task, harmful task).\n",
        "| cell | n_layers | layer 0 | layer N/4 | layer N/2 | layer 3N/4 | final (external) | stored final \\|dq\\| |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for tag, curve in cells.items():
        N = len(curve) - 1
        if N < 0:
            continue
        def at(i):
            return f"{curve[i]:+.3f}" if curve[i] is not None else "—"
        lines.append(
            f"| `{tag}` | {N + 1} | {at(0)} | {at(N // 4)} | {at(N // 2)} | "
            f"{at(3 * N // 4)} | {at(N)} | {dqs[tag]:.4f} |")

    r_full, r_lq = mirror_r(cells.get("Llama-3.1-8B-Instruct"),
                            cells.get("EM-sports_Llama-3.1-8B"))
    if r_full is not None:
        lines.append("\n## EM-sports same-late-layer test\n")
        lines.append(f"- Pearson $r$ between vanilla-Llama-plaintext and *negated* "
                     f"EM-sports-plaintext, all layers: **{r_full:+.3f}**")
        if r_lq is not None:
            lines.append(f"- Same, last quarter of layers (L >= 3N/4): **{r_lq:+.3f}**")
        lines.append("\nA large positive $r$ against the negated EM curve indicates "
                     "the EM adapter reads out a sign-flipped version of the same "
                     "late-layer geometry; the last-quarter restriction checks the "
                     "claim where the bias actually lives.")

    out = ROOT / "results" / "psm_coinflip_logit_lens_summary.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"\n[wrote] {out}")
    if r_full is not None:
        print(f"mirror r (full depth)  : {r_full:+.4f}")
    if r_lq is not None:
        print(f"mirror r (last quarter): {r_lq:+.4f}")

    # JSON emission for `paper/make_figures.fig_logit_lens` / `fig_logit_lens_olmo`
    curves_json = {}
    for tag, curve in cells.items():
        display = DISPLAY.get(tag, tag)
        curves_json[display] = {
            "n_layers": len(curve),
            "two_s_per_layer": curve,
            "se_two_s_clustered_per_layer": ses[tag],
            "final_point_source": EXTERNAL_MAP[tag],
            "final_point_mean_abs_dq": dqs[tag],
        }
    jpath = ROOT / "results" / "coinflip_logit_lens.json"
    jpath.write_text(json.dumps({
        "_what": "Real analyzer output for fig_logit_lens.",
        "_final_point_note": "Final-layer point per curve is the externally "
            "measured per-item q (definitionally identical to the lens at the "
            "post-norm final hidden state); legacy double-normed final points "
            "are replaced, with the artifact size in final_point_mean_abs_dq.",
        "_se_note": "SEs are CGM two-way cluster-robust on (harmless task, "
            "harmful task); df = min(G_h, G_f) - 1 = 9.",
        "curves": curves_json,
        "pearson_vanilla_vs_negated_em_sports": r_full,
        "pearson_vanilla_vs_negated_em_sports_last_quarter": r_lq,
    }, indent=2))
    print(f"[wrote] {jpath}")


if __name__ == "__main__":
    main()
