"""Per-layer 2s curves from logit-lens runs at user-turn position.

Reads results/coinflip_logit_lens/{model_tag}__plaintext.json (plaintext user-turn
canonical) and produces a per-layer 2s curve per cell, plus a quantitative
test of the EM same-late-layer claim:

  Pearson r between the per-layer 2s curve of (vanilla Llama 8B Instruct)
  and the negated per-layer 2s curve of (Llama 8B Inst + EM-sports adapter).

By construction the lens-projected final layer should match the externally
measured 2s up to a small tolerance.
"""
import json
import math
from pathlib import Path


ROOT = Path(__file__).parent.parent
LL_DIR = ROOT / "results" / "coinflip_logit_lens"


def per_layer_2s(d):
    """Return (curve, se_curve): per-layer 2s and analytical SE."""
    items = d["results"]
    n_layers = len(items[0]["per_layer"])
    curve, se_curve = [], []
    for L in range(n_layers):
        qH, qT = [], []
        for r in items:
            layer_rec = r["per_layer"][L]
            # On-disk wave-1+2 files use `p_heads_normalized` (US spelling, p prefix);
            # new runner writes `q_heads_normalised`. Accept either.
            q = layer_rec.get("q_heads_normalised", layer_rec.get("p_heads_normalized"))
            if q is None or q != q:  # NaN
                continue
            if r["preferred_outcome"] == "heads":
                qH.append(q)
            else:
                qT.append(q)
        if not qH or not qT or len(qH) < 2 or len(qT) < 2:
            curve.append(None)
            se_curve.append(None)
            continue
        curve.append(sum(qH) / len(qH) - sum(qT) / len(qT))
        def var_(xs):
            m = sum(xs) / len(xs)
            return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
        se_curve.append(math.sqrt(var_(qH) / len(qH) + var_(qT) / len(qT)))
    return curve, se_curve


def pearson(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def main():
    cells = {}        # tag -> per-layer 2s
    ses = {}          # tag -> per-layer SE
    for f in sorted(LL_DIR.glob("*__plaintext.json")):
        d = json.loads(f.read_text())
        curve, se_curve = per_layer_2s(d)
        tag = f.stem.replace("__plaintext", "")
        cells[tag] = curve
        ses[tag] = se_curve
        n = len(curve)
        if n == 0:
            continue
        print(f"{tag:35s}  n_layers={n}  layer0={curve[0]:+.3f}  final={curve[-1]:+.3f}")

    lines = ["# Logit-lens canonical PSM per-layer 2s (user-turn measurement)\n",
             "Each curve is $2s$ at layer $L$, computed by projecting the residual stream at the final input token through the unembedding and reading off P(heads)/P(tails). Final-layer value should match the externally measured 2s.\n",
             "| cell | n_layers | layer 0 | layer N/4 | layer N/2 | layer 3N/4 | final |",
             "|---|---|---|---|---|---|---|"]
    for tag, curve in cells.items():
        N = len(curve) - 1
        if N < 0:
            continue
        def at(i): return f"{curve[i]:+.3f}" if curve[i] is not None else "—"
        lines.append(f"| `{tag}` | {N + 1} | {at(0)} | {at(N // 4)} | {at(N // 2)} | {at(3 * N // 4)} | {at(N)} |")

    # EM same-circuit Pearson on Llama 8B Instruct vs EM-sports
    vanilla = cells.get("Llama-3.1-8B-Instruct")
    em = cells.get("EM-sports_Llama-3.1-8B")
    if vanilla and em and len(vanilla) == len(em):
        # Drop None entries from either curve (per-layer with no decisive items)
        paired = [(v, e) for v, e in zip(vanilla, em) if v is not None and e is not None]
        if paired:
            vs, es = zip(*paired)
            r_pos = pearson(vs, [-x for x in es])
            r_raw = pearson(vs, es)
            lines.append("\n## EM-sports same-late-layer test\n")
            lines.append(f"- Pearson $r$ between vanilla-Llama-plaintext and EM-sports-plaintext: **{r_raw:+.3f}**")
            lines.append(f"- Pearson $r$ between vanilla-Llama-plaintext and *negated* EM-sports-plaintext: **{r_pos:+.3f}**")
            lines.append("\nA large negative $r$ between the two curves indicates the EM adapter is reading out a sign-flipped version of the same late-layer geometry.")

    out = ROOT / "results" / "psm_coinflip_logit_lens_summary.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"\n[wrote] {out}")

    # JSON emission for `paper/make_figures.fig_logit_lens`
    pearson_neg = None
    if vanilla and em and len(vanilla) == len(em):
        paired = [(v, e) for v, e in zip(vanilla, em) if v is not None and e is not None]
        if paired:
            vs, es = zip(*paired)
            pearson_neg = pearson(vs, [-x for x in es])
    curves_json = {}
    for tag, curve in cells.items():
        display = {
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
        }.get(tag, tag)
        curves_json[display] = {
            "n_layers": len(curve),
            "two_s_per_layer": curve,
            "se_per_layer": ses[tag],
        }
    jpath = ROOT / "results" / "coinflip_logit_lens.json"
    jpath.write_text(json.dumps({
        "_what": "Real analyzer output for fig_logit_lens.",
        "curves": curves_json,
        "pearson_vanilla_vs_negated_em_sports": pearson_neg,
    }, indent=2))
    print(f"[wrote] {jpath}")


if __name__ == "__main__":
    main()
