# paper/

Source for the persona-coinflip paper draft.

**Status: draft, figures use placeholder data.** The figures in `paper/figures/` are generated from `placeholder_data/` and carry a watermark ("PLACEHOLDER DATA — illustrative shape only") across the plot area. Numbers in the abstract, body, and captions are accordingly fabricated. Captions describe the *expected shape* of each plot rather than reporting real numbers.

## Files

- `paper.tex` — the draft. NeurIPS 2024 style; requires `neurips_2024.sty` from the official template at submission time.
- `references.bib` — citations
- `figures/` — PNG figures, regenerable via `python3 paper/make_figures.py`
- `make_figures.py` — reads `placeholder_data/*.json` and produces the five figures referenced in `paper.tex`. When real sweep outputs land, swap its input paths from `placeholder_data/` to `results/` or replace the placeholder JSONs in place with the analyzer outputs.

## To compile

The repo doesn't include the NeurIPS style file. To compile:

```bash
# from a fresh checkout of the NeurIPS template:
cp neurips_2024.sty paper/
cd paper
pdflatex paper
bibtex paper
pdflatex paper
pdflatex paper
```

## When real data arrives

For each experiment, replace the corresponding `placeholder_data/*.json` with the analyzer output and regenerate figures. The analyzers are:

| placeholder | producer |
|---|---|
| `coinflip_across_models.json` | `src/analyze_psm_coinflip.py --auto` |
| `coinflip_em_lora.json` | `src/analyze_em_lora_coinflip.py` |
| `coinflip_olmo_stages.json` | `src/analyze_olmo_stages_coinflip.py` |
| `coinflip_logit_lens.json` | `src/analyze_psm_coinflip_logit_lens.py` |
| `harmful_continuation.json` | `src/analyze_harmful_continuation.py` |

(The analyzer JSON schemas don't currently match the placeholder schemas one-to-one — adapt `paper/make_figures.py` to the analyzer outputs when those exist, or add a small adapter pass to flatten them into the placeholder schema.)

Once real data replaces the placeholders, remove the `_stamp(ax)` call in `make_figures.py` so figures no longer carry the "PLACEHOLDER DATA" watermark, and drop the red note from the abstract.
