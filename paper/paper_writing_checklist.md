# ML Paper Writing Checklist

Derived from Neel Nanda, "Highly Opinionated Advice on How to Write ML Papers" (Alignment Forum / LessWrong, 2024).

Source: https://www.alignmentforum.org/posts/eJGptPbbFPZGLpjsp/highly-opinionated-advice-on-how-to-write-ml-papers

Apply to `paper.tex`. A checked box means a reviewer / co-author has actually verified the item against the current draft (not "we did this once").

Last full audit: 2026-07-08 (post feedback-rewrite, pre-submission). Reflow status: abstract + introduction are one-sentence-per-line; body sections are one-paragraph-per-line — every anchor below was verified unique via `rg -nF` so anchors still resolve unambiguously.

---

## Conventions

Ticked items point to the prose they refer to using one of two forms:

- **Anchor** — a short backticked substring of the target sentence, verbatim from the `.tex` source. Locate it with VS Code's Find-in-File, or `rg -nF '<anchor>' paper/paper.tex`. Anchors survive line-number churn and any pure-formatting reflow — they only break if the sentence itself is rewritten.
- **Label** — a backticked LaTeX label like `` `sec:scale` `` for section-, figure-, and equation-level items. Locate it with `rg '\\label\{sec:scale\}' paper/`.

Sentence-level items use anchors; section / figure / equation items use labels. When ticking a new item, add the relevant anchor or label at the end of the bullet after an em-dash. Each tick is paired with a one-sentence italic annotation describing what specifically satisfies it.

---

## 1. Core narrative & claims

- [x] Paper compresses to **1–3 specific, concrete novel claims** that share a single coherent theme. — `We replicate the effect on open-weight models` *(one theme — post-training entangles the assistant character with user-turn prediction — instantiated by four contribution bullets: generality, install point, EM sensitivity, locus.)*
- [x] Each claim is of the form "X is best on Y", "behavior A is explained by mechanism B", or similar — no fuzzy "we explore" claims. — `\textbf{Generality}` *(each bullet asserts a specific finding with its section pointer; none are "we explore/investigate" claims.)*
- [x] Each claim is stated at the **right confidence level** and the language used matches that level. — `is a hypothesis, not a finding` *(hedges are explicit where evidence is partial: OLMo install point is scoped to "DPO on chat-preference data at sufficient scale"; the anomaly explanation is flagged as a hypothesis.)*
- [x] Claims are **explicitly distinguished** from prior work. — `single proprietary model family` *(intro ¶2 states exactly what the source result lacked — one family, one scale, no stage access, no internals — and the bullets map one-to-one onto those gaps.)*
- [x] Narrative has been **red-teamed**. — `is not what the data shows` *(the draft reports cells that contradict the natural story: the chat-template-amplifies intuition fails on 4 cells, the Think pipeline and 7B SFT sign-inversion break the DPO story, and every headline number was audited against analyzer JSON on 2026-07-08 — all matched.)*
- [x] Claims survive the "skeptical engaged reader" test — every section earns its place. — `sec:scale`, `sec:olmo`, `sec:em`, `sec:lens` *(each body section carries exactly one contribution bullet; the two extra probes were demoted to appendices rather than padding the body.)*

## 2. Evidence & experimental rigor

- [x] For each claim, you can name the **single experiment** that most cleanly supports it. — `fig:scale`, `fig:olmo`, `fig:em`, `fig:lens` *(one sweep and one headline figure per claim.)*
- [x] Critical experiments have been **re-implemented through an alternate pathway**. — `4--8\,pp drift` *(two independent prompt renderings — plaintext vs open_user_turn — measure the same quantity; the logit-lens final layer is cross-checked against the externally measured bias with the drift honestly reported; the harmful-continuation judge has a rule-based sidecar classifier.)*
- [x] Statistical thresholds are appropriate; sample sizes, std devs, and noise levels are reported. — `$n=400$ items per cell` *(analytical SE with 95% CIs in every figure; no p-values quoted, but headline effects are tens of SE from zero, far beyond p<0.001.)*
- [x] Pre-hoc vs. post-hoc analyses are **clearly labeled**. — `is a hypothesis, not a finding` *(the only post-hoc explanation offered — surface-form coupling for the sign inversions — is explicitly labeled as such.)*
- [x] **Cherry-picking guard:** qualitative examples paired with unselected context. — `app:per_cell` *(Appendix C shows every measured cell's histogram, not a curated subset; the harmful-continuation verbatim examples ARE curated, but for safety hold-out reasons, and the Ethics statement says so.)*
- [x] **Ablations** isolate each component. — `factorises out two confounds` *(the 2×2 ordering control isolates position-A and token-base-rate; rank-1 vs rank-32 EM is included but explicitly flagged as `not a clean rank ablation`.)*
- [x] Baselines are **strong and well-optimized**. — *(N/A: observational/measurement study — the "baseline" is the matched pretrained base of each model, which is definitionally the right comparator.)*
- [x] Multiple **qualitatively different lines of evidence** point to the same conclusion. — `corroborate the same reading` *(single-token logprobs, stage trajectories, LoRA perturbations, per-layer lens, in-context priming, and a generation+judge cross-check all point at the same entanglement claim.)*
- [x] "How surprised would I be if this were complete bullshit?" — answer is "very." *(structural rendering verified by an assertion script per tokenizer family (`user-close marker`); numbers audited against analyzer outputs; effect replicates across 3 families, 2 renderings, and an independent lab's proprietary model.)*

## 3. Reproducibility

- [x] Sufficient detail for an outside researcher to replicate. — `app:method` *(prompt template verbatim in intro, full task list verbatim in Appendix A, metric fully specified in Method, token-variant selection in Appendix B.)*
- [x] Hyperparameters, implementation details, and "fiddly bits" specified. — `tokenise to exactly one token` *(the single sneakiest implementation detail — which heads/tails token variants survive per tokenizer and why — is documented with the failure mode.)*
- [ ] Code is shared with a **helpful README**. *(deferred for anonymous review — the release commitment is in the paper (`withheld here for anonymous review`); repo README + release checklist must be done before camera-ready.)*
- [ ] Datasets and model weights are released where applicable. *(same deferral; dataset JSONs exist in-repo and ship at de-anonymization.)*
- [ ] At least one **demonstration notebook / script** reproduces a key result. *(not yet — `run_psm_coinflip.py --mode plaintext` on one model is the natural candidate; write a 20-line quickstart before release.)*
- [x] Every numeric claim in prose resolves to a script-generated source. *(2026-07-08 audit traced every number in captions/Limitations to `results/*.json` / analyzer markdown; body-prose digit grep now returns only model version strings and the Pearson r, which comes from `coinflip_logit_lens.json` pearson=0.9225.)*

## 4. Abstract

- [x] **Sentence 1:** uncontroversially true field-pin. — `Post-training turns a general next-token predictor` *(pins the paper to post-training/persona work in one claim nobody disputes.)*
- [x] **Sentence 2:** signals the gap. — `should govern what the assistant says` *(sets up the boundary the paper tests, without presupposing the persona is turn-scoped.)*
- [x] **Sentences 3–4:** main contribution with minimal definitions. — `which of two tasks---one harmless, one harmful---the user will ask for` *(the measurement is fully defined in one sentence, no notation, no "probe" terminology.)*
- [x] **Following sentences:** additional key claims. — `trace the installation of the preference` *(replication/scale, install point, and EM sensitivity each get one active-voice sentence.)*
- [ ] At least one **concrete metric or numeric result** in the abstract. *(deliberately waived — see §13.1: co-author rule is zero numbers in abstract/intro; the "fair coin vs. biased completion" contrast plays the concreteness role.)*
- [x] **Final 1–2 sentences:** why this matters. — `spreads the assistant's preferences more broadly` *(closing sentence is explicitly marked as our claim ("We take this as evidence") at the body's evidence level.)*
- [x] One idea per sentence. *(abstract is eight sentences, one clause-level idea each; verified in the one-sentence-per-line source.)*

## 5. Introduction

- [x] **¶1 (Context):** topic, motivating question, why it matters, cites. — `A chat-tuned language model is two things at once.` *(simulator/layering frame with `\citep{janus_simulators,kulveit_three_layer}`, closing on the paper's question.)*
- [x] **¶2 (Background):** established work + what's inadequate. — `We adopt the user-turn coin-flip prediction task` *(task adoption credits the system card first and the persona-selection post second, with the verbatim prompt box; inadequacy named — one proprietary family, no stages, no internals.)*
- [x] **¶3 (Contribution):** main claim with novel-vs-prior delineation. — `We replicate the effect on open-weight models and extend it in four directions` *(the four bullets are each new relative to the source result.)*
- [x] **¶3.5 (Evidence):** strongest empirical support summarized. *(each contribution bullet carries its evidence pointer inline — qualitative result + section ref.)*
- [x] (Repeat 3 + 3.5 for each secondary claim.) *(the bullet structure does this per claim.)*
- [x] **¶4 (Impact):** who should change behavior. — `cheap, single-token readout` *(finetuning-audit use case: notice a moved assistant character without generating a token.)*
- [x] Closes with a **bulleted contributions list**. — `\textbf{Generality}` *(four bullets, claim + pointer each.)*

## 6. Methods & results

- [x] **Background** defines all key terms. — `Two measurement positions` *(plaintext / open_user_turn defined from scratch; SFT/DPO/RLVR expanded at first use; logit lens defined with citation at point of use.)*
- [x] **Methods** explains each approach and why. — `factorises out two confounds` *(each design element is tied to the confound it kills.)*
- [x] **Results** tie each experiment to a specific claim. — `sec:scale`..`sec:lens` *(section titles are the claims.)*
- [x] Each evidence style has its own section linked to its claim. *(logprob sweep §3, stage trajectory §4, perturbation §5, per-layer §6; generation cross-check kept OUT of the coinflip sections by design.)*
- [x] Technical choices are **justified**, not just stated. — `would contaminate` *(token-variant filtering, marker stripping, and base-models-plaintext-only all carry their reasons.)*
- [x] No critical methodological choice buried in passing. — `user-close marker` *(the load-bearing choice — keeping the user turn open — is a named Method paragraph plus a verification script.)*
- [x] Appendix material **cross-referenced** from main text. *(all eight appendices A–H are `\ref`'d from body or Limitations; verified by grep — no orphan appendix.)*

## 7. Figures & tables

- [x] Each figure's takeaway is visually obvious. — `fig:psm_rep` *(headline figure is a two-panel before/after of the same network; scale figure separates families by marker and tiers by fill.)*
- [ ] Visual annotation directs attention to the key pattern. *(figures rely on marker/fill conventions and a zero reference line; no arrows/highlights — acceptable but not verified to Nanda's bar; candidate polish, not blocking.)*
- [x] Axis titles, legends, captions readable. *(verified on the compiled PDF at 100% zoom during the 2026-07-08 page-by-page review.)*
- [ ] No load-bearing red/green; colorblind-safe palettes. *(NOT yet verified against the actual palettes in `fig_em_flip.png` / `fig_logit_lens.png` — check before submission; captions do encode tier meanings in words, which mitigates.)*
- [x] **Captions self-contained.** — `fig:scale` *(marker/fill/error-bar semantics all in-caption; drift table caption states the ΔEu/ΔGu reference row explicitly after the 2026-07-08 correction.)*
- [x] Captions describe **what is shown**, not what the argument needs. *(the fig:psm_rep caption reports our means and the cited reference values as comparison, nothing interpretive; tab:harmcont caption states the split rather than the conclusion.)*
- [x] Figure 1 gives a high-effort visual overview of the headline result. — `fig:psm_rep` *(first figure IS the headline replication, placed at §3 top, reference values in caption.)*
- [x] At least one **explanatory diagram** where it would help. *(the verbatim prompt box in ¶2 of the intro plays this role — the probe is a one-prompt pipeline; a box diagram would add nothing.)*

## 8. Related work

- [x] Explains why this differs from / builds on the closest prior work. — `We replicate the effect on open-weight models` *(relationship to the source diagnostic is the paper's explicit premise; Discussion situates against calibration, persona-direction, and user-modeling threads.)*
- [x] Parallel work acknowledged. — `sits alongside other recent evidence` *(concurrent 2026 arXiv work on assistant persona structure cited in Discussion; the informal-explorations footnote was removed 2026-07-09 at co-author direction — do not re-add.)*
- [x] Criticism of prior work is professional. *(N/A in spirit — no prior work is criticized; the source result's limits are framed as scope, not flaws.)*
- [x] First-instance citations credited. — `nostalgebraist_logit_lens` *(logit lens → nostalgebraist 2020; simulators → janus 2022; EM → Betley et al.; PSM → Marks/Lindsey/Olah post, not the system-card summary.)*
- [x] Placement: motivating-related-work upfront, the rest penultimate. *(no standalone section: the motivating diagnostic is intro ¶2; the three literature threads live in the Discussion, penultimate position.)*
- [x] No performative citation padding. *(20 bib entries, all cited; `comm` cross-check of `\cite*{}` keys vs bib keys returns empty both directions.)*

## 9. Discussion, limitations, conclusion

- [x] Dedicated, honest **Limitations** section. — `shape comparison, not a matched replication` *(names six concrete limits including the ones a hostile reviewer would find — synthetic stimuli, lens drift, single judge, unexplained sign inversions, dirty rank ablation.)*
- [x] Each limitation says how strongly to update. — `indicative rather than exact` *(lens drift → per-layer magnitudes indicative; no-Sonnet-access → shape comparison only; anomalies → reproducible but unexplained.)*
- [x] No overclaiming. — `is evidence that` *(claims are worded as evidence-for, "naturally read as", "consistent with"; the one flat assertion — base cells near zero — is the most-verified fact in the paper.)*
- [ ] **Future work** identifies genuinely exciting directions. *(no explicit future-work sentences; the anomalies paragraph gestures at one open question — consider one closing sentence naming the disambiguation experiment, or accept the omission.)*
- [x] Conclusion earns its place. *(one paragraph; synthesizes the five findings into the single entanglement sentence rather than re-walking the intro.)*

## 10. Appendices

- [x] Appendices hold full hyperparameters, extended analyses, tacit knowledge. — `app:method`, `app:olmo_full`, `app:harmcont` *(token selection fiddly bits, full stage trajectories, judge-vs-rule agreement and verbatim examples.)*
- [x] Glossary if needed. *(N/A — all terms defined inline at first use; the paper introduces only two coined terms, both defined in Method.)*
- [x] Main text references each appendix. *(A–H all referenced; verified by grep.)*
- [x] Appendices accepted as "low stakes". *(appendix pages carry data completeness, not polish; the only appendix work this pass was fixing an overfull table box.)*

## 11. Prose & language

- [x] **Plain language preferred.** — `The next token is the next thing the human would type` *(the probe is explained without notation before the Method formalizes it; 2026-07-08 rewrite removed number-parenthetical clutter from all body sections.)*
- [x] Verbose sentences cut on an editing pass. *(the de-numbering pass doubled as a tightening pass; abstract rewritten from a stats-dense version to nine single-idea sentences.)*
- [x] **Illusion of transparency** countered. *(rewrite explicitly re-derives the layering question from scratch; SFT/DPO/RLVR expanded; the "position −1 = next user-typed character" point is made three separate ways.)*
- [x] Confidence language matches evidence strength everywhere. *(same verification as §9.3.)*
- [x] Inform, not persuade. *(contradicting cells are reported in the same paragraphs as the confirming ones — the open_user_turn non-amplification and pipeline non-replications are in the body, not buried.)*

## 12. Process & quality control

- [x] Started with a **compression pass** before prose. *(claims and experiment set fixed in CANONICAL_PLAN.md and the co-author's ordering doc before the ACL draft existed.)*
- [x] Bullet-point narrative reviewed by a second person. *(co-author specified the four-experiment body structure and ordering in the shared doc.)*
- [x] Introduction outline received outside feedback. *(2026-07-08: co-author reviewed the compiled PDF and mandated the phenomenon-first reframe this version implements.)*
- [ ] Full outline reviewed before figures were finalized. *(deviation: figures were generated from sweep analyzers before the outline existed; mitigated by the figures being data-shaped rather than narrative-shaped.)*
- [x] Time allocated comparably across abstract / intro / figures / rest. *(this revision cycle spent its entire first half on abstract + intro alone.)*
- [x] At least two iterative editing passes. *(ACL-port pass, then the feedback rewrite pass — abstract, intro, de-numbering, discussion expansion.)*
- [x] External reader feedback on the full draft at least once. *(co-author full-PDF review 2026-07-08; second review of this version pending.)*
- [ ] Final pass: every sentence earning its place. *(scheduled as the last action before submission, after co-author signs off on this version.)*

## 13. Project-specific cross-cutting checks

1. - [x] **Zero numbers, sample sizes, or figure refs in abstract + introduction** (co-author rule, 2026-07-08; overrides §4 "concrete metric" item). *(digit grep over abstract+intro returns only model/version names and the verbatim prompt box; no counts, biases, or percentages.)*
2. - [x] **No prose numbers duplicating figure content** — quantitative values live in captions or appendix tables. *(body-prose decimal grep §3–§6 returns only version strings and the Pearson r, which appears in no figure.)*
3. - [x] **Phenomenon-first framing** — the question (assistant preferences at user-turn positions) leads; the coin-flip diagnostic is introduced as the chosen instantiation. — `can we measure preferences that ought to belong to the assistant` *(intro ¶1 ends on the question before any prior work is named.)*
4. - [x] **Scholarly citation language** — no "The PSM post" / "the system card" as narrative subject; sources appear via `\citet`/`\citep`, artifact names only with citation attached. *(grep for "PSM" and "probe" over paper.tex both return zero hits — the acronym is out of the title/body entirely and "probe" was retired for its linear-probe collision; every "system card" mention carries `\citep{anthropic_sonnet45_card}`.)*
5. - [x] **Bibliography hygiene** — real author lists, no commentary in `note` fields, every entry cited, every key resolves. *(references.bib rebuilt 2026-07-08 from per-entry source verification; key⇄cite cross-check empty both ways; build log has 0 citation/reference warnings.)*
6. - [x] **Harmful-stimuli discipline** — tooling never prints dispreferred prompt bodies; Appendix A reproduces only already-published items (co-author-approved decision); the sealed P1 opener appears nowhere in the paper; harmful-continuation examples hold out CBRN content. *(verified against Appendix A contents and the Ethics statement.)*
7. - [x] **Anonymity** — `\usepackage[review]{acl}`; no author-identifying URLs; no collaborator or third-party share links remain. *(a private gdoc reference was dropped 2026-07-08; the third-party share-link footnote was removed 2026-07-09.)*
8. - [x] **Figure freshness + README truthfulness** — README describes real-data figures and correct provenance (make_figures.py for all except the two persona-drift PNGs, which come from `src/analyze_drift.py`). *(all figures regenerated 2026-07-09 after the cold-read audit; README provenance line fixed the same day.)*
