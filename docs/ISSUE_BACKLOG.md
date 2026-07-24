# Issue Backlog

## Workflow

Statuses are `OPEN`, `IN_PROGRESS`, `BLOCKED`, and `DONE`. Before starting, add an owner and change status. “Parallel” lists issues that may safely proceed concurrently once dependencies are satisfied.

| ID | Status | Owner | Issue | Depends on | Parallel | Compute |
|---|---|---|---|---|---|---|
| I01 | IN_PROGRESS | Claude | Provision and pin Modal environment | — | I05 | GPU setup |
| I02 | IN_PROGRESS | Claude | Implement exact primary Qwen target–draft decoding | I01 | — | A100/H100 |
| I03 | IN_PROGRESS | Claude | Add equivalence, rejection, and KV-cache tests | I02 | I04,I05 | Small GPU/CPU |
| I04 | IN_PROGRESS | Claude | Implement synchronized latency instrumentation | I02 | I03,I05 | GPU |
| I05 | IN_PROGRESS | Claude | Build dataset ingestion and prompt-grouped splits | — | I01,I03,I04 | CPU |
| I06 | DONE | Claude | Implement and validate the trace writer | I02,I05 | I03,I04 | GPU/CPU |
| I07 | IN_PROGRESS | Claude | Run target-only, skip, and fixed-length sweep | I03,I04,I06 | — | A100/H100 |
| I08 | DONE | Codex | Implement entropy and recent-acceptance policies | I03,I06 | I09 | Small GPU |
| I09 | DONE | Codex | Reproduce a BanditSpec-style baseline | I03,I06 | I08 | GPU |
| I10 | DONE | Claude | Add selected-layer activation capture | I03,I06 | I11 | A100/H100 |
| I11 | DONE | Grok | Build and validate token-category annotation | I05,I06 | I10 | CPU |
| I12 | DONE | Claude | Train leakage-safe layerwise acceptance probes | I10,I11 | — | GPU/CPU |
| I13 | IN_PROGRESS | Claude | Evaluate calibration and incremental information | I07,I08,I09,I12 | I23 | CPU |
| I14 | OPEN | — | Implement compute-optimal selective speculation | I07,I13 | — | GPU |
| I15 | IN_PROGRESS | Claude | Run rejection-direction interventions and controls | I12 | I14 | A100/H100 |
| I16 | OPEN | — | Run domain- and traffic-shift experiments | I14 | I15,I17 | A100/H100 |
| I17 | OPEN | — | Add and validate the replication model pair | I03,I04,I06 | I15,I16 | A100/H100 |
| I18 | IN_PROGRESS | Claude | Generate acceptance atlas and primary figures | I11,I13,I14,I15,I16,I17 | — | CPU |
| I19 | IN_PROGRESS | Claude | Assemble anonymous artifact-driven journal manuscript | I18 | — | CPU |
| I20 | OPEN | — | Run clean reproduction and evidence audit | I19 | — | GPU/CPU |
| I21 | DONE | Grok | Verify landscape additions; maintain living comparison table | — | I01,I05 | CPU |
| I22 | OPEN | — | Reproduce SpecDec++-style learned acceptance-head baseline | I03,I06,I10 | I08,I09 | GPU |
| I23 | DONE | Claude | Pre-round acceptance prediction from cached representations | I10,I12 | I13,I14 | GPU |
| I24 | OPEN | — | Staged release package (benchmark, recipes, integration adapter) | I18,I20 | — | GPU/CPU |

## Build status (2026-07-10, Claude)

Phase-1 Steps 1–2 implemented. **Verified locally:** pure accept/commit/rollback
logic (`cas.commit`, 9 tests) and prompt-grouped splitting (`cas.data.splits`,
8 tests) — 17 passing, stdlib only. **Pending on Modal/H100** (local torch is
CPU-only and too old): the model-level bit-identity gate (`test_equivalence_gpu`),
`smoke_decode`, dataset ingestion, and revision pinning. Issues stay IN_PROGRESS
until the GPU gate passes; no results exist yet. Run order on Modal:
`verify_env` → paste SHAs into `cas.config` → `ingest_data` → `run_tests` →
`smoke_decode`.

## Build status (2026-07-11, Claude — I06 storage layer)

- I06 writer implemented and locally verified: `cas.trace` now has extended
  records (D018 fields: `target_argmax_ids`, frontier entropy/margin,
  activation-artifact slot, per-position match), `validate.py` (TRACE_SCHEMA
  invariants, pure stdlib), `writer.py` (Parquet, write-once + MANIFEST
  checksums). Repro: `PYTHONPATH=src python -m pytest tests/test_trace_writer.py -q`
  (14 passed; full suite 59 passed). I06 stays IN_PROGRESS until the engine
  emits the new fields (engine wiring + D017 stop-rule seam) and one GPU run
  round-trips through the writer on Modal.

## Build status (2026-07-10, Grok — I21 + I11)

- **I21 DONE:** Four planning-pass arXiv IDs verified on primary archive
  (2603.01639, 2605.02888, 2604.14682, 2606.30265). Mirror-found item was
  **2606.30265** — primary URL https://arxiv.org/abs/2606.30265. Living table
  `docs/landscape.md` updated; C04/C10 impacts recorded in
  `docs/CLAIMS_LEDGER.md`. C04 not pre-empted (domain grain only). PLAN.md §3
  edit proposed as a note in landscape.md (not applied; Claude owns PLAN.md).
- **I11 DONE:** `cas.annotate` (categories + phases, versions v1.0.0); interface
  recorded as D016. Seam: pure `annotate_token(...)` for I06 writer.
  Repro: `PYTHONPATH=src python -m pytest tests/test_annotate.py -q`
  (21 passed; stratified agreement script-printed with `-s`).

## Build status (2026-07-10, Codex — I08 + I09)

- **I08 DONE:** resettable rolling-acceptance action policy and entropy stop rule;
  frozen constructor hyperparameters, threshold boundaries, skip handling, and
  request resets are unit-tested. The engine consult point is recorded in D017
  for Claude to wire after I06.
- **I09 DONE (round 1; superseded by D019):** UCBSpec-style length-arm policy with round-robin cold start,
  accepted-plus-bonus reward, and the published confidence radius. Deviations
  are disclosed in the module and claims ledger. Repro:
  `PYTHONPATH=src python -m pytest tests/test_policies.py -q` (7 passed).

## Build status (2026-07-11, Codex — I09 round-2 repair)

- **I09 DONE:** D019 retains the original accepted-plus-bonus selector as
  `UCBSpecNaive` and repairs the primary selector with emitted tokens per frozen,
  development-measured action cost. The reward and confidence radius share
  throughput units. Deterministic tests cover poor-acceptance convergence away
  from an expensive `L=8`, cost-unit invariance, early draft stopping, invalid
  costs, frozen inputs, and request reset.
- **D018 policy scaffolding complete:** conditional-continuation survival,
  expected-yield/cost action selection, same-round counterfactual labels,
  prompt-grouped label-aware Platt calibration, and the optional
  confidence-abstaining probe-prior UCB interface are pure stdlib. Terminal rows
  are explicitly rejected by the nominal-yield label adapter. This is tooling
  only: I14 remains OPEN and no test traces were consumed.
- Repro: `PYTHONPATH=src python -m pytest tests/test_policies.py
  tests/test_survival.py -q` (32 passed).

## Build status (2026-07-11, Grok — I21 round 2)

- **I21 DONE (round-2 maintenance):** Corrections + sweep additions in
  `docs/landscape.md` and `docs/CLAIMS_LEDGER.md`. Primary-verified:
  Not-a-Bandit ID → **2510.20064** (2506.00285 is unrelated POMDP paper);
  DSpark **not** an SGLang ship (authors' production stack / live traffic);
  AdaEAGLE, Judge Decoding, WhiFlash, C2T, Sequoia, DISCO, AdaEDL, DSDE,
  TurboSpec, CaDDTree, SemanticSpec (full text), and methodology trio
  2509.10625 / 2606.14530 / 2506.08572. G2: SemanticSpec does not scoop
  C01 controlled study or C10 pre-round cell. G3: deployed-practice section
  (vLLM Dynamic SD, SGLang adaptive EMA, TensorRT-LLM static max_draft_len).
  G4: C04 domain-control positioning for the atlas. C10 claims-table cell
  updated with the three narrowing papers. PLAN.md edits proposed as notes
  only (Claude owns PLAN.md). Did not touch Claude engine/trace/analysis
  code or Codex policy modules.

## Build status (2026-07-19, Claude — I13/I23 generator-critic substrate)

- **I13 + I23 claimed IN_PROGRESS.** Execution method: the generator-critic
  autoresearch loop (`docs/generator_critic.md`, ratified as D023). Load-bearing
  finding: I10's `capture_activations` records only the DRAFT residual stream at
  proposal-generating positions (the within-round signal); the pre-round headline
  (C10/I23) needs the TARGET verified-context FRONTIER representation, which was
  not captured. Substrate under construction:
  - `src/cas/autoresearch/types.py` — shared contract (`FeatureSpec`, seed
    families, frontier artifact layout, frozen `preround_hardened` bar). DONE.
  - Step 1 `capture_frontier_activations` (Modal, target frontier capture);
    Step 2 `features.py` (seed-library transforms); Step 3 `eval.py` (incremental
    lift + equal-capacity controls + regret); Step 4 `cost.py` (deployed-cost
    probe); Step 5 `src/cas/autoresearch/generator_critic.js` (orchestration).
  - Frozen bar = `preround_hardened` (~0.73 AUROC); every candidate must beat it
    AND norm-matched + random controls under prompt-grouped GroupKFold OOF,
    dev-only. "Circuit"/"mechanism" language stays G2-gated (D020).

## Build status (2026-07-22, Claude — I13/I23/I15 autoresearch + causal)

Generator-critic loop delivered the pre-round signal + causal validation (D023,
D025; `docs/autoresearch_outcomes.md`, `docs/causal_intervention_report.md`).

- **I13/I23 (dev done; frozen test pending):** domain-controlled first-token
  acceptance lift beyond entropy+domain, +0.072/+0.112/+0.069 (Qwen-v1/v2/Llama),
  CI-clean, replicated cross-model + cross-corpus, near-zero cost (G3 microbench).
  First-token only (run-length null-to-harmful; length controller does not benefit).
  C10 stays `UNTESTED` until the frozen predictive test pass (running).
- **I15 (causal, replicated):** forward-hook steering of the first-token direction
  disrupts acceptance ~2–10× norm-matched controls, dose-dependent, beyond entropy,
  all 4 layers, on Qwen-v1 (`sealed_fidelity` 0.95) + Llama (1.00). Empirical G2 met;
  G2 language is a human gate (D020). Pure logic + tests in
  `src/cas/autoresearch/interventions.py`; runner `modal_app.py::intervene`.
- Capture-sampling bug (2-domain undersampling) found + fixed (stratified, D025);
  re-captured 4/7/4 domains; finding survived domain control (not domain identity).
- **Frozen test pass DONE (later 2026-07-22): 3/3 PASS.** Single-spec
  `raw_frontier`, domain-controlled, untouched test split: Δauroc +0.0755
  (Qwen-v1) / +0.0918 (Qwen-v2) / +0.0542 (Llama), all p(Δ≤0)=0, controls-clean.
  **C10 → `SUPPORTED` (first-token scope)**; I23 → DONE. Integrity note: a
  pre-existing v2 test artifact (earlier unrecorded unblinding) is disclosed in
  the ledger; conclusion unchanged. I13 stays IN_PROGRESS (calibration reporting
  + manuscript-facing figures remain).
- **I19 manuscript draft started (2026-07-22, Claude):** `paper/main.tex`
  (anonymous, compiles standalone; arXiv-ID-only citations pending full bib).
  Every number transcribed from the ledger's script-generated artifacts;
  G2-gated language (representation/diagnostic signal) per D020. Remaining for
  I19: figures (I18), full bibliography, citing-sweep refresh at freeze, owner
  G2 language decision, then G5 audit (I20).

## Acceptance criteria and artifacts

### I01 — Environment

- Produce a pinned, non-secret environment specification and a command that verifies GPU/framework compatibility.
- Record exact driver, CUDA, framework, Transformers, and attention-backend versions.
- Do not embed provider credentials or machine-specific tokens.

### I02 — Exact decoding

- Expose runtime actions `skip,1,2,3,4,6,8`.
- Correctly accept the longest matching prefix and repair cache state after rejection.
- Provide a minimal reproducible command using the pinned primary revisions.

### I03 — Correctness tests

- Cover all-accepted, first-token rejection, middle rejection, zero drafting, maximum drafting, termination, and cache rollback.
- Verify token-identical outputs against target-only greedy decoding across a fixed prompt suite.

### I04 — Timing

- Synchronize device timing boundaries.
- Record prefill, drafting, verification, controller, tracing, and total latency.
- Demonstrate stable repeated measurements and document timer overhead.

### I05 — Data

- Produce versioned dataset manifests and immutable development/test splits grouped by prompt.
- Record licenses, source revisions, exclusions, and prompt hashes.

### I06 — Trace writer

- Validate all `TRACE_SCHEMA.md` invariants automatically.
- Keep large activations separate and resolvable by artifact identifier.
- Demonstrate round/request aggregate consistency.

### I07 — Fixed baselines

- Run identical prompts for target-only and every fixed action.
- Select best global and per-domain fixed actions using development data only.
- Produce immutable raw traces and paired uncertainty estimates.

### I08 — Simple adaptive baselines

- Implement entropy and recent-acceptance policies with frozen test-time hyperparameters.
- Unit-test threshold boundaries and state reset between requests.

### I09 — Bandit baseline

- Match a documented BanditSpec-style formulation and record all deviations.
- Report cold-start and steady-state behavior separately.

### I10 — Activation capture

- Capture declared early, middle, and late hook points without changing output tokens.
- Measure capture overhead and validate token-to-activation alignment.

### I11 — Token annotation

- Implement versioned overlapping categories and validate a stratified manual sample.
- Preserve ambiguity rather than forcing mutually exclusive labels.

### I12 — Probes

- Use prompt-grouped splits and regularized models.
- Save model, feature, split, and calibration metadata.
- Compare early, middle, late, and final layers.

### I13 — Incremental information

- Compare metadata, entropy, margin, history, hidden features, and combined features.
- Report AUROC, AUPRC, Brier score, calibration error, and transfer degradation with uncertainty.

### I14 — Selective controller

- Estimate payoff for all actions using measured costs.
- Include controller overhead and a `skip` action.
- Freeze the main mapping before test evaluation.

### I15 — Interventions

- Use development-derived directions, multiple strengths, random controls, and norm-matched controls.
- Report proposed-token, entropy, divergence, and acceptance effects.
- Do not label results causal unless all criteria in `RESEARCH_SPEC.md` pass.

### I16 — Shift experiments

- Evaluate at least three ordered traffic changes.
- Report pre-shift performance, recovery requests, calibration drift, skip frequency, and cumulative latency regret.

### I17 — Replication

- Verify tokenizer compatibility and exact decoding independently.
- If the Llama pair fails feasibility, record the decision before using the Qwen fallback.
- Repeat central fixed, adaptive, and mechanistic comparisons.

### I18 — Analysis artifacts

- Generate the acceptance atlas and all figures from immutable traces via scripts.
- Link every plotted value to run identifiers.
- Include negative and contradictory results.
- Produce the two headline figures: (a) layerwise emergence of acceptance information with intervention effect sizes overlaid; (b) end-to-end latency per policy with the overhead decomposition visible (draft / verify / controller / capture). If the central result cannot be told in these two figures, it is not yet crisp enough.

### I19 — Manuscript

- Use the target journal's unmodified official template (venue named only in the owner's private channel, per D008) and maintain anonymity.
- Link every empirical claim to the claims ledger.
- Include limitations, reproducibility, and appropriate broader-impact discussion.

### I20 — Audit

- Reproduce the primary table from a clean environment and recorded commands.
- Verify claims, citations, anonymity, licensing, checksums, and absence of fabricated/illustrative results.
- Produce a written pass/fail recommendation against the submission gate.

### I21 — Landscape verification

- Verify the four planning-pass reference additions against primary sources: arXiv:2603.01639 (adaptive drafting via RL), arXiv:2605.02888 (draft confidence under KV compression), arXiv:2604.14682 (task-conditioned acceptance dynamics), arXiv:2606.30265 (theoretical treatment of acceptance); one was found via a mirror site and needs primary-archive confirmation.
- Re-scan for newer adjacent work; maintain a living comparison table in the repository.
- Record impact on claims (notably C04) in the ledger before novelty claims are frozen.

### I22 — Learned-head baseline

- Reproduce a SpecDec++-style acceptance-prediction head on draft representations; document all deviations from the published formulation.
- Report prediction quality and end-to-end latency under the same timing rules as every other policy; this is the closest published baseline for C01 and contract policy 8.

### I23 — Pre-round prediction (headline candidate, per D009)

- Predict next-round acceptance and accepted length from already-cached verified-context representations at selected layers, before any draft compute is spent.
- Compare marginal deployed cost and prediction quality against post-draft signals (entropy, margin, the I22 head); report offline value and deployed-path cost separately.
- Freeze feature and layer choice on development data; outcomes update C10.

### I24 — Release package (staged per D010)

- Package the versioned trace corpus/benchmark with documentation and licenses, reproducible cloud recipes with cost accounting, and (optional, G4) a serving-engine integration adapter or profiler.
- Document the corpus's secondary framing (naturally-labeled small/large-model disagreement dataset; see `RESEARCH_SPEC.md`) and include the governance documents (claims ledger, decision log) in the public release.
- Include a concise technical article built on the two I18 headline figures; open an upstream issue or pull request where the artifact solves a concrete serving problem.
- Each release stage requires the corresponding gate in `RESEARCH_SPEC.md`; no component may present claims beyond the ledger.

## Build status (2026-07-23, Claude — I19 C04 integration)

- **I19:** C04 folded into `paper/main.tex` as §6 (`sec:atlas`) per D026 —
  atlas framing + domain control, the pre-registered T1–T4 frozen-test table
  (`tab:atlas`), the two disclosures (expected-null reversal on Llama-code;
  weakest cell Qwen-v1 summ p=0.027), and the phase-axis narrowing. Also added:
  contribution bullet 4, an abstract clause, a Limitations scope paragraph, and
  a Conclusion sentence. All numbers copied from the script-generated ledger
  entry of commit b8eb122; none hand-derived. Compiles clean: 11 pages
  (was 9), 0 LaTeX errors, 0 undefined references.
- **Still open on I19:** figures (none exist yet — the atlas heatmap, frozen-test
  forest plot, dose–response, length-decay and reliability plots are the priority
  set), full bibliographic entries, I21 citing-sweep refresh at freeze, and the
  G2 language decision (D020, human gate). I19 stays IN_PROGRESS.
- **I18:** atlas-evidence half remains done; the figure half is unchanged and is
  now the binding constraint on I19.

## Build status (2026-07-23, Claude — I18 primary figures, first four)

- **I18:** `scripts/make_figures.py` added — regenerates all figures from sealed
  analysis JSONs pulled from the `cas-artifacts` Modal volume into a gitignored
  `artifacts/analysis/` mirror. No statistic is computed in the script; it only
  reshapes recorded numbers into geometry. Four figures produced and wired into
  `paper/main.tex`:
  - `fig_forest.pdf` (§5.1) — frozen-test lift, both protocols, with the
    equal-capacity control's own lift shown alongside.
  - `fig_dose.pdf` (§5.4) — dose-response steering, 4 layers × 2 families.
    Qwen-v2 has no panel (never intervened on); stated in the caption.
  - `fig_length.pdf` (§5.3) — per-length survival lift with CI bands.
  - `fig_atlas.pdf` (§6) — C04 atlas heatmap, category × domain, all 3 settings.
- Cross-checked: every plotted value matches the corresponding manuscript table
  (`tab:frozen`, `tab:length`, `tab:atlas`) and the ledger entries they came
  from. Compiles clean: 13 pages, 0 errors, 0 undefined references.
- `paper/.gitignore` gained `!figures/*.pdf` — the blanket `*.pdf` (for
  `main.pdf`) was excluding the figure PDFs, which must be tracked for the paper
  to build without the artifact volume.
- **Remaining on I18:** reliability/calibration diagram, regret-vs-draft-cost
  curve, and the intro timing schematic (TikZ, no artifacts needed). I18 and I19
  both stay IN_PROGRESS.

## Build status (2026-07-23, Claude — I18 remaining three figures)

- **I18:** three further figures added to `scripts/make_figures.py` and wired
  into `paper/main.tex`:
  - `fig_schematic.pdf` (§1) — round timeline showing that phi(t) is readable
    before the drafter runs while every literature signal needs post-draft
    state. No artifacts (a diagram, not a plot).
  - `fig_calibration.pdf` (§5.2) — ECE before/after the global Platt map, all
    three settings, frozen test under P2.
  - `fig_regret.pdf` (§5.5) — regret reduction vs draft cost, filled markers
    where the recorded CI clears zero.
- Compiles clean: 14 pages, 0 errors, 0 undefined references. Figure values
  re-checked against §5.2 and §5.5 prose (P2 ECE 0.037/0.024/0.021; Qwen-v1
  CI-robust at exactly one cost, Qwen-v2 at four, Llama at two).
- **DEVIATION — reliability diagram not produced.** A true reliability diagram
  (binned observed-vs-predicted) cannot be built from the sealed artifacts:
  `autoresearch_*_domctl.json` records scalar ECE/Brier only, and no per-example
  calibrated predictions are stored (`probes/<run>/` holds raw activations and
  metadata, not scored predictions). Producing one would require re-fitting the
  probe and re-scoring the frozen test split — which the protocol allows exactly
  once and which has been spent. `fig_calibration.pdf` is a scalar summary
  instead, and both the caption and the docstring say so.
  **Decision needed:** either (a) accept the scalar summary, or (b) add a
  bin-export to the scorer and re-run scoring for calibration bins only, with a
  dated DECISIONS entry recording that the re-score is calibration-reporting
  and does not revisit any test verdict. Not taken unilaterally — it touches the
  frozen-test protocol.
- **DEVIATION — schematic is matplotlib, not TikZ.** TikZ is not installed in
  the local TeX environment, so a TikZ figure could not be compiled or visually
  verified here. Drawing it with matplotlib keeps the whole figure set on one
  toolchain and adds no LaTeX dependency; the source is `fig_schematic()`.
- I18 figure work is complete for the current evidence set. I18 and I19 stay
  IN_PROGRESS pending the calibration decision above, full bibliographic
  entries, the I21 citing-sweep refresh at freeze, and the G2 gate (D020).

## Build status (2026-07-23, Claude — I19 manuscript build entry point)

- **Gap closed:** there was no committed way to build the paper. No Makefile, no
  build script, and `pdflatex` appeared in zero tracked files — the command
  existed only in session transcripts. Added `paper/Makefile` with four targets
  (`all`, `figures`, `artifacts`, `clean`) and a build section in `README.md`.
- `make -C paper` works on a fresh clone with no Modal access, because the
  figure PDFs are tracked; `make -C paper artifacts && make -C paper figures`
  is only needed when the sealed analysis JSONs change.
- The build gates on unresolved references: `pdflatex` exits 0 on undefined
  `\ref`, so an unchecked build silently ships `??` into the PDF. Verified the
  gate fires by injecting a bad `\ref` (build exits non-zero) and that a clean
  build still passes. `.DELETE_ON_ERROR` is set — without it a failed build left
  a stale `main.pdf` newer than `main.tex`, so the next `make` reported
  "nothing to be done" and the failure looked repaired. Both behaviours tested.
- `paper/.gitignore` fix (`!figures/*.pdf`) committed alongside: `make figures`
  writes into `figures/`, and without the negation any new or renamed figure PDF
  is silently ignored on a fresh clone.
- **Addendum (same day):** `make figures` was not reproducible — matplotlib
  stamps a `/CreationDate` into every PDF, so each run rewrote all seven figures
  with byte-different, content-identical files. That made `git status` unable to
  answer "did the figures actually change?". `scripts/make_figures.py` now saves
  with `metadata={"CreationDate": None}`; verified byte-identical across two
  consecutive runs. The one-time rewrite of the seven tracked PDFs is the
  timestamp removal only (confirmed: identical after stripping `/CreationDate`).

## Build status (2026-07-23, Claude — I19 plain-English rewrite)

- **I19:** full prose rewrite of `paper/main.tex` for readability (D027). Flesch
  Reading Ease 27.9 -> 73.5 whole-document, every section >= 70. New glossary
  section (`sec:terms`) defines all terms upfront; "frontier representation"
  renamed "frontier state" throughout. 14 -> 17 pages. Builds clean (0 errors,
  0 undefined refs, 7 figures, 1 bibliography).
- **Gates (durable):** `scripts/readability.py` (Flesch, per-section) and
  `scripts/check_invariants.py` (numbers/hedges/structure/vocabulary), wired as
  `make -C paper check`. Verified: 0 numbers dropped from any scientific section;
  17 whole-doc additions all glossary definitions; scope vocabulary 119 -> 160;
  labels/refs/figures/bibitems intact.
- **A real defect the gate caught:** the section-split for rewriting made the
  reproducibility section swallow the bibliography, and the assembler re-appended
  it -> two bibliographies (54 bibitems). Fixed; now 1 bibliography, 27 bibitems.
- **Still open on I19:** the abstract grew to ~500 words (was 280) and could be
  tightened; sentence length (~11 words) reads clipped for a journal; the G2
  language gate (D020) and the D010 anonymity decision remain human gates.
