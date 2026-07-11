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
| I10 | OPEN | — | Add selected-layer activation capture | I03,I06 | I11 | A100/H100 |
| I11 | DONE | Grok | Build and validate token-category annotation | I05,I06 | I10 | CPU |
| I12 | OPEN | — | Train leakage-safe layerwise acceptance probes | I10,I11 | — | GPU/CPU |
| I13 | OPEN | — | Evaluate calibration and incremental information | I07,I08,I09,I12 | — | CPU |
| I14 | OPEN | — | Implement compute-optimal selective speculation | I07,I13 | — | GPU |
| I15 | OPEN | — | Run rejection-direction interventions and controls | I12 | I14 | A100/H100 |
| I16 | OPEN | — | Run domain- and traffic-shift experiments | I14 | I15,I17 | A100/H100 |
| I17 | OPEN | — | Add and validate the replication model pair | I03,I04,I06 | I15,I16 | A100/H100 |
| I18 | OPEN | — | Generate acceptance atlas and primary figures | I11,I13,I14,I15,I16,I17 | — | CPU |
| I19 | OPEN | — | Assemble anonymous artifact-driven journal manuscript | I18 | — | CPU |
| I20 | OPEN | — | Run clean reproduction and evidence audit | I19 | — | GPU/CPU |
| I21 | DONE | Grok | Verify landscape additions; maintain living comparison table | — | I01,I05 | CPU |
| I22 | OPEN | — | Reproduce SpecDec++-style learned acceptance-head baseline | I03,I06,I10 | I08,I09 | GPU |
| I23 | OPEN | — | Pre-round acceptance prediction from cached representations | I10,I12 | I13,I14 | GPU |
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
