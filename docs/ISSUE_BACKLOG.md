# Issue Backlog

## Workflow

Statuses are `OPEN`, `IN_PROGRESS`, `BLOCKED`, and `DONE`. Before starting, add an owner and change status. “Parallel” lists issues that may safely proceed concurrently once dependencies are satisfied.

| ID | Status | Owner | Issue | Depends on | Parallel | Compute |
|---|---|---|---|---|---|---|
| I01 | OPEN | — | Provision and pin RunPod/Modal environment | — | I05 | GPU setup |
| I02 | OPEN | — | Implement exact primary Qwen target–draft decoding | I01 | — | A100/H100 |
| I03 | OPEN | — | Add equivalence, rejection, and KV-cache tests | I02 | I04,I05 | Small GPU/CPU |
| I04 | OPEN | — | Implement synchronized latency instrumentation | I02 | I03,I05 | GPU |
| I05 | OPEN | — | Build dataset ingestion and prompt-grouped splits | — | I01,I03,I04 | CPU |
| I06 | OPEN | — | Implement and validate the trace writer | I02,I05 | I03,I04 | GPU/CPU |
| I07 | OPEN | — | Run target-only, skip, and fixed-length sweep | I03,I04,I06 | — | A100/H100 |
| I08 | OPEN | — | Implement entropy and recent-acceptance policies | I03,I06 | I09 | Small GPU |
| I09 | OPEN | — | Reproduce a BanditSpec-style baseline | I03,I06 | I08 | GPU |
| I10 | OPEN | — | Add selected-layer activation capture | I03,I06 | I11 | A100/H100 |
| I11 | OPEN | — | Build and validate token-category annotation | I05,I06 | I10 | CPU |
| I12 | OPEN | — | Train leakage-safe layerwise acceptance probes | I10,I11 | — | GPU/CPU |
| I13 | OPEN | — | Evaluate calibration and incremental information | I07,I08,I09,I12 | — | CPU |
| I14 | OPEN | — | Implement compute-optimal selective speculation | I07,I13 | — | GPU |
| I15 | OPEN | — | Run rejection-direction interventions and controls | I12 | I14 | A100/H100 |
| I16 | OPEN | — | Run domain- and traffic-shift experiments | I14 | I15,I17 | A100/H100 |
| I17 | OPEN | — | Add and validate the replication model pair | I03,I04,I06 | I15,I16 | A100/H100 |
| I18 | OPEN | — | Generate acceptance atlas and primary figures | I11,I13,I14,I15,I16,I17 | — | CPU |
| I19 | OPEN | — | Assemble anonymous artifact-driven TMLR manuscript | I18 | — | CPU |
| I20 | OPEN | — | Run clean reproduction and evidence audit | I19 | — | GPU/CPU |

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

### I19 — Manuscript

- Use the unmodified official TMLR template and maintain anonymity.
- Link every empirical claim to the claims ledger.
- Include limitations, reproducibility, and appropriate broader-impact discussion.

### I20 — Audit

- Reproduce the primary table from a clean environment and recorded commands.
- Verify claims, citations, anonymity, licensing, checksums, and absence of fabricated/illustrative results.
- Produce a written pass/fail recommendation against the submission gate.
