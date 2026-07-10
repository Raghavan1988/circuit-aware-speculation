# Experiment Contract

This document locks the first-pass protocol. Deviations require a record in `DECISIONS.md`. (Amendments to policies 8–9 in this revision are authorized by D009.)

## Models

| Role | Primary | Replication |
|---|---|---|
| Target | Qwen2.5-7B-Instruct | Tokenizer-compatible Llama 8B |
| Draft | Qwen2.5-0.5B-Instruct | Tokenizer-compatible Llama 1B |
| Fallback | — | Second Qwen target–draft size ratio |

Record exact provider, revision, tokenizer revision, dtype, quantization, attention backend, and generation configuration. Do not use floating model aliases without a saved revision.

## Decoding

- Primary protocol: greedy, exact-match speculative decoding.
- Candidate actions: `skip`, `1`, `2`, `3`, `4`, `6`, `8`.
- The target-only and speculative paths must produce token-identical outputs from identical prompts.
- KV caches must be rolled forward or truncated correctly after rejection.
- Generation cap: choose 128 or 256 new tokens before the first full sweep and record the decision.

## Data

Use public datasets representing:

- code: HumanEval or MBPP;
- math: GSM8K plus one harder reasoning subset if licensing and compute permit;
- chat: MT-Bench or a documented alternative;
- held-out domain: summarization or retrieval-grounded QA.

Target 150–200 test prompts per category, subject to dataset size. Split by prompt into development and test sets before trace-derived model fitting. Freeze split manifests and record dataset versions and licenses.

## Policies

1. target-only;
2. fixed lengths `1`, `2`, `3`, `4`, `6`, `8`;
3. best global fixed length selected on development data;
4. best per-domain fixed length selected on development data;
5. entropy threshold;
6. recent-acceptance heuristic;
7. BanditSpec-style policy;
8. learned acceptance predictor on draft representations (SpecDec++-style head; closest published baseline — reproduce early, issue I22);
9. circuit-aware compute-optimal controller (deployed signal must be cheap, e.g., pre-round prediction from cached verified-context representations; offline activation capture must never run inside deployed timing);
10. offline oracle using realized outcomes.

Online policies must report cold-start behavior separately. The oracle must never be included in deployable comparisons.

## Timing

- Record hardware model, count, power mode where available, driver, CUDA, framework, and library versions.
- Warm models and kernels before measured requests.
- Synchronize the device at timing boundaries.
- Measure prefill, drafting, verification, controller, tracing, and total latency separately.
- Compare policies on identical prompts and output caps.
- Run at least three timing repetitions or bootstrap paired request-level confidence intervals.
- Primary concurrency/batch setting is one. Batch eight is optional validation, not a prerequisite.

## Metrics

### Primary systems metrics

- End-to-end wall-clock speedup over target-only decoding.
- TPOT and output-token throughput.
- Gain over best global and per-domain fixed policies.
- Rejected draft tokens per accepted token.
- Controller and tracing overhead.
- Peak device memory.
- Cumulative latency regret during traffic shift.

### Mechanistic metrics

- AUROC and area under the precision–recall curve.
- Brier score and expected calibration error.
- Incremental performance beyond entropy/margin/history.
- Cross-domain and cross-model degradation.
- Intervention effect size with uncertainty.
- Acceptance and waste by token category and generation phase.

## Analysis rules

- Use paired comparisons on the same prompts.
- Include uncertainty for primary metrics.
- Fit thresholds, probes, calibration, and action mappings on development data only.
- Never choose the best policy using test latency.
- Preserve excluded and failed requests with machine-readable reasons.
- Report both aggregate and per-domain outcomes.
- Treat multiple adjacent tokens from one prompt as dependent observations.

## Required artifacts

Each run must produce:

- immutable configuration and environment metadata;
- split manifest;
- request-level summary;
- round/token trace conforming to `TRACE_SCHEMA.md`;
- failure log;
- checksums or stable artifact identifiers;
- the command and code revision used.
