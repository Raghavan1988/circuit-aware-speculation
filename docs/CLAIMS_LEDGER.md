# Claims Ledger

Allowed statuses: `UNTESTED`, `SUPPORTED`, `PARTIAL`, `REFUTED`, and `RETIRED`.

No claim may move to `SUPPORTED` without experiment identifiers, applicable settings, uncertainty, and known counterexamples. A manuscript claim must be no broader than this record.

| ID | Proposed claim | Status | Required evidence | Experiment IDs | Counterexamples / limits |
|---|---|---|---|---|---|
| C01 | Draft hidden states contain acceptance information beyond entropy, margin, history, and domain. | UNTESTED | Prompt-grouped incremental comparison on held-out prompts | — | — |
| C02 | Acceptance information becomes accessible at identifiable draft-model layers. | UNTESTED | Layerwise probes replicated across domains and a second model setting | — | — |
| C03 | Rejection-associated directions have a controlled effect on draft–target divergence or acceptance. | UNTESTED | Dose-response intervention with random and norm-matched controls | — | — |
| C04 | Acceptance behavior differs systematically across token categories and generation phases. | UNTESTED | Acceptance atlas with paired uncertainty and annotation validation | — | Domain-level acceptance differences are prior (arXiv:2604.14682, I21 verified 2026-07-10); that work does **not** cover overlapping token-category labels or a fine-grained phase atlas — position as domain control/context. Annotation tooling landed I11 (`cas.annotate` v1.0.0); atlas evidence still pending I07+I18 |
| C05 | Selective speculation with a skip action reduces wasted compute relative to adaptive-length baselines. | UNTESTED | Held-out comparison including all overhead | — | — |
| C06 | The circuit-aware controller improves net latency over the best global fixed policy. | UNTESTED | Paired held-out wall-clock study with uncertainty | — | — |
| C07 | Any controller advantage persists against the best per-domain fixed policy. | UNTESTED | Per-domain held-out comparison | — | — |
| C08 | The signal or controller transfers under domain and traffic shift without full retuning. | UNTESTED | Shift study with calibration drift and latency regret | — | — |
| C09 | The principal finding replicates outside the primary Qwen pair. | UNTESTED | Compatible Llama pair or approved Qwen-ratio fallback | — | — |
| C10 | Next-round acceptance is predictable from cached verified-context representations before any draft compute, at deployable marginal cost. | UNTESTED | Prompt-grouped held-out comparison against post-draft signals (entropy, margin, learned head), with deployed-path overhead measured (I23) | — | I21 re-scan (2026-07-10): no primary-verified work predicts next-round acceptance from cached verified-context reps *before draft compute*. Closest are post-draft heads (SpecDec++, DSpark) and draft entropy/confidence (SVIP, SpecKV). Novelty threat low; re-check before freeze |
| C11 | The identified acceptance representation transfers beyond independent drafts to a modern speculator family. | UNTESTED | Cross-speculator evaluation; extension work, only after the core evidence gate (D009) | — | — |

## Landscape verification notes (I21)

### 2026-07-10 — Planning-pass four-paper verification (Grok)

- **Primary confirmed:** arXiv:2603.01639 (Learning to Draft / LTD — RL draft+verify throughput), arXiv:2605.02888 (SpecKV — draft confidence under compression), arXiv:2604.14682 (acceptance dynamics across cognitive domains), arXiv:2606.30265 (theory of acceptance certificates).
- **Mirror-found item:** arXiv:2606.30265 — **primary archive confirmed** at https://arxiv.org/abs/2606.30265 (v1, 29 Jun 2026). Not a phantom.
- **C04 verdict:** 2604.14682 is domain-grain (+ coarse early/late position bins, weak entropy–α). Does **not** pre-empt token-category/phase atlas. Status stays UNTESTED; prior work cited as control.
- **C10 / theory:** 2606.30265 constrains a pure-theory Track B option (certificates for local acceptance events) but is not an internal localization result and does not implement pre-round prediction.
- **Living table:** `docs/landscape.md` (authoritative). PLAN.md §3 edit proposed there as a note only.
- Logged by Grok, 2026-07-10.

### 2026-07-10 — I11 annotation tooling (not claim evidence)

- Implemented `cas.annotate` (CATEGORY_SET_VERSION=v1.0.0, PHASE_SET_VERSION=v1.0.0): overlapping categories + absolute generation-phase bins; pure stream function for the I06 seam.
- Stratified golden-sample agreement: run `PYTHONPATH=src python -m pytest tests/test_annotate.py -q -s` — agreement is script-printed, not hand-typed into this ledger as a scientific result.
- This does **not** move C04; acceptance-atlas numbers require real decode traces (I07+).
- Logged by Grok, 2026-07-10.

### 2026-07-11 — Literature-sweep impacts on C01/C10 (Claude; 25-agent verified sweep, details in CLAUDE_IDEAS.md)

- **C01 instrument scooped, science claim open:** DSpark (arXiv:2607.05147,
  2026-07-06, full text verified) publishes a calibrated **linear** head on
  draft hidden states for per-token acceptance (AUROC 0.81–0.90, ECE ≈1% after
  per-position temperature scaling). C01 must be framed as the controlled
  incremental-information study (vs combined entropy+margin+history+domain
  baseline, prompt-grouped splits, layer localization) — no published work has
  run that comparison; that negative finding survived checks across ~70 papers.
- **C10 narrowed, not scooped — new counterexamples to record:** AdaEAGLE
  (arXiv:2412.18910; pre-draft next-round length regression from the target's
  last-verified-token hidden state — same input class/timing, EAGLE setting,
  uncalibrated, no baselines, no skip); Judge Decoding (arXiv:2501.19309;
  linear head on target embeddings judging draft acceptability — during verify,
  relaxes losslessness); WhiFlash (arXiv:2606.07710; target-hidden-state
  pre-draft drafter routing). C10's pre-round, lossless, calibrated,
  baseline-controlled, independent-drafter cell remains unoccupied as of
  2026-07-11. Adjacent activity is monthly — supports staged-release urgency
  (D010).
- **C10 baseline hardening required before freeze:** target next-token entropy
  and top-1/top-2 margin at the last verified position are free byproducts of
  the verification pass and, per acceptance theory (arXiv:2606.30265), carry
  the governing variable. C10's baseline must include them (schema fields
  proposed for I06), else the claim is vulnerable to a one-line rebuttal.
- **ID/venue corrections for I21 queue:** Not-a-Bandit primary ID is
  2510.20064 (2506.00285 resolves to a different paper); DSpark's verified
  paper text does not mention SGLang — the landscape's "SGLang release"
  attribution needs re-verification.
- Logged by Claude, 2026-07-11.

## Evidence record template

When updating a claim, append:

```text
Claim:
Status:
Experiment IDs and code revision:
Models, datasets, and split:
Estimate and uncertainty:
Controller/tracing overhead included:
Known counterexamples:
Interpretation boundary:
Updated by/date:
```

## Run log: failed runs and negative results

Faithful record of failed/aborted runs per AGENTS.md. Numbers are copied from
immutable run logs, never hand-estimated.

### 2026-07-10 — I03 equivalence gate failed on the primary bf16 pair (unresolved → pending fp32 confirmation)

- **Run:** `CAS_GPU=A100 modal run modal_app.py::run_tests` (Modal app
  `ap-l3nL9Xr4BBy67rX0ML33vG`), Qwen2.5-7B-Instruct `a09a3545` target /
  Qwen2.5-0.5B-Instruct `7ae55760` draft, dtype **bfloat16**, eager attention,
  A100-40GB, transformers 4.46.3 / torch 2.5.1+cu124.
- **Result:** 18 passed, 7 failed in 604s. `test_bit_identical_to_greedy`
  failed for all L∈{1,2,3,4,6,8}; `test_fp_divergence_rate` = **6/12 = 0.50**
  sequence-level (ceiling 0.05). `test_skip_action_equivalent` (L=0) **passed**.
- **Root cause (identified, not a cache bug):** skip (single-token target
  forwards) is exact; every L>0 round runs the target over `gap+proposals` in one
  forward. The parallel-verify vs. sequential-decode arithmetic difference
  (D014.2) is ~1e-2 in **bf16**, not the ~1e-7 the <0.1% expectation assumed.
  Back-solving 50%/sequence over 96 tokens ⇒ **~0.7% per-token** argmax flip; one
  flip derails the rest of the greedy continuation. Cache/commit bookkeeping was
  re-derived by hand for k=0, 0<k<L, and k=L and is correct.
- **fp32 confirmation (RESOLVED):** `CAS_GPU=A100 modal run
  modal_app.py::run_tests --dtype float32` (Modal app `ap-ECNFNMgvfLdXnXdFBQYMSQ`),
  in-container dtype verified `EngineConfig.target.dtype='float32'`. **25 passed,
  0 failed**; `test_fp_divergence_rate` = 0/12. The cache/commit/rollback and
  batched-verify logic are therefore algorithmically exact; the bf16 gate failure
  was floating-point argmax-tie noise, not a bug. (A prior attempt setting
  `CAS_DTYPE` as a local env var did NOT reach the Modal container and silently
  re-ran bf16 — hence dtype is now passed as a function arg with an in-container
  assert.)
- **Remaining (reporting/calibration, not code):** (a) change
  `test_fp_divergence_rate` to a per-token rate with logged top-1/top-2 margins at
  each flip; (b) record the measured bf16 per-token rate as a dated D014 addendum
  (the "<0.1%" figure was fp32-calibrated). I03 algorithmic correctness is
  confirmed; the bf16 losslessness statement is "lossless up to logged argmax
  ties at the measured rate."
- Logged by Claude, 2026-07-10.

### 2026-07-11 — Engine review fixes + fp32 re-gate + I07 smoke characterization

- **Adversarial review (22 agents) of the D018 engine wiring** confirmed 5
  defects, one **critical and pre-existing**: tokens after an accepted mid-round
  eos were committed while greedy stops at eos — token-identity violation on any
  eos-terminating request (the prior gate's raw prompts rarely reached eos in 96
  tokens, hence never caught). Fixed same day (emit loop breaks at first
  committed eos; termination labeled from committed tokens; per-round
  start_output_pos snapshot; frontier signals moved inside measured "tracing"
  time; TokenTrace `accepted` split from counterfactual `target_match`). No
  results are invalidated: none existed. **Re-gate: fp32 73/73 passed** (Modal
  app `ap-85dmbyOiFYsL6Y1VxlTTRp`), including a new chat-template eos
  regression test, stop-rule equivalence at both extremes, and D018 field
  checks.
- **I07 smoke (cap=2, bf16, max_new=256, run `sweep-2026-07-11T172406`+fix):**
  all 9 policy trace runs sealed through the full writer path. Per-request
  equivalence vs the target-only reference: skip 2/2 identical; drafted
  policies 10/12 diverged — **consistent with the 2026-07-10 bf16
  characterization** (~0.7%/token ⇒ expected ≈83% sequence divergence at 256
  tokens; observed 83%). Divergences are fp argmax ties, flagged per-request in
  `equivalence_status`; per-round acceptance labels are exact in-context ground
  truth regardless of trajectory divergence.
- First smoke attempt OOMed (38GB): sweep-runner target-only helper lacked
  `torch.no_grad()`; fixed, plus allocator hygiene. Recorded as a failed run
  per AGENTS.md; no artifacts were sealed by the failed attempt.
- Logged by Claude, 2026-07-11.

### 2026-07-10 — I08/I09 baseline policy tooling (not claim evidence)

- Implemented resettable, pure-Python entropy-stop and rolling-acceptance
  policies plus a UCBSpec-style length-arm policy in `cas.policies`.
- UCBSpec deviations from arXiv:2505.15141: arms are the locked draft lengths
  including skip rather than heterogeneous hyperparameter specifications under
  one shared maximum; decoding is greedy exact-match; the generation cap is
  enforced by the engine; deterministic arm-order tie-breaking is used. The
  implementation retains round-robin initialization, accepted-plus-bonus output
  length as reward, and the published self-normalized confidence radius. The
  paper's stopping-time guarantee is not claimed for the changed arm definition.
- Unit tests demonstrate cold-start and post-initialization selection, threshold
  equality, rolling history, skip handling, and request reset. Reproduce with
  `PYTHONPATH=src python -m pytest tests/test_policies.py -q`.
- This is tooling only and does not change C05–C08. Held-out quality, latency,
  cold-start, and steady-state evidence require engine integration and real
  traces after I06/I07.
- Logged by Codex, 2026-07-10.
