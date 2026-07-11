# Claims Ledger

Allowed statuses: `UNTESTED`, `SUPPORTED`, `PARTIAL`, `REFUTED`, and `RETIRED`.

No claim may move to `SUPPORTED` without experiment identifiers, applicable settings, uncertainty, and known counterexamples. A manuscript claim must be no broader than this record.

| ID | Proposed claim | Status | Required evidence | Experiment IDs | Counterexamples / limits |
|---|---|---|---|---|---|
| C01 | Draft hidden states contain acceptance information beyond entropy, margin, history, and domain. | UNTESTED | Prompt-grouped incremental comparison on held-out prompts | — | — |
| C02 | Acceptance information becomes accessible at identifiable draft-model layers. | UNTESTED | Layerwise probes replicated across domains and a second model setting | — | — |
| C03 | Rejection-associated directions have a controlled effect on draft–target divergence or acceptance. | UNTESTED | Dose-response intervention with random and norm-matched controls | — | — |
| C04 | Acceptance behavior differs systematically across token categories and generation phases. | UNTESTED | Acceptance atlas with paired uncertainty and annotation validation | — | Task-conditioned acceptance behavior may be partially covered by prior work (arXiv:2604.14682); verify via I21 and position as control/context if covered |
| C05 | Selective speculation with a skip action reduces wasted compute relative to adaptive-length baselines. | UNTESTED | Held-out comparison including all overhead | — | — |
| C06 | The circuit-aware controller improves net latency over the best global fixed policy. | UNTESTED | Paired held-out wall-clock study with uncertainty | — | — |
| C07 | Any controller advantage persists against the best per-domain fixed policy. | UNTESTED | Per-domain held-out comparison | — | — |
| C08 | The signal or controller transfers under domain and traffic shift without full retuning. | UNTESTED | Shift study with calibration drift and latency regret | — | — |
| C09 | The principal finding replicates outside the primary Qwen pair. | UNTESTED | Compatible Llama pair or approved Qwen-ratio fallback | — | — |
| C10 | Next-round acceptance is predictable from cached verified-context representations before any draft compute, at deployable marginal cost. | UNTESTED | Prompt-grouped held-out comparison against post-draft signals (entropy, margin, learned head), with deployed-path overhead measured (I23) | — | — |
| C11 | The identified acceptance representation transfers beyond independent drafts to a modern speculator family. | UNTESTED | Cross-speculator evaluation; extension work, only after the core evidence gate (D009) | — | — |

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
- **Next step (no claim asserted until run):** (1) re-run the gate with
  `CAS_DTYPE=float32` to confirm exact algorithmic identity (expect 0 divergence);
  (2) re-characterize bf16 with a per-token rate + logged top-1/top-2 margins at
  each flip; (3) record the measured rate as a D014 addendum and set the test
  ceiling per-token. Until (1) passes, I03 stays IN_PROGRESS and no losslessness
  claim is made.
- Logged by Claude, 2026-07-10.
