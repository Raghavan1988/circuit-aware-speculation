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
