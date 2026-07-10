# Research Specification

## Thesis

Speculative decoding wastes compute when the draft continues through states that are likely to diverge from the target. Existing adaptive methods typically use output confidence, acceptance history, or online arm rewards. This project asks whether rejection information has a stable internal representation in the draft model, whether controlled interventions clarify its role, and whether the cheapest validated signal improves compute-aware decoding.

The intended contribution is not another generic bandit controller. It is a mechanistic account of acceptance paired with a controller whose action and overhead follow from that account.

## In scope

- Exact greedy speculative decoding with a runtime-selectable proposal length.
- A skip-speculation action for states where drafting has negative expected payoff.
- Token-level acceptance traces and an acceptance atlas.
- Layerwise, prompt-grouped acceptance probes.
- Incremental-information tests beyond entropy, margin, history, and domain.
- Controlled activation interventions with random and norm-matched controls.
- Compute-aware selection using measured drafting, verification, and controller costs.
- Domain shift, traffic shift, and at least one replication axis.

## Deferred scope

- Training domain-specific draft models.
- Full EAGLE-3 integration or tree-policy optimization.
- Large-batch serving beyond a small validation setting.
- Sampling-based equivalence as a primary result.
- Multimodal speculative decoding.
- Joint target-model routing and test-time scaling.

Deferred items may be reconsidered only after the central evidence gate is met.

## Acceptance atlas

Collect real decode traces and annotate positions as punctuation/whitespace, code delimiters, function words, content words, numbers/operators, named entities, sentence or clause boundaries, reasoning transitions, and repeated/copied spans. Report acceptance, target–draft divergence, chosen action, and wasted compute by category and generation phase.

Annotations should be automatic, versioned, auditable, and allowed to overlap where linguistically appropriate. Manual inspection may validate a stratified sample but must not silently overwrite automatic labels.

## Mechanistic analysis

### Localization

Train regularized probes at selected early, middle, and late layers for next-token acceptance and accepted-run-length bins. Compare against entropy, top-1 margin, recent acceptance, prompt/domain metadata, final-layer features, and combined models.

Primary evaluations are AUROC, area under the precision–recall curve, Brier score, expected calibration error, and cross-domain degradation. All splits are grouped by prompt.

### Intervention

Derive a rejection-associated direction on development data. At selected layers, apply mean ablation, projection removal, or controlled steering at multiple strengths. Compare against random and norm-matched directions and measure changes in proposed tokens, entropy, target divergence, and acceptance.

Interpretation tiers:

1. predictive correlation;
2. representational localization;
3. causal evidence only after controlled replication.

## Controller

For each decoding state, estimate the payoff of actions `skip`, `1`, `2`, `3`, `4`, `6`, and `8` using expected accepted tokens and measured costs. The deployed signal must be chosen using development data and frozen for the main test; an explicitly labeled online-recalibration variant may be evaluated separately.

If internal features improve prediction but their overhead erases latency gains, report the mechanism–systems trade-off. If they do not improve prediction beyond entropy, report that negative finding and use the cheapest sufficient signal.

## Essential comparisons

- Target-only autoregressive decoding.
- Every fixed candidate length.
- Best global fixed policy.
- Best per-domain fixed policy.
- Entropy threshold.
- Acceptance-history policy.
- BanditSpec-style adaptive policy.
- Learned output-confidence predictor.
- Circuit-aware compute-optimal policy.
- Offline compute oracle, labeled as an upper bound.

## Evidence gate for submission

Submission may proceed only when:

- every result is generated from saved artifacts;
- output equivalence is verified;
- the primary controller includes all overhead and is compared fairly with modern adaptive baselines;
- the central finding replicates across two model pairs or an equivalently strong held-out transfer setting;
- mechanistic language matches intervention strength;
- failures and negative results are disclosed;
- a clean environment reproduces the primary table;
- the manuscript is anonymous and uses the official TMLR template;
- central claims are not `UNTESTED` in the claims ledger.

Beating the controller baseline is not required for an honest diagnostic paper, but a negative systems result must be paired with a clear, replicated mechanistic or empirical insight.
