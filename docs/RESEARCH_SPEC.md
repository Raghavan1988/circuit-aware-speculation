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

## Secondary resource framing

Every speculative round yields a ground-truth label of small-model/large-model disagreement at zero annotation cost. The trace corpus is therefore independently valuable beyond this study: a naturally-labeled, prompt-grouped dataset for studying whether small models internally represent their disagreement with larger models, connecting to the introspection and calibration literature, and usable by others without GPU access. Package and document it accordingly (issue I24).

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

A distinguished candidate for the deployed signal is **pre-round prediction from cached representations**: predicting the next round's acceptance from already-computed verified-context representations at selected layers, so the decision precedes any draft compute and its marginal cost is near zero. Post-draft signals (entropy, margin, a learned head on drafted-token states) pay the draft forward pass first. Heavy activation capture remains offline-only and must never run inside deployed timing.

## Differentiation requirements

Adjacent work already covers single components. None of the following may be the sole claimed contribution: an entropy–acceptance correlation; a layerwise probe that predicts acceptance; an acceptance atlas; a bandit length controller; prompt-domain routing; a skip action; or beating an untuned fixed-length baseline. These are components, controls, or artifacts — not a paper identity.

The defensible thesis: identify a causally validated, transferable internal computation associated with draft–target agreement; explain when it fails; and show either that a cheap approximation improves real inference scheduling or that the mechanism establishes a principled limitation on activation-based control.

Before freezing novelty claims: verify the planning-pass landscape additions (arXiv:2603.01639 adaptive drafting via RL; arXiv:2605.02888 draft confidence under KV compression; arXiv:2604.14682 task-conditioned acceptance dynamics; arXiv:2606.30265 theoretical treatment of acceptance) against primary sources and re-scan for newer work (issue I21). Reproduce the closest published baseline (a SpecDec++-style learned acceptance head) early (issue I22).

## Decision gates (adopted per D009)

- **G1 — Predictive validity.** Proceed only if internal features improve prompt-grouped held-out prediction after controlling for entropy, margin, recent acceptance, token identity/category, domain, and position — or yield a stable, replicated negative result.
- **G2 — Mechanistic validity.** Use "mechanism" or "circuit" only if interventions are layer-specific, dose-responsive, replicated, and survive random and norm-matched controls; otherwise say "representation" or "diagnostic signal".
- **G3 — Systems validity.** A controller claim requires net wall-clock improvement after feature extraction, transfer, synchronization, and control overhead; acceptance-rate improvement alone is insufficient.
- **G4 — Impact and reusability.** The broader artifact release additionally requires at least one of: a serving-engine integration; a result beyond the primary 7B scale; a new theoretical connection; or a reusable artifact with independent value (trace benchmark, profiler, explorer).
- **G5 — Submission readiness.** Submit only after the clean-reproduction audit, second-model or transfer evidence, contemporary-baseline comparison, and a claims-ledger audit pass. Release timing follows the staged policy in D010.

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
- the manuscript is anonymous and uses the target journal's official template (venue unnamed in repository files, per D008);
- central claims are not `UNTESTED` in the claims ledger.

Beating the controller baseline is not required for an honest diagnostic paper, but a negative systems result must be paired with a clear, replicated mechanistic or empirical insight.
