Update: Based on guidance from Vizuara folks, i will upload a record to arXiv


# Circuit-Aware Speculation

**Working title:** *Circuit-Aware Speculation: Mechanistic Signals for Compute-Optimal Speculative Decoding*

This is the canonical research repository for a prospective journal submission on a focused question:

> Where does information about future target rejection emerge inside a speculative draft model, and can that information support compute-optimal speculation control under distribution shift?

## Status

**Active research repository with results.** The exact decoding engine, sealed
trace corpora, baselines, probes, and initial interventions are built and run on
cloud GPUs; every finding and its evidence level is tracked in
[the claims ledger](docs/CLAIMS_LEDGER.md). (The earlier "no experiments have been
run" status is superseded.) Language follows the gate policy below: until targeted
interventions pass the mechanistic gate (G2), the internal quantity is described as
a **representation** or **diagnostic signal**, not a "mechanism" or "circuit".

### What has been established (from this repo's sealed, script-generated artifacts)

- **Exact engine.** Greedy speculative decoding proven token-identical to
  target-only decoding (73/73 full-precision checks). Sealed, checksummed traces:
  a v1 corpus (644 prompts; ~302k labeled rounds) plus a larger v2 corpus (1,494
  prompts, 7 task types).
- **Adaptive draft length wins (RQ2).** A free entropy-based stopping rule beats
  the best fixed draft length by ~**+11.2%** efficiency (~62% less wasted
  drafting) on prompt-grouped held-out data, frozen dev→test, across corpora.
  Bandit-style baselines settle at the best fixed length and do not beat it.
- **Draft specialization does not help (RQ3).** A same-size general draft ties or
  beats topic-specialist drafts, even on the specialist's own topic.
- **Acceptance atlas (supported).** Per-token-category acceptance ranges ~0.52–0.88
  over ~302k rounds — structural tokens easy, meaning-heavy tokens hard.
  Latency-independent; frozen-test replicated (category axis), same qualitative
  picture on both the Qwen and Llama traces.
- **Pre-round predictability (supported; first-token scope).** Next-round
  acceptance is predictable from cached verified-context representations *before*
  any draft compute, at near-zero marginal cost; frozen dev→test transfer under two
  protocols. Cheap free signals predict acceptance about as well as internal-probe
  features (internal probing adds at most +0.006).

### In progress / open

- **Systems latency (G3-gated).** Efficiency gains are real; a full end-to-end
  wall-clock claim is still gated. The harness's small-model step was
  launch-overhead-bound; a fixed-shape, CUDA-graph draft step (issue I26) cured
  this on H100 (~**6.2×** faster per token in an initial single-prompt
  measurement), and replaying at that measured cost the adaptive-length opportunity
  is ~46% vs ~5% on the old setup. A multi-prompt, bootstrap-CI, context-swept run
  and the end-to-end controller stopwatch (which needs a lossless fixed-shape
  rollback path) remain to be done.
- **Replication** across a second model family (Llama) is wired and tested but
  blocked on model access; results are currently one family (Qwen).
- **Manuscript** figures and finalization; full v2-corpus run.

The older `/home/raghavan/13_Raghavan_Content_Aware_Speculation_Control` project is a toolkit and source of ideas. Its smoke metrics, illustrative figures, and draft manuscript are not scientific evidence and must not be copied into results.

## Research questions

1. **Localization:** At which draft-model layers does information predictive of target acceptance become linearly accessible?
2. **Mechanism:** Which token-level computations distinguish accepted from rejected proposals beyond entropy and probability margin?
3. **Dynamics:** How do rejection mechanisms change across syntax boundaries, semantic transitions, factual entities, reasoning steps, and code delimiters?
4. **Control:** Can a calibrated internal signal choose whether to skip, continue, or stop speculation while improving net latency over strong adaptive baselines?
5. **Robustness:** Does the mechanism transfer across domains, traffic shifts, target–draft ratios, and model families without retuning?

## Planned study

- **Primary pair:** Qwen2.5-7B-Instruct target with Qwen2.5-0.5B-Instruct draft.
- **Replication:** a tokenizer-compatible Llama 8B/1B pair; use a second Qwen size ratio if compatibility blocks reliable evaluation.
- **Actions:** skip speculation or draft `{1, 2, 3, 4, 6, 8}` tokens.
- **Workloads:** code, math, chat, and one held-out domain.
- **Baselines:** target-only, all fixed lengths, best global and per-domain fixed policies, entropy, acceptance history, BanditSpec-style adaptation, a SpecDec++-style learned acceptance head, the proposed circuit-aware controller, and an offline oracle.
- **Systems outcomes:** net wall-clock speedup, TPOT, throughput, wasted draft tokens, latency regret, memory, and controller overhead.
- **Mechanistic outcomes:** layerwise predictive performance, calibration, cross-domain transfer, token-category behavior, and controlled-intervention effect sizes.

The complete protocol lives in [the experiment contract](docs/EXPERIMENT_CONTRACT.md), and every recorded decode step must conform to [the trace schema](docs/TRACE_SCHEMA.md).

## Evidence standards

We distinguish three levels of evidence:

- **Predictive correlation:** a signal predicts acceptance on prompt-grouped held-out data.
- **Representational localization:** acceptance information is more accessible at particular layers, with leakage-safe controls.
- **Causal evidence:** a targeted intervention produces a layer-specific, dose-responsive, replicated change relative to norm-matched random controls.

Probe accuracy alone is not causal evidence. Negative results and failed interventions must be preserved in the claims ledger.

## Reproducibility rules

- Never hand-enter result tables or paper figures.
- Generate every reported number from immutable raw artifacts through versioned scripts.
- Split data by prompt, never by token, to prevent adjacent-token leakage.
- Measure drafting, verification, controller, and end-to-end latency separately with device synchronization.
- Include controller and activation-capture overhead in net performance.
- Report failures, exclusions, and counterexamples.
- Keep pre-submission manuscript artifacts anonymous.

## Seven-day draft milestone

Progress against this milestone is tracked in [the claims ledger](docs/CLAIMS_LEDGER.md):
items 1–3 and much of 4–6 are substantially complete (engine, baselines, atlas,
leakage-safe probes, the compute-optimal length rule, initial interventions);
end-to-end wall-clock validation, cross-family replication, and the final
manuscript remain open.

1. Build exact target–draft decoding and timing for the primary Qwen pair.
2. Establish fixed and adaptive baselines and collect the acceptance atlas.
3. Run leakage-safe layerwise probing and lock the controller signal.
4. Implement compute-optimal selective speculation, including the skip action.
5. Run interventions, traffic shifts, and begin replication.
6. Complete essential ablations and produce artifact-derived figures.
7. Reproduce the primary table from a clean environment and assemble an anonymous manuscript draft.

This is a draft milestone, not a hard submission deadline. Submission requires the evidence gate in [the research specification](docs/RESEARCH_SPEC.md).

## Taking over an issue

Agents should first read [AGENTS.md](AGENTS.md), then select an unblocked item from [the issue backlog](docs/ISSUE_BACKLOG.md). Record consequential choices in [the decision log](docs/DECISIONS.md) and update affected claims in [the claims ledger](docs/CLAIMS_LEDGER.md).

## Repository map

- [Research specification](docs/RESEARCH_SPEC.md): thesis, scope, study design, and submission gate.
- [Experiment contract](docs/EXPERIMENT_CONTRACT.md): locked first-pass experimental protocol.
- [Trace schema](docs/TRACE_SCHEMA.md): required run-, request-, round-, and token-level fields.
- [Issue backlog](docs/ISSUE_BACKLOG.md): dependency-aware implementation work.
- [Claims ledger](docs/CLAIMS_LEDGER.md): evidence status for every proposed conclusion.
- [Decision log](docs/DECISIONS.md): durable methodological and architectural decisions.
- [Execution plan](PLAN.md): schedule mapped to backlog issues, decision gates, budget, and the competitive landscape.
- [Manuscript](paper/main.tex): anonymous draft; build it with `make -C paper` (see below).

## Building the manuscript

```sh
make -C paper           # build paper/main.pdf from main.tex + tracked figures
make -C paper figures   # regenerate paper/figures/*.pdf from sealed artifacts
make -C paper artifacts # pull those artifacts from the Modal volume first
make -C paper check     # prose gates: readability score + claim invariants
```
file name generated would be main.pdf

Figure PDFs are tracked, so a plain `make -C paper` works on a fresh clone with
no Modal access. `figures` and `artifacts` are only needed when the underlying
analysis JSONs change. The build fails on unresolved references rather than
silently emitting `??`.

## License and citation

Code and documentation are released under the [Apache License 2.0](LICENSE). Citation metadata is provided in [CITATION.cff](CITATION.cff); update authorship before public release.
