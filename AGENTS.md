# Agent Operating Contract

These instructions apply to the entire repository.

## Before changing anything

1. Read `README.md`, `docs/RESEARCH_SPEC.md`, `docs/EXPERIMENT_CONTRACT.md`, `docs/DECISIONS.md`, and `PLAN.md` completely.
2. Choose one unblocked issue from `docs/ISSUE_BACKLOG.md` and mark its status there before implementation.
3. Check for overlapping work. Do not concurrently own the same module or artifact family.
4. Treat `/home/raghavan/13_Raghavan_Content_Aware_Speculation_Control` as read-only reference material unless a task explicitly authorizes changes to it.

## Research integrity

- Never present synthetic, smoke, estimated, or illustrative values as experimental results.
- Do not manually type result values into manuscript tables or figures.
- Keep raw artifacts immutable; transformations must be scripted and versioned.
- Use prompt-grouped splits. Token-level random splitting is prohibited.
- Include controller, tracing, synchronization, and routing overhead in end-to-end measurements.
- Record negative results, exclusions, failed runs, and counterexamples in `docs/CLAIMS_LEDGER.md`.
- Use “causal” only when an intervention is layer-specific, dose-responsive, replicated, and compared with norm-matched controls.

## Scope control

The first-pass model pairs, actions, baselines, split policy, and primary metrics are locked by `docs/EXPERIMENT_CONTRACT.md`. A change requires a dated entry in `docs/DECISIONS.md` explaining context, alternatives, and consequences.

Draft routing and EAGLE-3 are secondary. Do not start either while core exact decoding, fixed baselines, trace collection, probing, and selective speculation remain incomplete.

## Definition of done for an issue

An implementation issue is complete only when:

- its acceptance criteria are satisfied;
- relevant automated tests pass;
- a reproducible command is documented;
- produced artifacts follow the trace/metadata contract;
- limitations or deviations are recorded;
- the backlog status and affected claims are updated.

Implementation must not silently fall back to toy data or a different model. Fail clearly and explain recovery steps.

## Collaboration

- Prefer small changes scoped to one issue.
- Record ownership and status in the backlog: `OPEN`, `IN_PROGRESS`, `BLOCKED`, or `DONE`.
- State dependencies explicitly when adding an issue.
- If an interface changes, update its contract before downstream code.
- Do not rewrite unrelated user or agent work.

## Naming and disclosure policy (D008, D011)

- Do not name any conference or journal, any specific external organization or lab, or the owner's strategic goals in repository files. Refer to venues generically ("the target journal"). Specific names live only in the owner's private channel.
- Agent opinion/discussion files are non-binding. Migrate durable content into the contract, plan, or backlog and delete them (the 2026-07-10 opinion files were migrated and deleted per D012).

## Manuscript rules

- Use the target journal's unmodified official template (venue named only in the owner's private channel, per D008).
- Keep the submission PDF and repository artifacts anonymous until review policy permits disclosure. Public releases follow the staged policy in D010.
- Every empirical sentence must map to an experiment ID in the claims ledger.
- Remove illustrative figures from results; conceptual diagrams must be labeled as such.
- Do not submit while any central claim remains `UNTESTED` or lacks a reproducible supporting artifact.
