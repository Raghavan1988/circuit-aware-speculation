# Decision Log

Use dated entries with context, decision, alternatives, and consequences. Do not rewrite old decisions; append superseding records.

## D001 — Canonical repository

- **Date:** 2026-07-10
- **Context:** The earlier content-aware speculation project contains a roadmap, smoke implementation, and illustrative manuscript.
- **Decision:** `/home/raghavan/circuit-aware-speculation` is the canonical research repository. The earlier project is a read-only toolkit/reference and its placeholder results are not evidence.
- **Alternatives:** Continue in the toolkit; copy and clean the scaffold.
- **Consequences:** This repository begins with a research contract and no implementation or inherited claims.

## D002 — Primary thesis

- **Date:** 2026-07-10
- **Context:** Adaptive length and draft selection are substantially covered by BanditSpec, SpecDec++, and Not-a-Bandit.
- **Decision:** Center the work on mechanistic localization and controlled validation of speculative-token acceptance, followed by compute-aware control.
- **Alternatives:** A systems-only joint controller; a broad diagnostic survey.
- **Consequences:** Generic bandit control is a baseline, not the novelty claim.

## D003 — Secondary routing scope

- **Date:** 2026-07-10
- **Context:** Multiple specialized drafts multiply engineering, memory, and comparison costs.
- **Decision:** Defer multi-draft routing and EAGLE-3 until the core acceptance study meets its evidence gate.
- **Alternatives:** Preserve the original joint routing-and-length scope.
- **Consequences:** The initial action space is skip-or-length for one target–draft pair.

## D004 — Timeline

- **Date:** 2026-07-10
- **Context:** The starting point contains no real inference integration or measured results.
- **Decision:** Treat one week as a draft-and-experiment milestone, not a hard submission deadline.
- **Alternatives:** Submit after seven days regardless of evidence.
- **Consequences:** Journal submission waits for the explicit evidence gate.

## D005 — Initial repository boundary

- **Date:** 2026-07-10
- **Context:** Future agents need an authoritative handoff without inheriting premature code choices.
- **Decision:** The initial commit contains documentation contracts only: README, agent instructions, protocol, schema, backlog, claims, and decisions.
- **Alternatives:** Operational code skeleton; copied toolkit; minimal runnable baseline.
- **Consequences:** Implementation begins through backlog issues with interfaces chosen deliberately.

## D006 — Publication state

- **Date:** 2026-07-10
- **Context:** No GitHub destination or visibility was authorized.
- **Decision:** Initialize a local Git repository only.
- **Alternatives:** Immediately publish a remote repository.
- **Consequences:** Remote creation and pushing require a separate explicit task.

## D007 — Project license

- **Date:** 2026-07-10
- **Context:** The GitHub repository was initialized with Apache License 2.0, and the repository owner selected it as the intended project license.
- **Decision:** License the canonical repository under Apache-2.0 rather than MIT.
- **Alternatives:** Retain the initially drafted MIT license.
- **Consequences:** The remote Apache license is authoritative; README and citation metadata use the SPDX identifier `Apache-2.0`.

## D008 — Repository naming neutrality

- **Date:** 2026-07-10
- **Context:** Owner instruction (2026-07-10): repository files must not name any conference or journal, any specific external organization or lab, or the owner's strategic goals. Several files previously named the target journal.
- **Decision:** Repository files refer to venues generically ("the target journal", "a rolling-review journal"). Specific venue names live only in the owner's private channel. Strategy substance is retained in neutral terms (impact, reusability, adoption). Existing references were redacted in place, including one word in D004's consequences; this entry records that redaction so the append-only convention stays honest.
- **Alternatives:** Keep precise names in internal planning documents (rejected by owner instruction); delete strategy content entirely (loses substance).
- **Consequences:** Agents must not reintroduce venue or organization names or strategic-goal statements into repository files. Anonymized files also simplify double-anonymous compliance if the repository is shared during review.

## D009 — Revised decision gates and track structure

- **Date:** 2026-07-10
- **Context:** Two agent reviews (recorded in the non-binding opinion files) converged on raising the decision bar beyond "internal signals beat entropy with affordable overhead" and on structuring the work as tracks with one headline differentiator.
- **Decision:** Adopt gates G1–G5 as specified in `RESEARCH_SPEC.md`, superseding the single Day-5 gate in earlier planning. Adopt three tracks: Track A (publishable scientific core — the existing contract), Track B (one high-risk headline differentiator; primary bet: pre-round acceptance prediction from cached verified-context representations, with the causal-circuit analysis underneath it; transfer to a modern speculator family as the extension), Track C (reusable release artifacts, started only after the core stabilizes). Timeline reinterpretation: roughly seven days for core implementation and preliminary evidence, 10–14 days for a first complete scientific draft, and additional weeks for release engineering.
- **Alternatives:** Keep the original single gate; pursue several Track B differentiators simultaneously (rejected: scope explosion).
- **Consequences:** New backlog issues I21–I24; new claims C10–C11; `PLAN.md` §5 defers to the gates in `RESEARCH_SPEC.md`; the contract amendments in this pass (policies 8–9 sharpened, schema fields added) are authorized by this entry.

## D010 — Staged release policy

- **Date:** 2026-07-10
- **Context:** The timing of public communication (anonymous submission versus public preprint and artifact release) needed an explicit policy; closely adjacent work is appearing frequently.
- **Decision:** Staged release. A public preprint may be released once gates G1–G3 and the clean-reproduction audit pass. The broader artifact release (any serving-engine integration, benchmark publication, explorer, technical article) follows once G4 passes. Escape hatch: if directly competing work appears publicly, release immediately at whatever gate level is honestly met, with claims scoped exactly to the evidence in hand. The submitted manuscript remains anonymous per the target journal's policy.
- **Alternatives:** Hold the preprint until the full artifact is ready (risks losing priority); release at will (risks unsupported claims).
- **Consequences:** Scientific priority is decoupled from engineering polish; the claims ledger governs what any release may state.

## D011 — Document hierarchy and status of opinion files

- **Date:** 2026-07-10
- **Context:** The repository now contains a research contract, an execution plan, session-bootstrap guidance, and two agent opinion files written during planning.
- **Decision:** Authority order: (1) the research contract (`AGENTS.md` + `docs/`) is scientific law; (2) `PLAN.md` is the execution calendar; (3) the standards adopted in D009 are the ambition bar; (4) `docs/CLAIMS_LEDGER.md` is the arbiter of what may be claimed. `CLAUDE.md` is session bootstrap and must be re-verified when stale. `CLAUDE_OPINION.md` and `CODEX_OPINION.md` are non-binding discussion records: they predate D008, retain old naming, and must not be treated as policy or shared publicly without redaction or removal.
- **Alternatives:** Fold the opinions into the contract; delete the opinion files.
- **Consequences:** Conflicts resolve upward through the hierarchy; the opinion files are excluded from D008 enforcement but are flagged for redaction or removal before any public release of the repository.

## D012 — Opinion files migrated and deleted

- **Date:** 2026-07-10
- **Context:** D011 flagged the two agent opinion files for redaction or removal. The owner directed that any new/unique content be moved to appropriate files and the opinion files deleted.
- **Decision:** Unique content was migrated: secondary resource framing → `RESEARCH_SPEC.md`; an additional landscape reference (Talon — heterogeneous drafting with asynchronous execution; name collision with TALON arXiv:2601.07353), alternative Track B bets, and a risk register → `PLAN.md`; headline-figure specification → backlog I18; technical article, upstream engagement, governance-document inclusion, and resource-framing documentation → backlog I24. Purely historical discussion (plan comparisons, correction adjudications, agent-to-agent debate) was not retained. Both files were then deleted.
- **Alternatives:** Keep the files private and untracked; redact them in place.
- **Consequences:** The repository contains no files naming venues, external organizations, or owner goals. References to the opinion files in D011 and earlier records are historical.

## D013 — Step-1 infrastructure and data choices

- **Date:** 2026-07-10
- **Context:** Step 1 (issues I01, I05) requires binding provider, hardware, and dataset choices before implementation; the contract requires recording them.
- **Decision:** Provider: Modal (already configured locally; code-defined images; weights cached on a persistent volume). GPU: H100 80GB (~1.5–2× faster wall-clock than A100 at similar total cost; fewer elapsed days). Chat domain: MT-Bench first turns (80) supplemented from a documented open chat-prompt source to ~150 prompts. Held-out domain: summarization prompts (never used in any fitting). Default generation cap: 256 new tokens, to be confirmed or revised at the first full sweep (contract requires the cap decision recorded before that sweep).
- **Alternatives:** RunPod or hybrid (cheaper sustained $/hr, more setup); A100 80/40GB; MT-Bench-only chat (80 prompts, weaker statistics); retrieval-grounded QA as held-out.
- **Consequences:** Budget outlook shifts from 60–100 A100-hours to roughly 40–60 H100-hours (similar ~US$150–300 total, less wall-clock). The Llama replication pair (I17) still requires the owner to refresh the HF token.

## D014 — Step-2 engine implementation choices

- **Date:** 2026-07-10
- **Context:** The exact speculative-decoding engine (issues I02–I04) needs three implementation choices fixed before coding. The contract specifies greedy exact-match decoding, token-identical outputs, and overhead-inclusive timing, but not the backend, equivalence policy, or draft-cache strategy.
- **Decision:**
  1. **Attention backend: eager.** Chosen for mechanistic-interpretation friendliness — forward hooks reach any layer for Step-5 activation capture — and full determinism. Absolute latency is slower than fused kernels, but all policies share the backend so paired comparisons stay fair; production-realistic latency is Tier-2's job (serving-engine integration under load, per D010/G4).
  2. **Equivalence policy: bit-exact with logged exceptions.** Assert token-identical output to target-only greedy; investigate every mismatch; document genuine floating-point argmax-tie flips (parallel-verify vs. sequential arithmetic differ at ~1e-7) with a measured divergence rate (expected <0.1%). Reflects reality honestly and does not distort latency, unlike fp32/deterministic zero-tolerance verification. Distributional equivalence is reserved for the deferred sampling extension.
  3. **Draft KV cache: persistent with rollback, validated against a stateless oracle.** The draft keeps its cache across rounds and truncates in lockstep with the target after each round. Rationale: stateless re-prefill imposes a fixed per-round overhead independent of draft length ℓ that would inflate inter-token latency AND, more damagingly, tilt the per-round-optimal-length landscape toward longer ℓ while penalizing the skip/short-length policies — biasing the adaptive-vs-fixed comparison in our own favor. Build the stateless version first as a correctness oracle; prove the persistent version emits identical tokens; measure only with the persistent version.
- **Alternatives:** SDPA or FlashAttention-2 (faster, but capture-hostile / would force a backend switch between timing and probing); strict zero-tolerance or distributional equivalence; stateless re-prefill.
- **Consequences:** I03 correctness tests include a stateless-vs-persistent token-identity check and an FP-divergence-rate measurement. Timing numbers are relative, not production-absolute; the absolute claim is deferred to the serving-engine tier. Any later switch to a fused backend for a specific measurement must be recorded and must re-verify equivalence.

## D015 — Mechanistic-analysis tooling: nnsight for capture/intervention, scikit-learn for probes

- **Date:** 2026-07-10
- **Context:** Steps 5–6 (activation capture I10, probes I12/I13, interventions I15, pre-round prediction I23) need a tooling decision. Options considered: TransformerLens, nnsight, hand-rolled forward hooks, pyvene.
- **Decision:**
  1. **Capture + intervention: `nnsight`.** It wraps the *actual* Hugging Face module, so activations and interventions come from the same model whose exact greedy equivalence was proven in D014/I03 — numerical identity is preserved. Owner directive (2026-07-10): prefer a maintained library over a from-scratch hook layer.
  2. **Probes / calibration (I12/I13): `scikit-learn`** (already pinned): regularized logistic/linear probes, AUROC/AUPRC/Brier/ECE. Standard ML, no bespoke code.
  3. **TransformerLens is rejected for the correctness-critical path.** `HookedTransformer` re-implements the model (LayerNorm folding, weight centering), so its logits are not bit-identical to HF Qwen2.5. Because acceptance is a razor-thin argmax comparison — the bf16-vs-fp32 gate this session flipped ~0.7% of tokens from a ~1e-2 logit difference (see CLAIMS_LEDGER run log) — a systematic ~1e-4 deviation from TL folding could silently flip acceptance labels, so TL activations would describe a *different* model than the one generating our labels. TL remains acceptable **offline only** (logit lens, exploratory activation patching), never as the source of activations backing a published probe/intervention.
- **Alternatives:** TransformerLens (rejected above; reimplementation breaks exact-equivalence); raw PyTorch forward hooks (leanest, but owner prefers a maintained library); pyvene (config-driven interventions — heavier, less transparent for the G2 causal audit than nnsight tracing).
- **Consequences:** Add `nnsight` to the pinned env at I10 (re-lock the image). I10 must still assert capture does not change output tokens (read-only tracing), and I15 interventions must remain explicit enough to audit for the G2 causal gate. The earlier proposed TL-vs-HF fidelity check is moot (TL not used on the critical path); an equivalence assertion for nnsight capture replaces it.

## D016 — Token annotation interface (I11)

- **Date:** 2026-07-10
- **Context:** Issue I11 needs a versioned, overlapping category set and generation-phase label for the token-trace schema. The annotator must integrate with the I06 trace writer without depending on torch or loading a tokenizer inside the hot path.
- **Decision:**
  1. **Module:** `cas.annotate` with pure functions of the decoded token stream.
  2. **Signature:** `annotate_token(token_id: int, piece: str, position: int, context_pieces: list[str]) -> AnnotatedToken` where `AnnotatedToken` has `categories: frozenset[str]`, `phase: str`, `category_set_version: str`, and `phase_set_version: str`.
  3. **Overlap preserved:** categories are a set (may be empty only for pathological inputs; typical tokens emit ≥1 label). No mutually exclusive forced taxonomy.
  4. **Versions:** `CATEGORY_SET_VERSION = "v1.0.0"` and `PHASE_SET_VERSION = "v1.0.0"`; bump on any rule or bin change (TRACE_SCHEMA invariant 7).
  5. **Phases:** absolute bins over 0-indexed generation position — `prefix` [0, 32), `mid` [32, 128), `late` [128, ∞). Offline relative tertiles via `annotate_phase_relative` / `annotate_sequence(..., relative_phase=True)`.
  6. **Categories (v1):** whitespace, punctuation, code_delimiter, function_word, content_word, number, operator, named_entity, sentence_boundary, clause_boundary, reasoning_transition, repeated_span, newline, special — matching the acceptance-atlas list in RESEARCH_SPEC.md plus newline/special operational labels.
  7. **Validation:** seeded stratified golden sample in `tests/test_annotate.py`; agreement is script-computed, never hand-typed into result tables.
- **Alternatives:** spaCy/NER pipeline (heavier, non-reproducible without model pins); mutually exclusive BIO tags (loses ambiguity the atlas needs); phase as fraction of max_new_tokens only (not streaming-friendly).
- **Consequences:** I06 writer should call `annotate_token` per proposed/target token and store `categories` (sorted list or set), `phase`, and both version strings. No edits to `cas.trace` from this decision — seam only. Re-annotation of historical traces is possible offline via `annotate_sequence` when pieces are retained.

## D017 — Per-token draft stop-rule seam (I08)

- **Date:** 2026-07-10
- **Context:** The entropy baseline decides before each proposed draft token, while the existing action-policy seam selects only a maximum length at round start. Codex owns the pure policy object; Claude owns engine integration.
- **Decision:** Define `cas.policies.StopContext` with draft index, current entropy, current margin, and previously proposed token IDs, plus a resettable `StopRule` callable returning `True` to stop before the current proposal. The first implementation, `EntropyStopRule`, stops only when entropy strictly exceeds a development-selected frozen threshold.
- **Alternatives:** Encode entropy stopping as a round-level action (cannot react within a draft); edit the engine before its owner ratifies the consult point (would overlap active work).
- **Consequences:** Claude may wire this seam into the draft loop after I06. Equality at the threshold continues drafting; tests lock that boundary and request-reset behavior. No engine file was changed by I08.

## D018 — Measurement-first prioritization and schema consequences (owner-approved)

- **Date:** 2026-07-11
- **Context:** A verified 25-agent literature sweep (2026-07-11, ledger note of same date) found: the C01 instrument is published (a calibrated linear acceptance head on draft hidden states, arXiv:2607.05147); the controlled incremental-information comparison and the token-category acceptance atlas remain unoccupied; the pre-round cell (C10) is narrowed by arXiv:2412.18910 and arXiv:2501.19309 but open. Three agents' independent confidence estimates converged: positive-probe claims sit at ~35–65%; measurement-shaped deliverables sit at ~85–95%. Owner set an explicit bar (≥80% success probability, low cost) and approved the resulting ordering on 2026-07-11.
- **Decision:**
  1. **Execution order:** (1) atlas + surface-baseline strength + oracle headroom from the I07 traces; (2) controlled incremental-information study (C01 reframed as a measurement with the negative outcome pre-registered as publishable); (3) C10 pre-round prediction and skip-gating run as ride-along bets, not gating anything; (4) the atlas-derived category×phase lookup controller as a near-free extra policy. G2 (interventions) and G3 (end-to-end mechanistic-controller win) are demoted to exploratory with tripwires: G2 work starts only if the incremental study finds a localized residual.
  2. **Schema consequences (I06, before first sweep):** round records store the full per-position draft/target match vector (not only the accepted-prefix length — required for counterfactual all-action labels), the target's frontier-token entropy and top-1/top-2 margin from the verification pass (free byproducts; mandatory members of the hardened C10 baseline), and an activation-artifact identifier slot for later target-side frontier-state retention (I10).
  3. **Methodology ratified from the three-agent convergence:** survival/hazard formulation for accepted length; counterfactual full-information labels (a max-length round labels all shorter actions up to the first rejection — valid for per-round labels, not for multi-round trajectories); decision-regret reporting alongside AUROC; kill-chart evaluation order (surface stack first, wall-clock last); calibrate-and-abstain fallback for any deployed learned policy.
- **Alternatives:** Bet the headline on the probe/causal/latency chain (compound probability ~15–25%, fails the owner's bar); postpone schema fields until probing starts (would force a re-sweep of all traces).
- **Consequences:** I06 acceptance criteria extend to the fields above. I09's reward is repaired per Codex's note before any controller comparison uses it. The primary manuscript target remains the rolling-review journal (PLAN §6); its evidence bar favors exactly this measurement-first package. CLAUDE_IDEAS.md holds the full scoring table; the claims-ledger 2026-07-11 note holds the novelty impacts.

## D019 — Latency-aware I09 reward and accepted-length policy contracts

- **Date:** 2026-07-11
- **Context:** The round-1 I09 selector maximized emitted tokens per round. Because that reward does not charge for the requested action's latency, it is not a speed objective and can prefer an aggressive action even when skip is faster. D018 also required the survival/hazard and calibrate-and-abstain methodology to have a precise, leakage-safe policy interface before controller fitting.
- **Decision:**
  1. Preserve the round-1 selector as the explicitly named `UCBSpecNaive`. The repaired `UCBSpecPolicy` requires an immutable per-action cost profile measured only on development traces and uses non-terminal emitted tokens divided by the requested action's measured round cost. Its confidence radius is scaled into the same throughput units; no default or invented cost values are permitted.
  2. Define `q_j` as conditional continuation, `P(A >= j | A >= j - 1, x)`, so rejection hazard is `1 - q_j`, survival is `S_k = product(q_1, ..., q_k)`, and action utility is `(1 + sum(S_k, k <= L)) / cost(L)`. Exact utility ties choose the shortest action.
  3. Same-round counterfactual labels stop at the first rejection. An observed rejection labels longer maximum-length actions because they would stop at the same position; a shorter all-accepted row is right-censored. Terminal/capped rows are excluded from nominal-yield fitting, and counterfactual labels must not be rolled into multi-round trajectories.
  4. Calibration folds keep prompt groups intact and balance class/row counts. Base scores supplied to calibration must already be out-of-fold or come from a disjoint development calibration split. Controller features must be available before the current action; canonical same-round outcome fields are rejected.
  5. The optional probe-as-prior interface uses confidence-gated pseudo-observations. Below the frozen confidence threshold its prior weight is exactly zero, yielding the same choices as realized-history UCB.
- **Alternatives:** Retain raw emitted count as the main reward; silently fall back to raw reward when costs are missing; hardcode illustrative timings; use label-blind or row-random calibration folds; treat short all-accepted or terminal rows as fully observed; construct counterfactual multi-round rollouts.
- **Consequences:** I09 is complete at the pure-policy layer, while measured development costs and engine integration remain downstream work. The survival and prior modules are scaffolding, not completion of I14 or evidence for controller claims. The latency-aware radius and draft-length arm definition remain disclosed deviations from arXiv:2505.15141, and terminal rounds require an explicit future objective if they are to be modeled rather than excluded.
