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

## D020 — Project title is G2-gated; public naming decided at release time

- **Date:** 2026-07-11
- **Context:** Owner asked whether "circuit-aware speculation" remains an honest name given D018's measurement-first plan, under which the causal-intervention track (G2) is exploratory with ~15–25% odds. The G2 wording gate already prohibits "circuit"/"mechanism" language in claims until interventions pass; a title is the most prominent claim of all.
- **Decision:** The local repository name stays as-is (not itself a public claim). Every public-facing name — manuscript title, preprint, released artifact/benchmark (I24) — follows the same wording gate as prose: "circuit(-aware)" appears only if G2 passes before that release. Default public names if G2 has not passed: "acceptance-aware speculation," "pre-round speculation control," or artifact-led ("the acceptance atlas"); final pick is the owner's at preprint time. No agent bakes "circuit" into manuscript or release-artifact titles, filenames, or package names before the gate.
- **Alternatives:** Rename the repository now (churn for no external benefit); keep "circuit-aware" publicly as aspiration (violates the ledger rule that no claim exceeds evidence).
- **Consequences:** I19/I24 acceptance criteria implicitly include a title-vs-ledger check; the clean-reproduction audit (I20) verifies it. If the G2 tripwire fires and interventions replicate, the original name is earned and may be restored everywhere.

## D021 — Measurement-only dual-mode execution seam (eager capture / fast timing)

- **Date:** 2026-07-12
- **Context:** T3.4/T3.4b (see CLAIMS_LEDGER Run log 2026-07-12) established that the 0.5B draft forward is **launch-bound** under eager execution — ≈24 ms/token, invariant to signals/syncs, ≈ the 7B verify because cost tracks layer count not parameters — and that fused SDPA attention does not fix it (still ≈25 ms). The M3 oracle headroom (~5%, best fixed = skip) is therefore a harness artifact: the same sealed acceptance labels imply 25–46% headroom at a realistic draft cost (2–8 ms/token). To characterize the deployed-regime headroom and unblock an honest M3 re-decision, a fast (compiled / graph-captured) draft path is needed — **without** abandoning the eager, hookable path that activation capture (D014, D015; T4/I10) depends on.
- **Decision:** Add a **measurement-only** dual-mode seam to the model loader: `ModelSpec.compile_mode` (`None` = eager, hookable, the unchanged default; a `torch.compile` mode string = fast timing path), composed with the existing `attn_implementation`. This fast mode exists for **latency characterization only**. It is **not** serving-engine integration, which remains deferred to Tier-2 / gate G4 (D009, D010). Per D014's consequence clause — "any later switch to a fused backend for a specific measurement must be recorded and must re-verify equivalence" — **this entry is that record**, and token-identity equivalence MUST be re-verified for any compiled/fused path before it produces a *scientific* result (a latency-only characterization does not need re-verification, but must be labeled as such).
- **Alternatives:** Keep a single eager mode (cannot characterize the deployed-regime headroom, leaves M3 permanently confounded); build a full static-KV-cache CUDA-graph decode engine now (that is Tier-2 serving scope, deferred by D009/D010).
- **Consequences:** `_load_one` applies `torch.compile` when `compile_mode` is set; `bench_draft` gains `attn`/`compile_mode` parameters; the default capture path (T4/I10) is unchanged. Any controller net-latency claim (C05/C06) must state the execution mode and, if compiled, cite the equivalence re-verification. Standard operating posture becomes two-mode: **eager for activation capture, compiled for timing.**

## D022 — Exhaustive representative corpus (v2), expanding beyond the D013 four domains

- **Date:** 2026-07-13
- **Context:** D013 fixed an initial 4-domain corpus (HumanEval, GSM8K, MT-Bench, XSum; 644 prompts). A multi-agent dataset survey (27 agents; 127 candidates → 94 verified, license-checked; recorded in `CORPUS_PLAN.local.md`) established that under greedy exact-match, per-token acceptance is a deterministic function of (draft, target, context), so the corpus is the single load-bearing external-validity decision. The current 644 spans only 4 of ~11 production axes and likely yields a near-unimodal per-prompt acceptance histogram — the worst case for demonstrating (and honestly falsifying) adaptive length, routing/skip, and a calibrated acceptance probe.
- **Decision:** Adopt a versioned **Core v2 corpus** (~1,500 prompts, 8 axes): keep HumanEval/GSM8K/MT-Bench; add MBPP (code), OASST1 (chat, to the 150-target), WMT14 de-en (translation), Natural Questions Open (QA/RAG), JSONSchemaBench (structured/high-alpha); replace XSum-as-primary with CNN/DailyMail (Spec-Bench summarization slot), keeping XSum optional. Build into `/artifacts/data_v2/` so the sealed v1 corpus and every existing analysis stay valid. Provenance fields (`spdx`, `row_id`) and a `LICENSES.md` accompany it. Copyright-text datasets (CNN/DM, XSum) ship **row_id only** in any release; all reference completions are generated **in-house** with the Qwen pair (Apache-2.0), sidestepping output-license terms. A `spec_bench_comparable` subset is tagged for the exact Spec-Bench aggregate. Extended/Stretch tiers are added behind a caps flag.
- **Alternatives:** Keep the 4-domain 644 (underpowers the mechanisms, narrow acceptance distribution); mirror Spec-Bench's 80/subtask slices as the whole corpus (too few for ±0.02 acceptance CIs — used only for the comparability number).
- **Consequences:** `cas.data.ingest` gains v2 loaders + `ingest_core_v2`; a new `ingest_v2` Modal function writes the versioned corpus. Any results run on v2 requires re-pinning dataset revisions (several loaders currently stream) and re-verifying the sweep's split-stamp fix (prompt_hash keying). The greedy-vs-sampling external-validity caveat and the corpus-version used are recorded per result in `CLAIMS_LEDGER.md`.

## D023 — Generator-critic autoresearch substrate for I13/I23

- **Date:** 2026-07-19
- **Context:** The pre-round headline (I23/C10) needs a search over candidate
  cached-representation signals, but (a) the load-bearing datum — the TARGET
  model's verified-context frontier representation — is not captured (I10's
  `capture_activations` records only the DRAFT residual stream at
  proposal-generating positions, i.e. the within-round signal), and (b) features
  are hardcoded tuples in `cas.analysis.baselines` with no equal-capacity
  controls. `docs/generator_critic.md` specifies a generator-critic loop
  (eval-gated hill-climbing) as the execution method.
- **Decision:**
  1. Adopt the generator-critic loop as the execution METHOD for I13 (incremental
     information) and I23 (pre-round prediction). It is a method, not new
     scientific scope: locked corpus (D022), tooling (D015/D021), and
     measurement-first ordering (D018) are unchanged.
  2. **Target-frontier capture:** add `capture_frontier_activations` (mirrors
     `capture_activations`; D015 eager / `output_hidden_states` path) recording
     the target residual stream at the last-committed (frontier) position that
     exists BEFORE each round drafts, at `DEFAULT_LAYERS`, labeled by that same
     round's realized acceptance. New artifact family under
     `/artifacts/probes/<run_id>/frontier/`; contract in `cas.autoresearch.types`.
     No trace-schema change (activations live in separate `.npy`; the existing
     `activation_artifact_id` slot may reference them).
  3. **Generator = parameterized seed library, NOT free-form code generation.**
     Candidates are `FeatureSpec(family in {raw,lowrank,drift,align,norm}, layers,
     params)`. No arbitrary code executes inside the loop (reproducibility + safety).
  4. **Frozen bar + controls:** every candidate is scored as incremental lift
     over the frozen `preround_hardened` baseline (~0.73 AUROC) AND must beat
     norm-matched and random controls of equal dimensionality, under
     prompt-grouped `GroupKFold` OOF, with decision-regret reported (D018).
     Selection on dev only; test frozen.
  5. Mechanistic/"circuit" language stays G2-gated (D020): a predictive survivor
     is a "diagnostic signal", never a "circuit", until interventions (I15) pass.
- **Alternatives:** free-form code-gen generator (rejected: irreproducible,
  unsafe); reuse draft-side within-round activations as the pre-round signal
  (rejected: wrong side of the round — post-draft, not available before
  drafting); no controls (rejected: a lift over the zero-feature baseline is not
  evidence).
- **Consequences:** new package `src/cas/autoresearch/` (types, features, eval,
  cost); new `scripts/fit_autoresearch.py`; new Modal `capture_frontier_activations`;
  new `.claude/workflows/generator_critic.js`. I13/I23 -> IN_PROGRESS (Claude).
  Numbers land in `CLAIMS_LEDGER.md` (C10, incremental); killed candidates and any
  negative result are logged. No claim uses "circuit" pre-G2.

## D024 — Autoresearch orchestrator relocated into the tracked package

- **Date:** 2026-07-20
- **Context:** D023 created the generator-critic orchestrator at
  `.claude/workflows/generator_critic.js`. `.claude/` holds Claude Code tooling
  state (`settings.local.json`), not the research artifact; the owner directed
  that the orchestrator live in the tracked repository so the whole autoresearch
  loop (python package + orchestrator) is one committed, reproducible unit.
- **Decision:** Move the orchestrator to
  `src/cas/autoresearch/generator_critic.js`, co-located with the package it
  drives (types/features/eval/cost). It is invoked as a Workflow by `scriptPath`
  (that path), not by registered name — moving it out of `.claude/workflows/`
  removes name-based discovery, an accepted consequence.
  `.claude/settings.local.json` stays (local tooling config). The embedded fit
  command (`PYTHONPATH=src python scripts/fit_autoresearch.py ...`) is repo-root
  relative and unchanged.
- **Alternatives:** keep it in `.claude/workflows/` (rejected: not tracked as an
  artifact, tooling-specific); a new top-level `workflows/` dir (rejected: splits
  the loop across two locations); `scripts/` (reasonable, but package
  co-location is more cohesive).
- **Consequences:** D023's consequences line naming the old path is historical
  (append-only convention). The backlog build-status note and
  `docs/generator_critic.md` §8 reference the new path. The file is untracked at
  the new location until committed.

## D025 — Autoresearch methodology fixes: stratified capture, regularized fit, domain control, hook-based interventions

- **Date:** 2026-07-22
- **Context:** The I13/I23 pre-round autoresearch and the I15 causal validation
  surfaced four methodology choices needing a record (evidence in
  `docs/autoresearch_outcomes.md`, `docs/causal_intervention_report.md`, and the
  2026-07-22 claims-ledger note).
- **Decision:**
  1. **Domain-stratified capture sampling.** `capture_frontier_activations` selects
     prompts round-robin across domains, not sorted-then-truncated. The prior
     `cap_prompts`+sorted truncation captured only ~2 domains/run, inflating the v2
     lift (weak summarization baseline) and making domain-control vacuous. Capture is
     also parameterized by `cas_pair` (qwen/llama) and `data_dir` (v1 `data` / v2
     `data_v2`) with the HF secret, enabling the transfer captures.
  2. **Regularized, reproducible probe fit (`c_reg=0.1`).** Logistic `C=1.0`
     under-regularizes a 14k-feature probe: lbfgs does not converge and the iterate
     is BLAS-thread-dependent, so AUROC wobbles run-to-run. `c_reg=0.1` converges the
     strictly-convex objective to its unique (thread-independent) optimum and is the
     right regularization; it also raised the lift (overfit removed). `c_reg` is a
     threaded parameter; the autoresearch entrypoints default to 0.1.
  3. **Domain-controlled baseline (C04).** The incremental test adds a one-hot
     `domain` block to the frozen baseline (`preround_hardened + domain`), so the
     claim is "beyond entropy AND domain". Required on multi-domain corpora.
  4. **I15 interventions via forward hooks (deviation from D015's nnsight).** The
     runner uses transparent PyTorch forward hooks (auditable, no image re-lock);
     nnsight remains swappable. A `sealed_fidelity` no-op check validates the hook
     site/layer index; disruption is measured in the self-consistent re-forward
     frame; the causal verdict is DISRUPTION-based (the intervention is an
     inverted-U peaking at α=0), not endpoint-monotone.
- **Alternatives:** sorted-truncation capture (undersamples domains); `C=1.0`
  (non-reproducible, overfit); domain-free baseline (C04 violation); nnsight now
  (image re-lock for a pilot); endpoint-monotone verdict (wrong for the inverted-U —
  a false negative). All rejected for the reasons above.
- **Consequences:** `modal_app.py` capture/fit/intervene gain
  `cas_pair`/`data_dir`/`c_reg`/`domain_control`; `_baseline_design` gains
  `include_domain`; new `src/cas/autoresearch/interventions.py` (+ tests) and
  `modal_app.py::intervene`. Numbers land in the 2026-07-22 ledger note; C10 stays
  `UNTESTED` until the frozen predictive test pass; mechanistic language stays
  G2-gated (D020) despite the empirical causal pass.

## D026 — C04 folded into the primary manuscript (no separate companion paper)

- **Date:** 2026-07-23
- **Context:** The I19 manuscript (`paper/main.tex`, commit 3b7905e, 18:18) was
  drafted ~3 hours before the C04 atlas chain landed (80ee48b/968c425/b8eb122,
  21:22–21:29), so C04 had no presence in it. The 2026-07-22 C04 test-pass note
  in the claims ledger provisionally routed the atlas and contrast tables to a
  "companion analysis manuscript".
- **Decision:** C04 goes into the primary manuscript as its own section
  (`\S\ref{sec:atlas}`, "What acceptance is about: the token-category atlas"),
  with a contribution bullet, an abstract clause, and its own scope paragraph in
  Limitations. No separate companion paper is planned. The provisional
  companion-manuscript wording in the ledger is superseded by this entry.
- **Rationale:** (a) The primary paper's only positive result otherwise rests on
  a probe; C04 adds a second frozen-test-passing, pre-registered positive that is
  purely behavioral. (b) C04 carries no G2 language exposure (D020), so it cannot
  be destabilized by the pending human decision on mechanism wording. (c) Two
  thin manuscripts were judged weaker than one, given the atlas alone lacks a
  systems or causal arc.
- **Scope guard:** C04 is presented as complementary evidence, NOT as an
  explanation of the probe's signal. The manuscript makes no claim that the
  category structure is what the frontier representation reads; §Limitations
  states this explicitly. Phase axis stays narrowed out.

## D027 — Plain-English manuscript rewrite; "frontier representation" renamed "frontier state"

- **Date:** 2026-07-23
- **Context:** At the owner's instruction the manuscript was rewritten for
  readability. Target: Flesch Reading Ease >= 70 (measured, not judged), with
  every term defined upfront. Baseline was 27.9 (specialist-journal register).
- **Decision:**
  1. **Terminology.** The core term "frontier representation" is renamed
     **"frontier state"** everywhere in `paper/main.tex`. It denotes the same
     object (phi(t): the target's residual-stream state at the last committed
     position, layers {6,12,18,24} concatenated, 14,336 dims Qwen / 16,384
     Llama). "State" is accurate and shorter, and makes no mechanism claim, so
     it stays clear of the G2 gate (D020). The claims ledger keeps its original
     "frontier representation" wording for the C10 evidence record; the two are
     the same object (cross-reference added to the ledger).
  2. **Register.** Plain professional English, ~11 words/sentence, ~1.46
     syllables/word. Achieved 73.5 whole-document, every section >= 70.
  3. **Definitions upfront.** A new glossary section ("Terms used in this
     paper", `\label{sec:terms}`) defines all 19 terms before any technical
     section uses them, per the owner's specific request.
- **Invariants held (mechanically checked, `scripts/check_invariants.py`):** no
  numeric token dropped from any scientific section (verified per-section: 0
  dropped / 0 added across all 10 original sections; the 17 whole-document
  additions are all glossary definitions); scope vocabulary did not shrink
  (119 -> 160); all labels, refs, figures and bibitems preserved; no affirmative
  "mechanism"/"circuit" usage. The prior draft is recoverable from git and from
  the pre-rewrite backup.
- **Tooling:** `scripts/readability.py` (Flesch scorer) and
  `scripts/check_invariants.py` (number/hedge/structure/vocabulary gate) are the
  durable gates, wired as `make -C paper check`. Both report via `textstat`
  plus an independent implementation so the score does not rest on one library.
- **Consequence:** The manuscript grew 14 -> 17 pages (plain English uses more
  words). Science unchanged.
