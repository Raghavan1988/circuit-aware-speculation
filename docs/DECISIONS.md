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
