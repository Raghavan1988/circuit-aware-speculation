# Execution Plan (Claude)

Written 2026-07-10; revised the same day after decisions D008–D011. This is the
execution calendar for the repository, subordinate to the research contract
(`docs/RESEARCH_SPEC.md`, `docs/EXPERIMENT_CONTRACT.md`, `docs/ISSUE_BACKLOG.md`).
Where this plan and the contract differ, the contract wins unless a dated entry in
`docs/DECISIONS.md` says otherwise. Per D008, no venue or external organization is
named in this file; submission windows appear as dates only.

---

## 0. Phase structure

The work splits into two phases with a hard boundary:

- **Phase 1 — custom-harness science (Steps 1–9).** Everything the paper's
  claims rest on: exact decoding, baselines, probes, interventions, the
  controller, replication. Runs entirely in our own `cas` harness on Modal/H100.
  Batch-1, unloaded. **No serving engine.** This is what is being built now.
- **Phase 2 — serving-engine integration + load validation (Step 10 / I24 /
  gate G4).** All SGLang work lives here: implementing per-request draft-length +
  skip control in SGLang (the unserved upstream gap), then demonstrating TTFT/ITL
  impact **under concurrent load**. Gated by D010 (artifact stage) and only
  started after the Phase-1 evidence gate. The engine choice (SGLang lean) and
  the upstream-contribution map are Phase-2 concerns; §3 and §5 note them but
  they are not touched during Phase 1.

Nothing in Phase 1 imports or depends on SGLang. Phase 2 is a separate track.

## 1. Where this plan comes from

1. An earlier plan for the reference project at
   `/home/raghavan/13_Raghavan_Content_Aware_Speculation_Control` (read-only): a
   content-aware controller that jointly picks the draft model and the speculation
   length per request, evaluated against the best fixed (draft, length) policy.
2. A competitive-landscape scan (Section 3) showing the bandit-controller space
   has become crowded in the last ~8 months, which supports this repository's
   mechanistic thesis (D002).
3. A cross-review between the two planning agents; the resulting standards were
   ratified in D009 (gates, tracks) and D010 (staged release).

## 2. Compute and environment facts (verified 2026-07-10)

- Local machine: RTX 4090 **Laptop**, 16 GB VRAM. Base conda env has **CPU-only
  torch 2.0.1** and transformers 4.29 — do not use the base env for experiments.
- Cloud: user will provision **Modal or RunPod** (`~/.modal.toml` exists locally).
  Plan assumes A100/H100-class GPUs per the backlog's compute column.
- Hugging Face: cached token is **invalid** (401 on gated repos). The primary
  Qwen2.5 pair is ungated and unblocked. The Llama replication pair is gated and
  **blocked until the user refreshes the token** (`huggingface-cli login`) — this
  gates issue I17.
- Rough budget (per D013: Modal, H100 80GB at ~US$3–5/hr): 40–60 GPU-hours for
  the full protocol ≈ **US$150–300**. Trace/activation storage: plan tens of GB.

## 3. Competitive landscape (scanned 2026-07-10; verify via I21 before freezing claims)

Closest published/preprint neighbors. None provides a mechanistic account of
acceptance; that is the differentiation (see `RESEARCH_SPEC.md`,
"Differentiation requirements"). Cite all; compare where marked.

| Work | What it adapts | Relevance here |
|---|---|---|
| SpecDec++ (arXiv:2405.19715) | Length via a **trained acceptance head on draft hidden states** | Closest learned-signal baseline; contract policy 8. Reproduce early (I22). |
| BanditSpec (arXiv:2505.15141) | Length/config as bandit; stopping-time regret | Contract policy 7 (I09). Must compare. |
| SVIP (arXiv:2411.18462) | Entropy stop rule | Contract policy 5 (I08). Must compare. |
| TapOut (arXiv:2511.02017) | Bandit over length *strategies* | Cite; optional comparison. |
| AdaSD (arXiv:2512.11280), TALON (arXiv:2601.07353) | Adaptive lengths / token trees | Cite. |
| Talon (2025; name collision with TALON above) | Heterogeneous drafting with asynchronous execution | Verify primary source via I21; cite. |
| OnlineSpec "When Drafts Evolve" (arXiv:2603.12617) | Draft *weights* via online learning | Different axis; cite. |
| Learning to Draft (arXiv:2603.01639) | Throughput-optimized drafting via RL | Verify (I21); cite. |
| Multi-drafter alignment feedback (arXiv:2604.05417) | Drafter pool | Deferred-scope relative; cite. |
| Acceptance dynamics across domains (arXiv:2604.14682) | Task-conditioned acceptance behavior | Verify (I21); may reposition claim C04 as control. |
| SpecKV (arXiv:2605.02888) | Draft confidence under KV compression | Verify (I21); cite. |
| Theory of acceptance (arXiv:2606.30265) | Theoretical treatment of acceptance | Verify against primary archive (found via mirror); constrains Track B option "theory". |
| SGLang adaptive speculation (docs, 2026) | Runtime `num_steps` via EMA of accepted length; server-level tiers with pre-captured CUDA graphs | Engines now ship adaptive length (EAGLE-only, no per-request control, no custom policies); cite; strengthens the mechanistic differentiation. |
| DSpark (arXiv:2607.05147, 2026-07-06; authors' production stack, **not** SGLang — corrected 2026-07-11 per I21 R2) | Per-request **verify** budgets from a calibrated trained linear confidence head; live-traffic deployment | Production-deployed adjacent method — cite. Trims verification after drafting; does not avoid draft compute and needs a trained head — contrast with the pre-round bet (I23), which decides before drafting from cached states. Also publishes probe-quality metrics (AUROC 0.81–0.90, ECE ≈1%) — see 2026-07-11 ledger note on C01 reframing. |
| Not-a-Bandit (arXiv:2510.20064; earlier cited ID 2506.00285 was wrong — corrected 2026-07-11) | No-regret full-information drafter selection | Deferred-scope relative; cite. Full-information label trick reused for counterfactual training labels (D018.3). |
| MetaSD (2024) | Per-step drafter UCB | Deferred-scope relative; cite. |
| Task detection + heterogeneous drafting (arXiv:2505.08600) | Task-based routing | Closest to the deferred routing idea; cite. |

Implication: **speed matters.** Several groups are active on adjacent questions;
the staged-release policy (D010) exists so priority is never hostage to polish.

I21 updates: (2026-07-10, Grok) all four planning-pass queue items verified on
primary arXiv; the mirror-found item was 2606.30265; C04 not pre-empted by
2604.14682 (domain grain only). (2026-07-11) Not-a-Bandit ID and DSpark
attribution corrected in place above; `docs/landscape.md` is authoritative for
comparison detail and now includes the 2026-07-11 sweep additions (AdaEAGLE,
Judge Decoding, WhiFlash, C2T, Sequoia, DISCO, AdaEDL, DSDE, TurboSpec) and a
deployed-practice section.

## 4. Execution schedule (cloud compute)

Maps onto `docs/ISSUE_BACKLOG.md` IDs. Days are working days; overnight GPU runs
expected. Timeline interpretation per D009: ~7 days core + preliminary evidence;
10–14 days first complete scientific draft; further weeks for Track C release
engineering.

| Day | Backlog issues | Deliverable / gate |
|---|---|---|
| 1 | I01, I05, I21 | Pinned cloud env; dataset ingestion + frozen prompt-grouped split manifests; landscape additions verified, living comparison table started. |
| 2 | I02, I03, I04 | Exact greedy target–draft decoding (actions `skip,1,2,3,4,6,8`), KV-cache correctness; equivalence tests pass (token-identical to target-only); synchronized timing. |
| 3 | I06, I07 | Trace writer conforming to `docs/TRACE_SCHEMA.md`; **overnight**: target-only + skip + full fixed-length sweep on all four workloads. |
| 4 | I08, I09, launch I10+I11 | Entropy, acceptance-history, and BanditSpec-style policies in-loop; activation capture and token-category annotation started. |
| 5 | I12, I13, I22 | Leakage-safe layerwise probes; calibration + incremental-information tests; learned-head baseline reproduced. **Evaluate against gate G1** (`RESEARCH_SPEC.md`). |
| 6 | I14, I23, launch I15–I17 | Compute-optimal selective controller frozen on dev data; pre-round cached-representation prediction (headline candidate, C10); interventions, shift runs, replication launched as compute permits. |
| 7 | I18, start I19 | Acceptance atlas + all primary figures generated from artifacts; manuscript assembly begins. |
| 8–14 | I15–I17 complete, I19, I20 | Replicated interventions (G2), shift studies, replication pair; manuscript; clean-reproduction audit (G5 precondition). |
| Weeks 3+ | I24 | Staged release package per D010: preprint at G1–G3+audit; benchmark/recipes/integration at G4. |

## 5. Decision gates

Superseded by D009: the gates are specified in `docs/RESEARCH_SPEC.md`
("Decision gates", G1–G5) and are not duplicated here. Operational notes:

- The Day-5 checkpoint evaluates G1. A stable, replicated negative result is a
  valid G1 outcome and proceeds as a diagnostic contribution (contract's own
  rule); routing is future work, not a fallback identity.
- "Circuit"/"mechanism" language is G2-gated; before G2 passes, write
  "representation" or "diagnostic signal" everywhere, including commit messages.
- Track structure (D009): Track A = the contract; Track B = one headline
  differentiator — primary bet: pre-round prediction from cached
  verified-context representations (I23, C10), causal analysis underneath (I15,
  C03), cross-speculator transfer as extension (C11); Track C = release
  artifacts (I24), staged per D010.
- Alternative Track B bets, preserved from the planning review, if the primary
  bet fails G1/G2: (a) mechanism-derived theory connecting an internal quantity
  to a bound or estimator of acceptance probability — first verify that
  arXiv:2606.30265 does not already occupy this; (b) a robustness mechanism
  explaining and mitigating acceptance collapse under domain shift, compression,
  or target–draft mismatch. Pick one, never several (D009).
- (Phase 2) Serving-engine choice for the G4 adapter is deferred until G4. Current lean:
  **SGLang** — its 2026 adaptive-speculation tiers provide a plausible
  integration surface for a per-request controller, and an open upstream feature
  request for dynamic step control shows appetite. *(Correction 2026-07-11, I21
  R2: DSpark itself is NOT an SGLang feature — arXiv:2607.05147 deploys in the
  authors' own production stack; the earlier "DSpark's ragged per-request verify
  in SGLang" rationale was wrong. The SGLang-side DSpark tracking issue below is
  an integration request, not shipped code. The engine lean is therefore weaker
  than first recorded — re-verify the whole surface at G4.)* The core science
  (Steps 1–9) stays in the custom harness regardless: engines expose no
  per-round policy hooks, no signal APIs, and no activation access under CUDA
  graphs. Re-verify both engines at G4.
- (Phase 2) Upstream contribution map (verified 2026-07-10; re-verify at G4): per-request
  draft-length control remains unserved — upstream issue #21459 requested it and
  was closed inactive; the Q2-2026 spec roadmap (#23005) names per-request
  adaptive configuration as a goal but shipped only batch-size-level adaptation
  (#23705); the DSpark tracking issue (#30344) is open and unassigned, listing
  an adaptive cost model and acceptance-observability metrics as follow-ups that
  overlap I14 and our profiler. The `STANDALONE` algorithm covers independent
  small-LLM drafts, matching the contract's model setup. Sequencing per D010:
  one small early credibility contribution (e.g., docs issue #18268 or metrics)
  is acceptable; the controller PR waits for the validated signal (G4).

## 6. Submission-window calendar (dates only, per D008)

- **Rolling-review journal** (primary target): no deadline; submit when the
  evidence gate in `docs/RESEARCH_SPEC.md` passes. Reviews take ~2 months.
- **Public preprint**: at G1–G3 + clean-reproduction audit, per D010; immediately
  if directly competing work appears (escape hatch).
- Rolling-review cycle windows: **Aug 3, 2026** (commitment Oct 11) — likely
  forfeited by the honest timeline; **Oct 12, 2026** — backup.
- Non-archival community venue for an early core-result preview: ~**Aug 29, 2026**.
- Major conference window: abstract **Sept 19**, paper **Sept 24, 2026** — natural
  target for the extended version (deferred scope: routing, larger batch, modern
  speculator integration).
- Systems-venue window: **Oct 30, 2026** — only if a serving-engine version is built.

Do not submit to two archival venues concurrently; sequence journal vs. cycle
submissions.

## 7. Deltas between Claude's earlier plan and this repository's contract

Recorded so the owner can adjudicate with both agents' views on the table:

| Topic | Claude's earlier plan | Repo contract | Position after D009 |
|---|---|---|---|
| Thesis | Joint (draft, length) controller | Mechanistic account + compute-optimal control | **Contract** — better differentiated (§3). |
| Draft routing | Core (Phase 2) | Deferred | Deferred; extension axis only. |
| EAGLE-3 / modern speculators | Nice-to-have | Deferred | Deferred from core; extension (C11) and Track C integration target. |
| Batch sweep | Restored via cloud (1/8/32/64) | Batch 1 primary, batch 8 optional | Contract stands for the journal target; sweep belongs to the extended version. |
| Sampling T>0 | High value add | Deferred as primary result | Acceptable; note as limitation. |
| Skip action | Absent | Included | **Contract** — component, not headline (differentiation rules). |
| Decision gate | Three-branch Day-5 gate | Principle stated in spec | **Superseded** by G1–G5 (D009). |
| Baselines | Fixed + SVIP + UCB + drafter-UCB | Ten policies incl. learned head + oracle | Contract, with policy 8 sharpened to the SpecDec++-style head (I22). |

## 8. Reusable assets from the reference project (read-only)

From `/home/raghavan/13_Raghavan_Content_Aware_Speculation_Control` — ideas and
starting points only; nothing there is evidence, and code must be re-owned here:

- `src/controllers.py` — entropy-threshold / epsilon-greedy / UCB implementations
  (skeleton for contract policies 5–7 after adaptation to real rewards).
- `src/metrics.py` — percentile/TTFT/TPOT helpers (pure python, tested).
- `paper/paper.tex` — background/related-work prose may be mined; every results
  sentence and figure there is placeholder and must not be copied.

## 9. Risk register (preserved from the planning review)

| Risk | Mitigation (where encoded) |
|---|---|
| Novelty collapse — the chosen internal signal turns out already covered | Living comparison table + reference verification (I21); reproduce the closest baseline early (I22) |
| Probe leakage — adjacent tokens inflate apparent generalization | Prompt-grouped splits (contract, schema invariant); prompt-level bootstrap |
| Pseudo-mechanistic claims — a predictive direction merely encodes confidence | Incremental controls (G1); specificity tests and interventions (G2) |
| Overhead inversion — features improve prediction but slow inference | Offline capture vs. deployed-signal separation (spec Controller; policy 9); G3 |
| Custom-harness skepticism — gains may not survive production scheduling | Batch-8 validation (contract); optional engine integration (G4, I24); state as limitation |
| Scope explosion — routing, batching, sampling, robustness all at once | One headline bet, one deployment test, one replication axis (D009) |
| Premature public claims | Staged release (D010); the claims ledger governs every release |
