# Execution Plan (Claude)

Written 2026-07-10. This is Claude's execution plan for the repository, reconciled
with the existing research contract (`docs/RESEARCH_SPEC.md`, `docs/EXPERIMENT_CONTRACT.md`,
`docs/ISSUE_BACKLOG.md`). Where this plan and the contract differ, the contract wins
unless a dated entry in `docs/DECISIONS.md` says otherwise.

---

## 1. Where this plan comes from

This plan was developed in two stages:

1. An earlier plan for the reference project at
   `/home/raghavan/13_Raghavan_Content_Aware_Speculation_Control` (read-only): a
   content-aware controller that jointly picks the draft model and the speculation
   length per request, evaluated against the best fixed (draft, length) policy.
2. A competitive-landscape scan (Section 3) that showed the bandit-controller space
   has become crowded in the last ~8 months, which supports this repository's pivot
   to the mechanistic ("circuit-aware") thesis.

**Claude's opinion on the pivot:** the mechanistic reframing is the right call. The
joint (draft, length) controller is still an open gap, but it is a shrinking one, and
a controller-only paper now needs to beat several 2025–26 methods to be credible. The
mechanistic questions (where acceptance information lives in the draft model, whether
interventions confirm it, whether the cheapest validated signal suffices) are much
less contested, and the negative result is publishable under this repo's own rules.
The original joint-routing idea survives as deferred scope and as a fallback paper.

## 2. Compute and environment facts (verified 2026-07-10)

- Local machine: RTX 4090 **Laptop**, 16 GB VRAM. Base conda env has **CPU-only
  torch 2.0.1** and transformers 4.29 — do not use the base env for experiments.
- Cloud: user will provision **Modal or RunPod** (`~/.modal.toml` exists locally).
  Plan assumes A100/H100-class GPUs per the backlog's compute column.
- Hugging Face: cached token is **invalid** (401 on gated repos). The primary
  Qwen2.5 pair is ungated and unblocked. The Llama replication pair is gated and
  **blocked until the user refreshes the token** (`huggingface-cli login`).
- Rough budget estimate at ~US$2–3/hr for an 80 GB A100: 60–100 GPU-hours for the
  full protocol ≈ **US$150–300**. Trace/activation storage: plan tens of GB.

## 3. Competitive landscape (scanned 2026-07-10)

Closest published/preprint neighbors. None of these provides a mechanistic account
of acceptance; that is the differentiation. Cite and, where marked, compare.

| Work | What it adapts | Relevance here |
|---|---|---|
| SpecDec++ (arXiv:2405.19715) | Length via a **trained acceptance head on draft hidden states** | Closest learned-signal baseline; covered by the contract's "learned output-confidence predictor" (policy 8). Must compare. |
| BanditSpec (arXiv:2505.15141) | Length/config as bandit; stopping-time regret | Contract policy 7. Must compare. |
| SVIP (arXiv:2411.18462) | Entropy stop rule | Contract policy 5. Must compare. |
| TapOut (arXiv:2511.02017) | Bandit over length *strategies* | Cite; optional comparison. |
| AdaSD (arXiv:2512.11280), TALON (arXiv:2601.07353) | Adaptive lengths / token trees | Cite. |
| OnlineSpec "When Drafts Evolve" (arXiv:2603.12617) | Draft *weights* via online learning | Different axis; cite. |
| Multi-drafter alignment feedback (arXiv:2604.05417) | Drafter pool | Deferred-scope relative; cite. |
| Not-a-Bandit (arXiv:2506.00285) | No-regret drafter selection | Deferred-scope relative; cite. |
| MetaSD (2024 workshop paper) | Per-step drafter UCB | Deferred-scope relative; cite. |
| Task detection + heterogeneous drafting (arXiv:2505.08600) | Task-based routing | Closest to the deferred routing idea; cite. |

Implication of the scan: **speed matters**. Several groups are active on adjacent
questions; a public preprint should go up as soon as the evidence gate allows.

## 4. Seven-day execution schedule (cloud compute)

Maps directly onto `docs/ISSUE_BACKLOG.md` IDs. Days are working days; overnight
GPU runs are expected. This is the same shape as the README's seven-day draft
milestone, made concrete.

| Day | Backlog issues | Deliverable / gate |
|---|---|---|
| 1 | I01, I05 | Pinned cloud env (versions recorded); dataset ingestion + frozen prompt-grouped split manifests. |
| 2 | I02, I03, I04 | Exact greedy target–draft decoding (actions `skip,1,2,3,4,6,8`), KV-cache correctness; equivalence tests pass (token-identical to target-only); synchronized timing (prefill/draft/verify/controller measured separately). |
| 3 | I06, I07 | Trace writer conforming to `docs/TRACE_SCHEMA.md`; **overnight**: target-only + skip + full fixed-length sweep on all four workloads. |
| 4 | I08, I09, launch I10+I11 | Entropy, acceptance-history, and BanditSpec-style policies running in-loop; activation capture and token-category annotation started. |
| 5 | I12, I13 | Leakage-safe layerwise probes; calibration + incremental-information tests. **Decision gate (see §5).** |
| 6 | I14, launch I15–I17 | Compute-optimal selective controller frozen on dev data; interventions, shift runs, and replication pair launched as compute permits. |
| 7 | I18, start I19 | Acceptance atlas + all primary figures generated from artifacts; manuscript assembly begins. |

Honest scheduling note: I19 (manuscript) and I20 (clean reproduction + evidence
audit) realistically land on days 8–10, and the full evidence gate (replicated
interventions, shifts, replication pair) may take 10–14 days total. The seven-day
milestone produces a complete draft plus the core result, not the audited final.

## 5. Day-5 decision gate

After I13, exactly one of these holds; each has a pre-agreed response:

1. **Internal signals beat entropy/margin/history with affordable overhead** →
   proceed with the circuit-aware controller as the headline (contract unchanged).
2. **Internal signals predict better but overhead erases latency gains** → headline
   becomes the mechanism–systems trade-off; controller ships the cheapest
   sufficient signal. Record in `docs/DECISIONS.md`.
3. **Internal signals add nothing beyond entropy** → report the negative finding
   (required by the spec); the compute-optimal controller with skip action on cheap
   signals remains the systems contribution. Record in `docs/DECISIONS.md`, and
   consider reviving the deferred joint-routing axis as the follow-up.

## 6. Venue timeline (names withheld by request; dates verified 2026-07-10)

- **Rolling-review journal** (this repo's stated target): no deadline; submit when
  the evidence gate in `docs/RESEARCH_SPEC.md` is met. Reviews take ~2 months.
- **Preprint server**: immediately at gate — priority protection given §3.
- NLP rolling-review cycle: submit **Aug 3, 2026** (commitment Oct 11, 2026).
- Efficiency-workshop contributions (non-archival hedge): ~**Aug 29, 2026**.
- Major ML conference: abstract **Sept 19**, paper **Sept 24, 2026** — the natural
  target for an extended version with the deferred scope (routing, larger batches).
- ML-systems conference: **Oct 30, 2026** — only if a serving-engine version is built.
- (A large general-AI conference has abstract July 21 / paper July 28, 2026 —
  judged too tight to meet the evidence gate honestly. Not recommended.)

Do not submit to two archival venues concurrently; sequence journal vs. cycle
submissions.

## 7. Deltas between Claude's earlier plan and this repository's contract

Recorded so the user can adjudicate with both agents' views on the table:

| Topic | Claude's earlier plan | Repo contract | Claude's current position |
|---|---|---|---|
| Thesis | Joint (draft, length) controller | Mechanistic account + compute-optimal control | **Contract** — better differentiated (§3). |
| Draft routing | Core (Phase 2) | Deferred | Accept deferral; it is the extended-version axis. |
| EAGLE-3 | Nice-to-have | Deferred | Agree. |
| Batch sweep | Restored via cloud (1/8/32/64) | Batch 1 primary, batch 8 optional | Contract is fine for the journal target; full sweep belongs to the extended version. |
| Sampling T>0 | High value add | Deferred as primary result | Acceptable; note as limitation. |
| Skip action | Absent | Included | **Contract** — genuinely good addition; oracle data will show when drafting has negative payoff. |
| Baselines | Fixed + SVIP + UCB + drafter-UCB | Ten policies incl. learned predictor + oracle | Contract is stronger. |

## 8. Reusable assets from the reference project (read-only)

From `/home/raghavan/13_Raghavan_Content_Aware_Speculation_Control` — ideas and
starting points only; nothing there is evidence, and code must be re-owned here:

- `src/controllers.py` — entropy-threshold / epsilon-greedy / UCB implementations
  (usable as the skeleton for contract policies 5–7 after adaptation to real rewards).
- `src/metrics.py` — percentile/TTFT/TPOT helpers (pure python, tested).
- `paper/paper.tex` — background/related-work prose may be mined; every results
  sentence and figure there is placeholder and must not be copied.
