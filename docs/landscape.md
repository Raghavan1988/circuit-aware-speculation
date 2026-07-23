# Living Comparison Table (issue I21)

Adjacent work on adaptive / diagnostic-signal / systems speculative decoding.
Maintained so novelty claims stay honest; re-scan before freezing claims or
submitting. **Authoritative for comparison detail**; propose short notes for
`PLAN.md` §3 rather than large rewrites of that file.

`Verified` = checked against a primary source (arXiv abs page and/or HTML);
`search-only` = surfaced via search and awaiting primary confirmation.

**Differentiation this repo asserts** (see `docs/RESEARCH_SPEC.md`): none of
these provides a *causally validated, transferable internal* account of
acceptance, nor a pre-round (pre-draft-compute) signal from cached verified
context under a lossless, calibrated, baseline-controlled independent-drafter
cell. That is the gap.

**Last full verification pass:** 2026-07-11 (Grok, I21 round 2). Corrections
and sweep additions below; 2026-07-10 planning-pass four remain verified.

## Comparison table

| Work | Adapts | Signal | Mechanistic? | Verified? | Relation / contrast here | Baseline / cite |
|---|---|---|---|---|---|---|
| SpecDec++ (arXiv:2405.19715) | draft length | trained acceptance head on **post-draft** hidden states | no (learned head) | verified | closest learned-signal analog; pays draft forward first | baseline (policy 8, I22) |
| BanditSpec (arXiv:2505.15141) | length/config | bandit, stopping-time regret | no | verified | length-only online control | baseline (policy 7, I09) |
| SVIP (arXiv:2411.18462) | length | draft entropy stop rule | no | verified | length-only, cheap signal | baseline (policy 5, I08) |
| MetaSD (ENLSP 2024) | drafter choice | per-step UCB | no | verified | drafter-only; no length | cite (deferred routing) |
| **Not-a-Bandit (arXiv:2510.20064)** | drafter | no-regret selection across drafters | no | **verified (ID corrected)** | drafter-only; **primary ID is 2510.20064** — 2506.00285 is an unrelated POMDP robotics paper | cite (deferred routing) |
| TapOut (arXiv:2511.02017) | length strategy | bandit over strategies | no | verified | meta-level length control | cite; optional compare |
| AdaSD (arXiv:2512.11280) | length | adaptive length | no | search-only | length-only | cite |
| TALON (arXiv:2601.07353) | token trees | adaptive trees | no | search-only | tree-length; **≠** Talon (heterogeneous async) | cite |
| Talon (heterogeneous + async drafting; 2025) | drafter/system | heterogeneous + async | no | search-only | name collision with TALON above; verify primary ID | cite |
| OnlineSpec / "When Drafts Evolve" (arXiv:2603.12617) | draft weights | online learning | no | search-only | different axis (training) | cite |
| **Learning to Draft / LTD** (arXiv:2603.01639) | draft+verify budgets | RL co-adaptive policies; throughput of draft–verify cycle | no | **verified** | throughput-optimized coordination; no internal acceptance localization | cite (I21 ✓) |
| Multi-drafter alignment feedback (arXiv:2604.05417) | drafter pool | alignment feedback | no | search-only | deferred-routing relative | cite |
| **Acceptance dynamics across domains** (arXiv:2604.14682) | — (analysis) | task/domain-conditioned acceptance; weak entropy–α correlation; early/late position bins | no (empirical atlas at **domain** grain) | **verified** | **C04 context, not pre-emption** — domain-level + coarse position bins only; no overlapping token-category labels, no internal representations | cite as domain control; see C04 ledger + §C04 positioning |
| **SpecKV** (arXiv:2605.02888) | length γ | draft confidence + entropy under KV compression (MLP) | no | **verified** | length under compression; post-draft signals; ~0.34 ms overhead | cite (I21 ✓) |
| **Theory of acceptance** (arXiv:2606.30265) | — (theory) | KL/margin certificates for greedy, relaxed, top-m, entropy-thresholded, tree acceptance | theory only (not internal) | **verified (primary arXiv; full text 2026-07-22)** | was found via mirror; **primary archive confirmed**. Constrains Track B pure-theory bet; complementary, not competitive with localization. **Constructive for C10 (2026-07-22):** margin γ_p = target top-1/top-2 gap; certificates for strict greedy (e.g., KL(p‖q)≤ε ∧ γ_p>g⁻¹(ε) ⇒ certified acceptance) **formalize C10's free margin baseline as carrying the governing variable** — the baseline control is mandated by theory, not optional. No learned probe, no calibration, no controller in the paper (post-draft certificates need the drafter distribution). Single-author v1, not yet peer-reviewed | cite; Track B theory option narrowed; grounds C10 baseline design |
| Task detection + heterogeneous drafting (arXiv:2505.08600) | routing | task classifier | no | search-only | closest to deferred routing | cite |
| FASER (arXiv:2604.20503) | per-request draft length in continuous batch; early prune in verify | systems phase management | no | search-only (rescanned) | serving-side fine-grained phase control; not representation-level | cite |
| PARD-2 (arXiv:2605.08632) | dual-mode drafting | confidence-adaptive tokens | no | search-only (rescanned) | draft-structure + confidence; post-draft | cite |
| SpecBranch (arXiv:2506.01979) | hybrid draft length | draft confidence + target features | no | search-only | hybrid adaptive; still post-draft / feature reuse | cite |
| FlashSpec (search hit 2026-06) | draft selection | online bandit | no | search-only | bandit draft selection | cite if primary confirmed |
| Decentralized SD / impact tokens (arXiv:2511.11733) | verification strictness | high- vs low-impact tokens (punctuation/fillers relaxed) | no | search-only | token-role heuristics for **relaxed verify**, not acceptance atlas of exact SD | cite as adjacent to C04 granularity |
| CSD frequency-guided candidates (arXiv:2604.13634) | candidate rescue | frequency / semantic gating | no | search-only | post-hoc analysis of rescued tokens by type — descriptive, not mechanistic | cite |
| SGLang adaptive speculation (docs) | num_steps (server) | EMA of accepted length; tier switch | no | verified | engine-level length tiers; EAGLE-only | cite; Phase-2 integration target; see §Deployed practice |
| **DSpark (arXiv:2607.05147)** | per-request verify length | calibrated linear confidence head on **draft** hidden states + engine throughput profiles | no (learned head) | **verified (system attribution corrected; timing re-verified 2026-07-22)** | C01 **instrument** published; trims verify *after* drafting. Paper deploys in the **authors' production serving stack under live traffic** (abstract: vs MTP-1 baseline) — **not SGLang**. Prior landscape row misattributed venue/system. **2026-07-22 re-read:** within-round timing confirmed — confidence head emits per-position prefix-survival scalars DURING drafting; Algorithm-1 l_r=0 is a post-draft discard, **not a pre-round skip**; its k=1 prefix survival is a first-token quantity but computed on the drafted block. Head-input wording differs between our two full-text reads (hidden-states vs confidence features) — re-quote from the PDF before any manuscript sentence quotes DSpark's head input; not decisive for C10 either way | cite; C01 must be incremental-info framing (I13), not "first head" |
| **AdaEAGLE (arXiv:2412.18910)** | draft length (EAGLE) | Lightweight Draft Length Predictor (LDLP); context-aware adaptive draft structure | no | **verified (full v1 text, 2026-07-22 re-read)** | **Closest structural C10 prior — 3/8 cell criteria.** LDLP input verbatim (§3.2): embedding + target-model **final-layernorm hidden state of the last validated-prefix token** — exactly C10's signal class and timing. But: predicts **LENGTH** (uncalibrated scalar L1 regression; length-classification only a losing ablation §4.3.2), deliberately logit-free ("entropy" once, hypothetical; "margin" zero times; no entropy/margin baselines), **no skip/zero-draft action**, EAGLE trained-head (non-independent) drafter. v1 only (25 Dec 2024), no later versions | cite; C10 counterexample (narrows, does not scoop); primary differentiation target |
| **Judge Decoding (arXiv:2501.19309)** | verification criterion | linear head on **target** embeddings judging draft acceptability | no | **verified** | **During-verify** judgment; **relaxes losslessness** (accepts non-aligned but "valid" continuations). Mandatory C10 contrast: different timing (not pre-round) and different correctness contract | cite; C10 counterexample |
| **WhiFlash (arXiv:2606.07710)** | drafter paradigm (AR vs diffusion) | token-level entropy or learned neural policy on target-side signals | no | **verified** | Pre-draft **drafter routing** (cross-paradigm), not acceptance prediction or skip; deferred-routing-adjacent | cite; C10 counterexample (routing, not accept/reject) |
| **C2T (arXiv:2502.13652)** | tree construction | lightweight classifier over draft-token features (beyond joint probability) | no | **verified** | Tree acceptance predictor for pruning candidates; post-draft / during tree build | cite; cheap-baseline frontier |
| **Sequoia (arXiv:2402.12374)** | tree structure + hardware budget | dynamic programming over token trees; hardware-aware size/depth | no | **verified** | Tree topology optimizer; not internal acceptance localization | cite |
| **DISCO (arXiv:2405.04304)** | speculation lookahead (SL) | dynamic SL selection (training-free / cheap) | no | **verified** | Cheap dynamic-length baseline; ~10% over best static SL, exact same text | baseline-adjacent (I08/I14 compare) |
| **AdaEDL (arXiv:2410.18351)** | draft stop (early exit) | entropy-based lower bound on token acceptance probability | no | **verified** | Training-free early draft stop; SVIP-class cheap signal | baseline-adjacent (I08) |
| **DSDE (arXiv:2509.01083)** | speculation length | post-hoc KLD variance / regional stability + adaptive length cap | no | **verified** | Training-free post-hoc diagnostic length control for serving | cite; systems cheap baseline |
| **TurboSpec (arXiv:2406.14066)** | intra-request parallelism amount | closed-loop goodput feedback; profiles environment | no | **verified** | Serving-system speculation control (implemented on vLLM in paper); load/acceptance-aware K | cite; systems baseline; see §Deployed practice |
| **CaDDTree (arXiv:2606.01813)** | diffusion draft-tree budget | cost-aware throughput objective; unimodal budget search | no | **verified** | Cost-aware tree budget (Codex anchor, now primary-verified); not internal probes | cite |
| **SemanticSpec (arXiv:2602.03708)** | sequence-level accept (semantic) | MLP probes on **draft and target** multi-layer hidden states → semantic probability during verify | no (semantic-relaxed SD) | **verified (full text)** | **C01/C10 threat check:** probes internals for **semantic-sequence acceptance**, not exact token acceptance; **during verify of draft sequences**, not pre-round from cached verified context; **not lossless**. No incremental-info table vs entropy/margin/history/domain. See ledger G2 note | cite as adjacent; does **not** scoop C01 controlled study or C10 pre-round cell |
| **Hybrid Verified Decoding / HVD (arXiv:2606.01019)** | draft SOURCE (cache draft vs model drafter) | accepted-LENGTH regression of a retrieved cache draft from runtime cache-match/history/token-structure features (uncalibrated MLP) | no | **verified (full text, 2026-07-22)** | Pre-verification payoff estimate, but: requires the retrieved cache-draft content to exist (not draft-free), regresses length (not first-token accept), routes between two always-speculate arms (**no skip**), and uses **zero** hidden states/embeddings/logits (grep: 0 hits). Lossless standard verification | cite; C10-adjacent (routing), does not occupy the cell |
| **SpecLA (arXiv:2607.16673)** | draft-tree candidate set (linear-attention targets) | confidence pruning of already-drafted tree paths via drafter path log-probs | no | **verified (full text, 2026-07-22; v1 18 Jul 2026 — newest in window)** | Within-round pruning only ("changes only the submitted candidate set"); target-aligned EAGLE-style drafter (not independent); no pre-round prediction, no calibration, no skip. Core contribution (stateful linear-attention verification runtime) orthogonal to C10's transformer cached-frontier setting. Very recent v1 — may change | cite |
| **Speculative Verification (arXiv:2509.24328)** | verification length | third **companion model** (draft-sized) compares draft-vs-companion distributions on already-drafted tokens | no | **verified (v2 full text, 2026-07-22)** | Within-round; extra model compute (latency-hidden via overlapped processes, but not a free signal); no skip (always drafts, always verifies ≥1); target hidden states not inputs | cite |
| **AdaSpec (arXiv:2503.05096)** | draft continue/stop (mid-round) | product of DRAFT-token confidences (Eq. 7) + batch/context/system profiling | no | **verified (two full-text passes, 2026-07-22)** | Predict-execute-correct loop DURING drafting; zero target-hidden-state/probe inputs. v1 (2025-03-07) named "SpecServe", renamed v2 (2026-01-12). **Name collision:** unrelated "AdaSPEC" (arXiv:2510.19779, distillation) — cite 2503.05096. NOTE: the claim that AdaSpec itself articulates the pre-round-prediction gap was REFUTED 0-3 — do not cite it as naming C10's gap | cite |
| Question-only correctness probes (arXiv:2509.10625) | — (analysis) | linear probes on pre-generation activations → answer correctness | methodology | **verified** | Pre-generation probe methodology (not SD acceptance) | method context for I12/I23 |
| Code pre-generation correctness (arXiv:2606.14530) | — (analysis) | prompt-final hidden-state probe of code correctness; residualization controls | methodology | **verified** | Pre-generation probe + honest residualization; method template, not SD | method context |
| Truth geometries orthogonal across tasks (arXiv:2506.08572) | — (analysis) | linear "truth" directions fail to transfer across tasks | methodology | **verified** | Transfer warning for any acceptance probe (C08/I16) | method context |

## Verification log

### I21, 2026-07-10 (planning-pass four)

| arXiv ID | Title (short) | Primary URL | Status | Notes |
|---|---|---|---|---|
| 2603.01639 | Learning to Draft (LTD) | https://arxiv.org/abs/2603.01639 | verified | RL co-adaptive draft+verify policies |
| 2605.02888 | SpecKV | https://arxiv.org/abs/2605.02888 | verified | MLP on draft confidence/entropy; compression-aware γ |
| 2604.14682 | Acceptance Dynamics Across Cognitive Domains | https://arxiv.org/abs/2604.14682 | verified | Domain-level acceptance; early/late position bins only |
| 2606.30265 | When Is a Draft Accepted? (theory) | https://arxiv.org/abs/2606.30265 | **verified on primary arXiv** | **Mirror-found item.** Primary archive exists (v1, 29 Jun 2026) |

### I21, 2026-07-11 (round-2 corrections + sweep additions)

| arXiv ID | Title (short) | Primary URL | Status | Notes |
|---|---|---|---|---|
| 2510.20064 | Not-a-Bandit | https://arxiv.org/abs/2510.20064 | **verified; ID fix** | Primary Not-a-Bandit ID. Prior row 2506.00285 is *Lazy Heuristic Search for POMDPs* (cs.RO) — unrelated |
| 2506.00285 | Lazy Heuristic Search for POMDPs | https://arxiv.org/abs/2506.00285 | verified-as-wrong-ID | Do not cite as Not-a-Bandit |
| 2607.05147 | DSpark | https://arxiv.org/abs/2607.05147 | **verified; system fix** | Abstract: deployed in authors' production serving stack under live traffic vs MTP-1; **never claims SGLang**. Confidence-scheduled verification + semi-AR draft |
| 2412.18910 | AdaEAGLE | https://arxiv.org/abs/2412.18910 | verified | LDLP adaptive draft length; EAGLE; closest structural C10 prior |
| 2501.19309 | Judge Decoding | https://arxiv.org/abs/2501.19309 | verified | Target-embedding judge head; relaxes alignment/losslessness |
| 2606.07710 | WhiFlash | https://arxiv.org/abs/2606.07710 | verified | Token-level AR↔diffusion drafter routing |
| 2502.13652 | C2T | https://arxiv.org/abs/2502.13652 | verified | Classifier-based tree construction |
| 2402.12374 | Sequoia | https://arxiv.org/abs/2402.12374 | verified | Hardware-aware tree SD |
| 2405.04304 | DISCO | https://arxiv.org/abs/2405.04304 | verified | Dynamic speculation lookahead |
| 2410.18351 | AdaEDL | https://arxiv.org/abs/2410.18351 | verified | Entropy lower-bound early draft stop |
| 2509.01083 | DSDE | https://arxiv.org/abs/2509.01083 | verified | KLD-stability dynamic length |
| 2406.14066 | TurboSpec | https://arxiv.org/abs/2406.14066 | verified | Closed-loop goodput speculation control |
| 2606.01813 | CaDDTree | https://arxiv.org/abs/2606.01813 | verified | Cost-aware diffusion draft trees |
| 2602.03708 | SemanticSpec | https://arxiv.org/abs/2602.03708 | **full-text verified** | Semantic-sequence probes on draft+target hiddens during verify; see G2 |
| 2509.10625 | No Answer Needed | https://arxiv.org/abs/2509.10625 | verified | Question-only correctness linear probes |
| 2606.14530 | Code correctness pre-gen probes | https://arxiv.org/abs/2606.14530 | verified | Pre-generation + residualization methodology |
| 2506.08572 | Geometries of Truth Orthogonal | https://arxiv.org/abs/2506.08572 | verified | Cross-task non-transfer of truth directions |

### C04 impact (detail)

arXiv:2604.14682 establishes that **task domain** is a strong predictor of
acceptance under a fixed draft–target pair, that entropy–acceptance correlation
is weak (ρ ≈ −0.20 to −0.15), and that coarse early vs late position bins can
differ at free-choice (depth-1) slots. It does **not**:

- annotate overlapping **token categories** (punctuation, code delimiters,
  function words, content words, numbers/operators, named entities, sentence /
  clause boundaries, reasoning transitions, repeated/copied spans);
- report acceptance by those categories;
- use internal model representations or interventions.

Therefore C04 (token-category and generation-phase systematic differences)
remains a distinct, finer-grained claim. Position 2604.14682 as **domain-level
context and a control reference**, not as covering the acceptance atlas.

### C04 domain-control positioning (G4; atlas must include)

For the acceptance atlas (I18) and any C04 write-up, 2604.14682 should read as
**our control**, not our competitor. Required comparisons:

1. **Reproduce domain-marginal acceptance** on our primary pair and prompt suite
   (code / math / reasoning / chat or our contract domains): same qualitative
   story as 2604.14682 — domains differ; entropy alone is weak. Report this as
   a control panel / appendix table linked to run IDs.
2. **Then** break each domain by **overlapping token category × generation
   phase** (I11 `cas.annotate` v1.0.0): this is the new axis. If category/phase
   structure vanishes after conditioning on domain, C04 weakens; if it remains
   within domain, C04 is strictly finer than 2604.14682.
3. **Do not** claim 2604.14682 measured token categories. Cite it only for
   domain-grain priors and weak entropy–α.
4. **Optional control:** position bins (early vs late) as in 2604.14682, then
   show phase labels from I11 refine those bins.

### C10 / pre-round novelty (detail; updated 2026-07-11)

No primary-verified work occupies the full C10 cell: **next-round acceptance
(or accepted length) from already-cached verified-context representations,
before any draft compute, under lossless exact decoding, with calibration and
baselines including free frontier entropy/margin, on an independent drafter.**

Narrowing neighbors (must be named in any C10 discussion):

| Paper | Why adjacent | Why not C10 |
|---|---|---|
| AdaEAGLE (2412.18910) | Pre-draft length from target verified-context features | EAGLE setting; uncalibrated; no skip; no full baseline table |
| Judge Decoding (2501.19309) | Target embeddings judge draft tokens | **During verify**; **relaxes losslessness** |
| WhiFlash (2606.07710) | Target-side signals before/at draft choice | Drafter **routing** (AR vs diffusion), not accept/reject or skip |
| SpecDec++ / DSpark | Trained acceptance heads | **Post-draft** (or draft) states; pay draft forward first |
| SpecKV / SVIP / AdaEDL | Entropy/confidence length control | Draft-time or post-hoc signals after drafting starts |
| SemanticSpec (2602.03708) | Probes hidden states | **During verify**; **semantic-relaxed** sequence accept; not pre-round |

Novelty threat to the full C10 cell remains **moderate-low** after this pass
(higher than 2026-07-10 because of AdaEAGLE's structural proximity). Re-check
before freeze. **C10 baselines must include target next-token entropy and
top-1/top-2 margin at the last verified position** (free from the verify pass;
cf. 2606.30265).

### SemanticSpec full-text threat check (G2)

arXiv:2602.03708 ("Beyond Tokens: Semantic-Aware Speculative Decoding…"):

- **What it probes:** multi-layer hidden states of **both draft and target**
  while verifying **drafted semantic sequences** (split on `\n\n`); average-
  pools token hiddens; trains a 3-layer MLP to predict *semantic probability*
  (cluster frequency of meaning-equivalent sequences).
- **When:** online **during verification** of draft sequences — not before
  draft compute from cached verified context alone.
- **Correctness contract:** **semantic-aware / relaxed** acceptance
  (token-level exact match is explicitly the problem they drop); reports
  pass@1 drops vs target-only.
- **Baselines:** SpecReason, Speculative Thinking, token-level SpecSampling —
  **not** entropy/margin/history/domain incremental-information tables.
- **C01 threat:** does **not** scoop the controlled incremental-information
  study of *exact* draft–target acceptance from draft (or target) hiddens vs
  cheap metadata. Different estimand (semantic probability of sequence
  meaning). Status: **no scoop** of C01 science claim; still cite as "internal
  states used for speculation control" prior.
- **C10 threat:** **no scoop**. Timing is mid-verify of draft content; not
  pre-round prediction that can choose skip/length before drafting.

### Rescan additions (2026-07-10)

FASER (2604.20503), PARD-2 (2605.08632), SpecBranch (2506.01979), decentralized
impact-token verify (2511.11733), CSD token-type rescue analysis (2604.13634).
None collapses the mechanistic or pre-round differentiators.

## Deployed practice (G3)

What production open-source engines ship as speculation control. Open-source
project names (vLLM, SGLang, TensorRT-LLM) are fine under D008; no company/lab
names. This is the **cheap systems baseline** I13/I14 must beat or honestly
compare against.

### vLLM

| Control | Signal / rule | Source |
|---|---|---|
| Static `num_speculative_tokens` | Fixed K per step (method-dependent: draft_model, eagle/eagle3, mtp, ngram, suffix, dflash, …) | [Speculative decoding docs](https://docs.vllm.ai/en/latest/features/speculative_decoding/) |
| **Dynamic speculative decoding** | `num_speculative_tokens_per_batch_size`: list of `[start_bs, end_bs, optimal_K]`; K falls as concurrency rises; K=0 disables drafting at high BS | [Dynamic SD docs](https://docs.vllm.ai/en/latest/features/speculative_decoding/dynamic_speculative_decoding/) (2026-07) |
| Queue auto-disable (older/arg path) | Disable speculation for new requests when enqueued request count exceeds a threshold (`speculative_disable_by_batch_size` in older engine-arg docs) | Historical engine-args / RFCs (TurboSpec line: arXiv:2406.14066) |
| Suffix decoding depth | Tree depth / min token probability heuristics | Same speculative-decoding docs |

**Decision rule class:** load / batch-size tiers (and optional queue cutoff).
**Not used:** internal representation probes, token-category atlas, or
calibrated pre-round acceptance models.

### SGLang

| Control | Signal / rule | Source |
|---|---|---|
| Static EAGLE/EAGLE3/MTP/draft/ngram | Fixed `speculative_num_steps` / draft tokens | [Speculative decoding](https://docs.sglang.ai/advanced_features/speculative_decoding.html) |
| **Adaptive speculative decoding** | Per batch-size-range **EMA of accepted draft length**; switches among pre-captured step tiers (e.g. `[1,3,7]`); warmup, update interval, up/down hysteresis, optional ceiling coefficient. EAGLE/EAGLE3 only, `eagle_topk=1` | [Adaptive SD docs](https://docs.sglang.io/advanced_features/adaptive_speculative_decoding.html) |

**Decision rule class:** accept-length EMA + BS-slot tier ladder.
**Not used:** learned acceptance heads in the adaptive path (static methods
only); no pre-round representation probe.

### TensorRT-LLM

| Control | Signal / rule | Source |
|---|---|---|
| Fixed `max_draft_len` | Single draft sequence length per request for draft/target, EAGLE3, NGram, MTP, user drafter | [Speculative decoding](https://nvidia.github.io/TensorRT-LLM/1.2.0rc6/features/speculative-decoding.html) |
| Dynamic length | **Not supported** in one-model path; docs state no dynamic disable of speculation and speedups mainly at low batch sizes | Same docs (Quick Start) |
| MTP relaxed thinking (optional) | `use_relaxed_acceptance_for_thinking` with top-k / delta filters during thinking phase | Same docs |

**Decision rule class:** mostly static max draft length; optional relaxed
accept for thinking tokens on MTP.
**Not used:** online adaptive K from accept-length or representation probes.

### Implication for this repo

Deployed practice is **EMA / batch-size / goodput feedback** on length (or
static K). The science gap remains: (1) measurement of *where* acceptance
information lives and whether it beats free frontier entropy/margin; (2)
pre-round, lossless, calibrated control with a true `skip`. I13/I14 should
include an **engine-style EMA or BS-tier length policy** as a systems baseline
alongside SVIP/BanditSpec, not only paper policies.

## Open queue

1. Primary-source confirm AdaSD (2512.11280), TALON (2601.07353), Talon
   (heterogeneous async — resolve exact arXiv ID), OnlineSpec (2603.12617),
   FlashSpec.
2. Keep rescanning monthly (or before any novelty freeze / preprint) for
   pre-round / cached-context predictors and token-category acceptance atlases.
3. Re-verify whether any OSS port of DSpark's confidence head lands in vLLM's
   `dspark` worker path; paper attribution remains production-stack-not-SGLang.
4. Propose PLAN.md §3 notes (below); Claude owns PLAN.md.

## PLAN.md §3 edit proposals (do not apply here; Claude owns PLAN.md)

```
I21 (2026-07-10, Grok): Verified arXiv:2603.01639, 2605.02888, 2604.14682,
2606.30265 on primary arXiv. Mirror-found item was 2606.30265 — primary URL
https://arxiv.org/abs/2606.30265. C04 not pre-empted (domain-level study only).
Living table: docs/landscape.md.

I21 (2026-07-11, Grok round 2): Not-a-Bandit primary ID → 2510.20064
(2506.00285 is unrelated POMDP work). DSpark (2607.05147) deploys in authors'
production stack, not SGLang — correct any Phase-2 engine wording that cited
"DSpark in SGLang". Landscape + ledger updated with AdaEAGLE/Judge/WhiFlash
and deployed-practice baselines (vLLM Dynamic SD, SGLang adaptive EMA,
TensorRT-LLM static max_draft_len).
```

### I21, 2026-07-22 (novelty re-check for the C10 freeze; adversarially-verified sweep)

Multi-agent verified sweep (100 agents; 24 claims, every one surviving 3-0
adversarial verification; 12 papers verified on primary arXiv pages, most with
full-text reads). Purpose: freeze the "C10 cell unoccupied" statement before the
manuscript (I18/I19).

| arXiv ID | Work (short) | Status | One-line verdict vs the C10 cell |
|---|---|---|---|
| 2412.18910 | AdaEAGLE | verified (full v1 text) | Nearest neighbor, 3/8 criteria: pre-round + target frontier state + lossless; but LENGTH regression, uncalibrated, logit-free (no entropy/margin baselines), no skip, EAGLE head |
| 2501.19309 | Judge Decoding | verified | Target embeddings but on DRAFTED tokens at verify time; relaxes losslessness |
| 2606.30265 | Acceptance theory | verified (full text) | No predictor; margin certificates for strict greedy GROUND C10's mandatory margin baseline |
| 2606.01019 | HVD | verified (full text) | Cache-draft accepted-length routing; needs draft content; no skip; zero hidden-state features |
| 2607.05147 | DSpark | verified (full text) | Within-round prefix-survival → verification-length scheduling; l_r=0 is post-draft discard, not pre-round skip |
| 2607.16673 | SpecLA | verified (full text) | Within-round tree pruning; EAGLE-style drafter; newest in window (v1 18 Jul 2026) |
| 2603.01639 | LTD | verified (full text) | RL draft-depth/verify-size control from draft-tree probabilities; no skip (min depth 1); shares only losslessness |
| 2509.24328 | Speculative Verification | verified (v2 full text) | Third companion model, within-round verification-length adaptation |
| 2605.02888 | SpecKV | verified (full text) | Draft-side γ∈{2,4,6,8} run-length control; γ=0 structurally impossible |
| 2503.05096 | AdaSpec (ex-SpecServe) | verified (2 full-text passes) | Mid-round draft continue/stop from draft confidences; no target-side inputs |
| 2410.18351 | AdaEDL | verified | Within-round draft-entropy early stop; no target distribution access during drafting |
| 2505.15141 | BanditSpec (UCBSpec/EXP3Spec algorithms) | verified (full text) | History-only bandit over configs — functionally our recent-acceptance-history FREE baseline, not a probe |

**Verdict:** the C10 cell (pre-round × target cached-frontier representation ×
lossless exact-match × calibrated × baseline-controlled vs free entropy/margin ×
independent drafter × skip decision × first-token) is **unoccupied** across the
12 papers verified this round PLUS the four adjacent works primary-verified in
the earlier I21 rounds and already tabled above (WhiFlash 2606.07710 = pre-draft
drafter routing, not accept/skip; SemanticSpec 2602.03708 = mid-verify
semantic-relaxed; 2604.14682 = domain-grain analysis; Not-a-Bandit 2510.20064 =
drafter selection). Those four produced no surviving claims in THIS round's
harness (fetch/verify attrition, not refutation) — their scopes rest on the
2026-07-10/11 in-repo primary verifications, which stand.

**Caveats (honest):** (a) absence of other new work in the 2026-06-15→07-22
window is absence of evidence — the citing-papers sweeps for
AdaEAGLE/Judge/WhiFlash returned no surviving claims, so a very recent citer
could exist unseen; re-run the citing sweep at manuscript freeze (the likeliest
occupation path is someone adding calibration+skip+baselines to an
AdaEAGLE-style predictor). (b) SpecLA (4 days old), DSpark (16 days old), and
the theory paper (single-author v1) may still change. (c) DSpark head-input
wording differs between our two full-text reads — re-quote before quoting.
(d) One refuted claim recorded: AdaSpec does NOT articulate the pre-round gap
(0-3) — never cite it as naming C10's gap.

**Open method question worth a manuscript paragraph:** does 2606.30265's margin
certificate imply a CEILING on how far any probe can beat the margin baseline
under strict greedy? (If margin certifies acceptance in a region, probe headroom
lives only in the uncertified region — connects our +0.04–0.10 AUROC lift to
theory.)
