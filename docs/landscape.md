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
context. That is the gap.

**Last full verification pass:** 2026-07-10 (Grok, I21). All four planning-pass
queue items confirmed on primary arXiv (see §Verification log).

## Comparison table

| Work | Adapts | Signal | Mechanistic? | Verified? | Relation / contrast here | Baseline / cite |
|---|---|---|---|---|---|---|
| SpecDec++ (arXiv:2405.19715) | draft length | trained acceptance head on **post-draft** hidden states | no (learned head) | verified | closest learned-signal analog; pays draft forward first | baseline (policy 8, I22) |
| BanditSpec (arXiv:2505.15141) | length/config | bandit, stopping-time regret | no | verified | length-only online control | baseline (policy 7, I09) |
| SVIP (arXiv:2411.18462) | length | draft entropy stop rule | no | verified | length-only, cheap signal | baseline (policy 5, I08) |
| MetaSD (ENLSP 2024) | drafter choice | per-step UCB | no | verified | drafter-only; no length | cite (deferred routing) |
| Not-a-Bandit (arXiv:2506.00285; also 2510.20064v2 cross-ref) | drafter | no-regret selection | no | verified | drafter-only | cite (deferred routing) |
| TapOut (arXiv:2511.02017) | length strategy | bandit over strategies | no | verified | meta-level length control | cite; optional compare |
| AdaSD (arXiv:2512.11280) | length | adaptive length | no | search-only | length-only | cite |
| TALON (arXiv:2601.07353) | token trees | adaptive trees | no | search-only | tree-length; **≠** Talon (heterogeneous async) | cite |
| Talon (heterogeneous + async drafting; 2025) | drafter/system | heterogeneous + async | no | search-only | name collision with TALON above; verify primary ID | cite |
| OnlineSpec / "When Drafts Evolve" (arXiv:2603.12617) | draft weights | online learning | no | search-only | different axis (training) | cite |
| **Learning to Draft / LTD** (arXiv:2603.01639) | draft+verify budgets | RL co-adaptive policies; throughput of draft–verify cycle | no | **verified** | throughput-optimized coordination; no internal acceptance localization | cite (I21 ✓) |
| Multi-drafter alignment feedback (arXiv:2604.05417) | drafter pool | alignment feedback | no | search-only | deferred-routing relative | cite |
| **Acceptance dynamics across domains** (arXiv:2604.14682) | — (analysis) | task/domain-conditioned acceptance; weak entropy–α correlation; early/late position bins | no (empirical atlas at **domain** grain) | **verified** | **C04 context, not pre-emption** — domain-level + coarse position bins only; no overlapping token-category labels, no internal representations | cite as domain control; see C04 ledger note |
| **SpecKV** (arXiv:2605.02888) | length γ | draft confidence + entropy under KV compression (MLP) | no | **verified** | length under compression; post-draft signals; ~0.34 ms overhead | cite (I21 ✓) |
| **Theory of acceptance** (arXiv:2606.30265) | — (theory) | KL/margin certificates for greedy, relaxed, top-m, entropy-thresholded, tree acceptance | theory only (not internal) | **verified (primary arXiv)** | was found via mirror; **primary archive confirmed** `https://arxiv.org/abs/2606.30265`. Constrains Track B pure-theory bet; complementary, not competitive with localization | cite; Track B theory option narrowed |
| Task detection + heterogeneous drafting (arXiv:2505.08600) | routing | task classifier | no | search-only | closest to deferred routing | cite |
| FASER (arXiv:2604.20503) | per-request draft length in continuous batch; early prune in verify | systems phase management | no | search-only (rescanned) | serving-side fine-grained phase control; not representation-level | cite |
| PARD-2 (arXiv:2605.08632) | dual-mode drafting | confidence-adaptive tokens | no | search-only (rescanned) | draft-structure + confidence; post-draft | cite |
| SpecBranch (arXiv:2506.01979) | hybrid draft length | draft confidence + target features | no | search-only | hybrid adaptive; still post-draft / feature reuse | cite |
| FlashSpec (search hit 2026-06) | draft selection | online bandit | no | search-only | bandit draft selection | cite if primary confirmed |
| Decentralized SD / impact tokens (arXiv:2511.11733) | verification strictness | high- vs low-impact tokens (punctuation/fillers relaxed) | no | search-only | token-role heuristics for **relaxed verify**, not acceptance atlas of exact SD | cite as adjacent to C04 granularity |
| CSD frequency-guided candidates (arXiv:2604.13634) | candidate rescue | frequency / semantic gating | no | search-only | post-hoc analysis of rescued tokens by type (math fmt, punct, synonyms, connectives) — descriptive, not mechanistic | cite |
| SGLang adaptive speculation (docs, 2026) | num_steps (server) | EMA of accepted length | no | verified | engine-level length tiers; EAGLE-only | cite; Phase-2 integration target |
| DSpark in SGLang (2026-07-06) | per-request verify length | trained confidence head + calibration | no | verified | trims verify *after* drafting; needs a head | cite; contrast with pre-round bet (I23 / C10) |

## Verification log (I21, 2026-07-10)

| arXiv ID | Title (short) | Primary URL | Status | Notes |
|---|---|---|---|---|
| 2603.01639 | Learning to Draft (LTD) | https://arxiv.org/abs/2603.01639 | verified | RL co-adaptive draft+verify policies; 2.24×–4.32× vs static; compares to EAGLE-3 |
| 2605.02888 | SpecKV | https://arxiv.org/abs/2605.02888 | verified | MLP on draft confidence/entropy; compression-aware γ; open artifacts claimed |
| 2604.14682 | Acceptance Dynamics Across Cognitive Domains | https://arxiv.org/abs/2604.14682 | verified | Domain-level acceptance study (code/math/reasoning/chat); position early/late bins only |
| 2606.30265 | When Is a Draft Accepted? (theory) | https://arxiv.org/abs/2606.30265 | **verified on primary arXiv** | **This was the mirror-found item.** Primary archive record exists (v1, 29 Jun 2026). Theory of deterministic/local acceptance certificates; eval on Qwen3 |

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

### C10 / pre-round novelty (detail)

No rescanned work predicts next-round acceptance from **already-cached verified
context representations before any draft compute**. Closest neighbors:

- SpecDec++ / DSpark: trained heads on **draft** (or post-draft) states;
- SpecKV / SVIP: draft entropy/confidence after drafting starts;
- 2606.30265: theory of acceptance events, not a deployed pre-round predictor.

Novelty threat to C10 remains **low** after this pass; re-check before freezing.

### Rescan additions (2026-07-10)

Added to table as search-only or verified where primary URL confirmed: FASER
(2604.20503), PARD-2 (2605.08632), SpecBranch (2506.01979), decentralized
impact-token verify (2511.11733), CSD token-type rescue analysis (2604.13634).
None collapses the mechanistic or pre-round differentiators.

## Open queue

1. Primary-source confirm AdaSD (2512.11280), TALON (2601.07353), Talon
   (heterogeneous async — resolve exact arXiv ID), OnlineSpec (2603.12617),
   FlashSpec.
2. Keep rescanning monthly (or before any novelty freeze / preprint) for
   pre-round / cached-context predictors and token-category acceptance atlases.
3. Propose a one-line PLAN.md §3 note: "I21 2026-07-10: all four queue items
   primary-verified; 2606.30265 was the mirror item and is on arXiv; C04 not
   pre-empted by 2604.14682 (domain grain only)."

## PLAN.md §3 edit proposal (do not apply here; Claude owns PLAN.md)

```
I21 (2026-07-10, Grok): Verified arXiv:2603.01639, 2605.02888, 2604.14682,
2606.30265 on primary arXiv. Mirror-found item was 2606.30265 — primary URL
https://arxiv.org/abs/2606.30265. C04 not pre-empted (domain-level study only).
Living table: docs/landscape.md.
```
