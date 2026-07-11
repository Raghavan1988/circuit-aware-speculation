# Living Comparison Table (issue I21)

Adjacent work on adaptive / mechanistic speculative decoding. Maintained so
novelty claims stay honest; re-scan before freezing claims or submitting.
`Verified` = checked against a primary source (arXiv/proceedings/official repo);
`search-only` = surfaced via search and awaiting primary confirmation.

Differentiation this repo asserts (see docs/RESEARCH_SPEC.md): none of these
provides a *causally validated, transferable internal* account of acceptance,
nor a pre-round (pre-draft-compute) signal. That is the gap.

| Work | Adapts | Signal | Verified? | Relation | Baseline / cite |
|---|---|---|---|---|---|
| SpecDec++ (arXiv:2405.19715, COLM 2025) | draft length | trained acceptance head on draft hidden states | verified | closest learned-signal analog; reads *post-draft* final states | baseline (policy 8, I22) |
| BanditSpec (arXiv:2505.15141) | length/config | bandit, stopping-time regret | verified | length-only online control | baseline (policy 7, I09) |
| SVIP (arXiv:2411.18462) | length | draft entropy stop rule | verified | length-only, cheap signal | baseline (policy 5, I08) |
| MetaSD (ENLSP 2024) | drafter choice | per-step UCB | verified | drafter-only; no length | cite (deferred routing) |
| Not-a-Bandit (arXiv:2506.00285) | drafter | no-regret selection | verified | drafter-only | cite (deferred routing) |
| TapOut (arXiv:2511.02017) | length strategy | bandit over strategies | verified | meta-level length control | cite; optional compare |
| AdaSD (arXiv:2512.11280) | length | adaptive | search-only | length-only | cite |
| TALON (arXiv:2601.07353) | token trees | adaptive trees | search-only | tree-length | cite |
| OnlineSpec / "When Drafts Evolve" (arXiv:2603.12617) | draft weights | online learning | search-only | different axis (training) | cite |
| Learning to Draft (arXiv:2603.01639) | drafting | RL for throughput | search-only | throughput-optimized draft | verify (I21); cite |
| Multi-drafter alignment feedback (arXiv:2604.05417) | drafter pool | alignment feedback | search-only | deferred-routing relative | cite |
| Acceptance dynamics across domains (arXiv:2604.14682) | — (analysis) | task-conditioned acceptance | search-only | **may pre-empt claim C04 (atlas)** | verify (I21); reposition C04 if covered |
| SpecKV (arXiv:2605.02888) | length | draft confidence under KV compression | search-only | length under compression | verify (I21); cite |
| Theory of acceptance (arXiv:2606.30265) | — (theory) | acceptance bound/estimator | search-only (mirror) | **may occupy Track B "theory" bet** | verify primary archive; constrains Track B |
| Task detection + heterogeneous drafting (arXiv:2505.08600) | routing | task classifier | search-only | closest to deferred routing | cite |
| SGLang adaptive speculation (docs, 2026) | num_steps (server) | EMA of accepted length | verified | engine-level length tiers; EAGLE-only, no per-request/custom | cite; Phase-2 integration target |
| DSpark in SGLang (2026-07-06) | per-request verify length | trained confidence head + calibration | verified | trims verify *after* drafting; needs a head | cite; contrast with pre-round bet (I23) |

## Verification queue (I21)

1. Confirm arXiv:2603.01639, 2605.02888, 2604.14682, 2606.30265 against primary
   sources; 2606.30265 was found via a mirror — confirm the arXiv/venue record.
2. Resolve the name collision: "TALON" (2601.07353) vs "Talon" (heterogeneous +
   async drafting) are different works; cite distinctly.
3. Record the C04 verdict in docs/CLAIMS_LEDGER.md once 2604.14682 is read.
