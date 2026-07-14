# Content-Aware Speculation Control — Methodology & Implementation

How the harness, evaluation corpus, and per-question analyses are built. Every
reported number is script-generated from immutable, write-once trace artifacts.

- **Date:** 2026-07-13
- **Engine:** exact greedy
- **Pairs:** Qwen, Llama
- **Companion:** Progress Debrief
- **Source artifact:** https://claude.ai/code/artifact/d6fb20a7-e44b-47ef-bcf4-e1c83221ae76

---

## A. The harness

An **exact, lossless** speculative-decoding engine: the target's greedy argmax is
the reference, so committed output is token-identical to target-only greedy.
Correctness is separated from timing by a dual-mode seam (D021): an eager,
hookable path for all science, and a compile/fuse path for timing only.

### Engine components

- **Target 7B** — Ground-truth verifier. One batched forward per round over
  `[uncached gap + proposals]`; its **argmax is the reference output** (lossless).
  Qwen2.5-7B-Instruct, pinned revision, bf16, eager.
- **Draft 0.5B** — Cheap proposer sharing the target's vocabulary (asserted
  equal). Autoregressively proposes up to L tokens, one forward each.
- **Verify + commit** — Pure stdlib rule: accept the maximal prefix where
  `d_i == t_i` (length k), emit `d_1..d_k + t_(k+1)`, report `first_rejection`.
  Torch-free, CPU unit-tested.
- **KV caches** — Two persistent caches with covered-length trackers; each round
  re-feeds only the bounded gap, then both crop back to `context + k` so rejected
  and bonus tokens are re-processed next round.
- **Signal recorder** — Per position: draft entropy + top1-top2 margin (fp32,
  nats). D018 byproduct: target frontier entropy/margin at the accepted tail,
  timed under a tracing clock so the sync stays inside end-to-end latency.
- **Controller** — A `RoundContext -> L` callable over the action set
  `{0,1,2,3,4,6,8}`. Both the pick and the commit run inside the controller
  clock, so all overhead is charged to latency.

### One speculative round (decode loop)

1. **Controller pick** (timed): `L = policy(ctx)`.
2. **Draft** (timed): re-feed draft gap, greedy-propose up to L; record
   entropy/margin; optional early stop-rule.
3. **Verify** (timed): one batched target forward over gap + proposals; argmax
   the last L+1 positions.
4. **Commit** (timed): accept prefix k; emit accepted + one target bonus token;
   record `first_rejection`.
5. **Cache rollback**: crop both caches to `context + k`.
6. **Emit + stop**: append emitted ids; halt at first eos (greedy-identical) or
   max_new.
7. **Record** (timed): frontier target entropy/margin, the freshest
   pre-next-round signal.
8. **Trace**: write RoundTrace (proposed ids, target argmax, k, latencies) and
   loop.

### Counterfactual labeling (the key idea)

Each round stores the full **match vector** `m_i = (d_i == t_i)` at every drafted
position, independent of where acceptance actually stopped. Exact-greedy
acceptance is the all-true prefix, so any shorter action H's accepted length is
just the first-false-truncated prefix of `m[:H]`. A **fixed_8** run therefore
labels **all** actions {0..8} offline from one trace, with **zero extra
forwards**.

### Equivalence gate + dual mode

A GPU test asserts `output == greedy_reference` for every fixed length
(independent code path); the **fp32** knob forces batched-verify and
sequential-decode to agree at ~1e-7. Gate passes **118/118** on the Llama pair.
Timing-only fast paths (compile / fused attention) never touch the eager capture
path and must re-pass equivalence before yielding any scientific number.

> **Figure A — One round.** Solid = data flow within the round; dashed =
> pre-draft feedback to the controller. Draft cost tracks layer count, not
> parameters (launch-bound), so the 0.5B draft and 7B verify are comparably
> priced on the eager harness.

---

## B. Evaluation corpus and collection

Under greedy exact-match, per-token acceptance is deterministic in
(draft, target, context), so **corpus breadth is the load-bearing decision**
(D022). Corpus v2 spans 7 representativeness axes over 9 public sources, split by
prompt group before any trace-derived fit.

| Source                 | Axis          | License (SPDX)              | Prompts |
|------------------------|---------------|----------------------------|--------:|
| HumanEval              | code          | MIT                        |     164 |
| MBPP (sanitized)       | code          | CC-BY-4.0                  |     200 |
| GSM8K                  | math          | MIT                        |     200 |
| MT-Bench               | chat          | Apache-2.0                 |      80 |
| OASST1                 | chat          | Apache-2.0                 |     150 |
| WMT14 de-en            | translation   | shared-task terms          |     150 |
| Natural Questions Open | qa_rag        | CC-BY-SA-3.0               |     200 |
| JSONSchemaBench        | structured    | MIT                        |     150 |
| CNN / DailyMail        | summarization | Apache-2.0 code; row-ids only | 200 |
| **Total** — 9 sources · 7 axes · prompt-grouped | | |   **1,494** |

- **Split** — **Prompt-grouped** by `group_key` = prompt_hash (conversation-tree
  id for OASST1; dedup-cluster for near-duplicates). Split runs before any fit;
  the manifest is then frozen. **Token-level random splits are prohibited.**
- **Provenance** — Every record carries `spdx` + `row_id`; copyright-text sources
  ship row-ids only with in-house completions; a NOTICE file accompanies the
  release.
- **Versioning** — Built to `data_v2/` (D022) so the sealed v1 corpus and all
  prior analyses stay valid.

> **Figure B — Collection pipeline.** 9 sources → normalize + license-tag →
> dedup + hash (group_key) → prompt-grouped split (freeze manifest) → dev/test →
> sealed fixed_8 parquet. Sealed fixed_8 traces are the single immutable input to
> every downstream analysis; nothing is recomputed by hand.

---

## C. Per-question implementation

All three questions are answered **offline** from the same sealed fixed_8 traces.
One trace per prompt yields per-position acceptance for every action, so the
controllers, the draft × domain matrix, and the atlas are replays, not new GPU
runs.

> **Figure C — Sequence for one round** (solid) with pre-draft feedback (dashed).
> The payoff: the counterfactual match vector makes every downstream question a
> cheap replay of sealed traces — ONE fixed_8 trace labels all actions {0..8}
> with zero extra forwards → RQ2 headroom · RQ3 draft × domain matrix · LHF ·
> acceptance atlas.

### Metric and confidence

```
# serving-cost efficiency (draft priced at 0.1x a verify forward)
eff_serving = Σ emit(L) / Σ cost(L)     cost(L) = 1 + 0.1·L,  emit(L) = accepted_under(match, L) + 1
# accepted_under = leading all-match prefix of the first L proposals; +1 = target bonus token
CI = prompt-grouped bootstrap, 2000 resamples over prompts, 95% percentile interval, plus P(Δ≤0)
```

### How each question is computed

- **RQ2 · length** — Replay each policy on the trace's counterfactual labels;
  accumulate (emit, cost) per prompt; compare entropy-stop (τ frozen from dev) vs
  best fixed by **prompt-grouped bootstrap**. Bandits replay online across the
  stream.
- **RQ3 · routing** — Drift-free, teacher-forced acceptance of each draft against
  a shared target greedy continuation, per domain/axis. **Size-matched paired**
  bootstrap CIs (specialist minus same-size general) over prompts; oracle
  router = per-axis best draft.
- **RQ1 · joint** — Compose the length controller with the routing oracle.
  Because the routing axis is a no-go, the joint value equals the length
  controller alone: RQ1 reduces to RQ2 on this pair.

---

*Naming follows the G2 gate ("representation" / "diagnostic signal" until
interventions pass) and D008 (dates and generic descriptions only). Action set
{0,1,2,3,4,6,8}; 0 = skip (pure autoregressive step). Reproduced from sealed runs
sweep-2026-07-11T203836 (Qwen) and sweep-llama-f8-2026-07-13 (Llama). Companion
document: Progress Debrief.*
