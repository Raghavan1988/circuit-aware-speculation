# Acceptance-Aware Speculation: Results Summary

> **Prepared 2026-07-13** (supersedes the 2026-07-12 version)
> **Target model:** Qwen2.5-7B-Instruct · **Drafts:** Qwen2.5 0.5B and 1.5B (general and specialized) · **Compute:** Modal, A100

---

## TL;DR

All three roadmap research questions now have answers, plus several findings the roadmap did not anticipate.

- ✅ **Adaptive draft length works.** A simple, training-free entropy rule beats the best fixed draft length by **+11.2% efficiency** on held-out data and cuts wasted drafting by **62%**.
- ❌ **Draft-model routing does not work here.** Domain-specialized drafts tie or lose to a same-size generalist — even on their own home domain. "Route the draft" has no headroom.
- ➡️ **The joint question collapses.** Since routing adds nothing, the joint (draft + length) question reduces to the length question.
- 📊 **The core findings stand.** Acceptance has strong, interpretable structure by token type (the *atlas*); cheap free signals predict acceptance well and internal-state probes do not beat them; and the research harness makes the small draft artificially expensive (a *launch-bound* effect), which defers any wall-clock speed claim.
- 🗂️ **A better corpus is built.** Version 2: 1,494 prompts, 7 workload axes, 9 datasets, license-audited. A replication of the routing result on it is running now.

---

## 1. The research questions and where they stand

| RQ | Question | Answer | Key evidence |
|---|---|---|---|
| **RQ2** | Does adaptive draft length beat the best fixed length? | **Yes** | +11.2% efficiency held-out; 62% less wasted drafting; generalizes dev→test (+11.3 → +11.2) |
| **RQ3** | Does routing among domain-specialized drafts add more? | **No** | Specialists tie or lose to a size-matched generalist; the math specialist is worse even on math; an oracle router equals the general 1.5B draft in every domain |
| **RQ1** | Does the joint (draft + length) controller beat the best fixed policy in net speedup? | **Collapses to RQ2** | The draft axis adds nothing (RQ3); the net-speedup metric is blocked by the launch-bound harness |

---

## 2. What we built

- **Exact greedy speculative-decoding engine** — proven token-identical to running the target alone (73/73 equivalence checks in full precision).
- **A sealed corpus** — 644 prompts across code, math, chat, and summarization, decoded under 8 policies → ~302,000 labelled draft tokens as immutable, checksummed data.
- **An offline policy-replay evaluator** — every round records the target's choices at all drafted positions, so any length policy can be evaluated exactly without re-running a model.
- **A draft-pool comparison harness** — all candidate drafts are scored on the identical target continuation, so comparisons are apples-to-apples.
- **A version-2 corpus** — 1,494 prompts, 7 axes, 9 public datasets, with per-dataset license metadata, built as a separate versioned artifact.
- **A measurement-first discipline** — every reported number is script-generated from sealed data; negative results are recorded in the claims ledger.

---

## 3. Main results

### 3.1 Adaptive draft length beats the best fixed length — *answers RQ2*

We replayed length policies over the sealed traces, tuned the entropy threshold on the development split, froze it, and evaluated on the held-out test split — using efficiency metrics that do not depend on wall-clock timing (drafting priced at a serving-realistic fraction of verification cost).

| Policy (held-out test) | Tokens/round | Wasted per emitted token | Efficiency |
|---|---|---|---|
| Best fixed length (8) | 4.32 | 1.08 | 2.40 |
| **Entropy-stop controller** | 3.85 | **0.41** | **2.67 (+11.2%)** |
| Oracle (per-round best) | 4.32 | 0.02 | 3.23 |

**The win:** the controller beats the best fixed length by **+11.2%**, with a prompt-grouped bootstrap 95% CI of **+10.3 to +12.1%** over 322 held-out prompts (probability of zero-or-negative gain below 1 in 2000). It:

- cuts wasted draft tokens by **~62%** while keeping ~89% of the yield (the two policies' wasted-token CIs do not overlap);
- captures about **one third** of the oracle headroom;
- transfers dev→test without retuning (+11.3 → +11.2);
- wins within **every** individual domain.

**Why it works — it's the context, not the learning:**

- Online bandit controllers (ε-greedy and UCB over lengths) converge to the best fixed length (efficiency 2.40) and **cannot beat it** — expected, since a context-free bandit just learns the best single arm, which *is* the fixed baseline.
- Only the contextual rule that reads the draft's per-step **entropy** beats it. Recent-acceptance history alone adds just 0.5%.
- The signal is well calibrated: acceptance falls monotonically from **0.97 to 0.16** across entropy bins.
- Remaining headroom is domain-skewed: the controller over-drafts on math and under-drafts on summarization and chat.
- A coarse four-length menu retains **~98%** of the value — simplifying deployment.

### 3.2 Specialized drafts do not beat a generalist — *answers RQ3*

We measured a full draft-by-domain acceptance matrix: five tokenizer-compatible drafts, each scored on the same target continuations, with size-matched statistical comparisons.

| Draft | chat | code | math | summarization |
|---|---|---|---|---|
| general 0.5B | 0.738 | 0.898 | 0.910 | 0.672 |
| Coder 0.5B | 0.675 | 0.899 | 0.877 | 0.603 |
| **general 1.5B** | **0.789** | **0.917** | **0.928** | **0.736** |
| Math 1.5B | 0.662 | 0.862 | 0.919 | 0.554 |
| Coder 1.5B | 0.753 | 0.917 | 0.917 | 0.695 |

**Findings** (prompt-grouped 95% CIs):

- Code specialists **statistically tie** their size-matched generalist on code (differences ~−0.003 to −0.004; intervals cover zero).
- The math specialist is **significantly worse** than the same-size generalist *even on math* (−0.009; interval entirely below zero).
- A perfect per-domain router would just pick the general 1.5B everywhere — so routing among off-the-shelf specialists adds essentially nothing.

**Why:** acceptance measures how well the draft matches the *target model's own distribution*. A general-instruct draft from the same family already matches a general-instruct target better than a domain expert does. **Draft size helps; domain specialization does not.** A replication on the larger v2 corpus (translation, retrieval QA, structured output) is running now.

### 3.3 The acceptance atlas — *headline positive finding*

Acceptance rate varies systematically from **0.52 to 0.88** depending on the kind of token being drafted, measured over **302,464 tokens** in 38 category-by-phase cells with confidence intervals.

- **Easy (structural):** code delimiters, operators, numbers — ~0.85 to 0.88.
- **Hard (semantic/surprising):** named entities, reasoning transitions — ~0.58 to 0.67.

This is the measured mechanism behind the adaptive-length win, and it is independent of any timing question.

### 3.4 Cheap signals are strong; internal-state probes do not beat them

- Free signals the draft already produces (entropy, confidence margin, recent acceptance history) predict token acceptance at **AUROC ~0.87** on the probe subset.
- Linear probes on the draft's internal representations peak at **0.803** (layer 18 of 24) and add **at most +0.006** when combined with the cheap signals.

This is a well-powered **null result**: for this model pair, the linearly accessible acceptance information is already in the free signals.

### 3.5 The harness makes the small draft artificially expensive — *systems finding*

Running the 0.5B draft one token at a time costs **~24 ms/token** — nearly the cost of one 7B verification pass (**~30 ms**) — because single-token decoding is dominated by fixed per-layer dispatch overhead rather than compute.

- Fused attention and standard compilation **do not fix it**; the real fix is a fixed-shape cache plus graph capture (deliberately deferred serving engineering).
- **Consequence:** adaptive routing is worth only ~5% wall-clock on this harness, but the same acceptance data implies **25 to 46%** at serving-grade draft cost.

We therefore report **efficiency** (§3.1) rather than wall-clock speedup, and state this limitation openly.

### 3.6 A representative corpus, version 2

A systematic survey (127 candidate datasets, 94 verified with licenses) concluded that the original 4-domain corpus under-represents the acceptance extremes adaptive methods exploit.

Corpus v2: **1,494 prompts across 7 axes** (code, math, chat, translation, retrieval QA, structured JSON output, summarization) from 9 public datasets, each with license and provenance metadata, plus a licenses manifest. Copyright-sensitive sources ship row identifiers rather than raw text. The original corpus and all results on it remain intact; v2 is a separate versioned artifact.

---

## 4. What this means

The contribution is now a coherent story where positive and negative results reinforce each other:

1. **The positive (RQ2).** Acceptance has strong, interpretable structure (the atlas), and a training-free controller reading a cheap signal converts that structure into a real efficiency win over any fixed policy.
2. **The two nulls (RQ3 + probes).** Two natural-sounding upgrades do not pay: internal-state probes do not beat free signals, and domain-specialized drafts do not beat a same-size generalist. Both are well-powered nulls that save the community effort and sharpen where the real levers are — draft size, target matching, and length adaptation.
3. **Honest scope on speed.** Efficiency gains are demonstrated now; wall-clock gains await a serving-grade draft path, and we quantify exactly how much is waiting (**25 to 46%**).

---

## 5. Next steps

1. **Report the v2 routing replication** (running now; expected within hours).
2. **Decide on the full 8-policy v2 sweep** (~2.3× the original sweep cost, roughly $60–100 of A100 time) — the main remaining experiment, recomputing the atlas and adaptive-length result on the representative corpus. *Before it runs:* pin dataset revisions for the streaming loaders and apply the recorded split-labeling fix.
3. **Complete the controller table** — evaluate ε-greedy and UCB length bandits through the same replay harness (cheap, CPU-only).
4. **Package the citable dataset** — fix the split column, strip copyright text to row identifiers, write the dataset card, release per the staged policy.
5. **Write the manuscript** around the atlas, the two nulls, the adaptive-length win, and the launch-bound systems finding, with figures generated from the sealed artifacts.
6. **Deferred but scoped** — the fixed-cache fast draft path (unlocks the wall-clock claim), and target-distilled draft heads as the principled alternative to domain specialists.

---

## 6. Risks

- **v2 could shift v1 conclusions.** *Mitigation:* identical protocol on both corpora; the routing replication is already running; differences will be reported, not smoothed over.
- **Wall-clock claim blocked** until the serving-grade draft path exists. We will not state speed numbers from the current harness beyond the launch-bound characterization.
- **Unpinned v2 loaders.** Several stream data without pinned revisions; v2 results runs require pinning first.
- **Split-labeling bug** in the sweep runner must be fixed before any new sweep. (The existing corpus recovers the split at analysis time, so no current result is affected.)
- **Llama replication blocked** — access to the 1B draft repo is still not granted (the 8B target appears approved). Findings are currently single-family (Qwen).
- **Field velocity** — adjacent speculative-decoding papers appear monthly; the staged-release policy (preprint when science gates pass) mitigates being scooped.
- **Budget** — the only remaining discretionary spend is the v2 sweep; all other next steps are CPU-level costs.

---

## 7. Replication status

Second model family (Llama 3.1 8B target with 3.2 1B draft): wiring is complete and tested; blocked on gated access to the 1B draft repository. Within-family replication of the routing result on corpus v2 is in progress.

---

## Appendix: key numbers

| Item | Value |
|---|---|
| Equivalence gate (full precision) | 73/73 token-identical |
| Corpus v1 / corpus v2 | 644 prompts, 4 axes / 1,494 prompts, 7 axes |
| Labelled draft tokens (v1 sweep) | ~302,000 |
| Atlas cells / acceptance range | 38 / 0.52 to 0.88 |
| Cheap-signal predictor AUROC | ~0.84 (full dev); 0.870 (probe subset) |
| Best hidden-state probe AUROC | 0.803 (layer 18); combined adds at most +0.006 |
| Adaptive length vs best fixed (held-out) | +11.2% efficiency, 95% CI +10.3 to +12.1; wasted tokens 0.41 vs 1.08 per emitted |
| Bandit controllers (ε-greedy, UCB) | converge to best fixed (2.40); do not beat the contextual rule (2.67) |
| Oracle headroom captured by the controller | ~33% |
| Specialist vs generalist drafts | ties on code; significantly worse on math (its own domain) |
| Draft cost on this harness | ~24 ms/token (launch-bound); verify ~30 ms |
| Routing headroom (this harness / serving-grade) | ~5% / 25 to 46% |
