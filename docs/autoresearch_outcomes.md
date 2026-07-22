# Autoresearch Outcomes — Pre-round Acceptance Signal (I13 / I23 / C10)

Outcomes of the generator–critic autoresearch loop (`docs/generator_critic.md`,
D023) run against the sealed Qwen-v1 fixed-8 traces. This is a **development-set,
tentative** report. No claim here is frozen; **C10 remains `UNTESTED`** in
`docs/CLAIMS_LEDGER.md`. Per D020, a predictive survivor is a **diagnostic
signal / representation**, never a "circuit" or "mechanism", until interventions
(I15/G2) pass — that language is used throughout.

## Status banner

- **Split:** development only. The test split was captured but never fitted or
  selected on; it stays frozen.
- **Replication (done):** validated across **two model families** (Qwen, Llama) and
  **two corpora** (v1 4-domain, v2 7-domain), **domain-controlled**, on
  domain-balanced captures — see §1b. (An earlier capture-sampling bug limited every
  capture to ~2 domains; fixed 2026-07-21 with domain-stratified round-robin
  sampling, then re-captured and re-run.)
- **Reproducibility (resolved):** the earlier run-to-run AUROC wobble came from an
  under-regularized (non-converged) 14k-feature fit. Fixed by stronger L2
  (`c_reg=0.1`), which converges the strictly-convex objective to its unique,
  thread-independent optimum. The firmed-up figures in §1a use `c_reg=0.1` and
  **supersede** the exploratory `c_reg=1.0` numbers in §2–§5; note proper
  regularization *raised* the lift (the old fit was overfit).

## Current status — findings, blocking, next steps (2026-07-21)

### Findings (validated)
- **Pre-round first-token acceptance signal beyond entropy AND domain, replicated
  3/3.** Domain-controlled baseline (`preround_hardened + domain`), on
  domain-balanced captures: `raw_frontier` Δauroc **+0.072 (Qwen-v1), +0.112
  (Qwen-v2, 7 domains), +0.069 (Llama)** — all CI-clean (p≤0=0). Two model families,
  two corpora. See §1b.
- **First-token only — does NOT extend to run-length.** Per-length survival probes
  (3/3): the lift is strong + clean at k=1, then **null-to-harmful for k≥4** (worse
  than entropy+domain at k=8 on v2/Llama, where domain already predicts long runs).
  The Tier-2 length *controller* does not benefit and mostly hurts. See §1c.
- **Near-zero deployed cost.** Signal = read cached frontier reps + a 16 µs probe =
  **0.015% of a decode round** (§G3); materialization ~free on the eager path.
- **Robust, not an artifact.** Survives 7-domain domain-control (not domain
  identification); beats equal-capacity random/norm controls (not capacity);
  Platt-calibrated (not a calibration artifact); reproducible at `c_reg=0.1`.
- **Only the full representation works.** The cheap/deployable variants (align,
  drift, norm, lowrank) show only small, inconsistent lift — the deployable-cheap
  signal is marginal.
- **Causal — replicated (v1 + Llama).** I15 forward-hook steering: perturbing the
  first-token acceptance direction disrupts acceptance **~2–10× more than
  norm-matched controls**, dose-dependently, **beyond entropy**, at all 4 layers on
  **both** Qwen-v1 (`sealed_fidelity` 0.95) and Llama (`sealed_fidelity` 1.00,
  controls ≈0) — a **replicated, representation-level causal** result (the direction
  causally controls the target's next-token agreement-ability; *not* a draft–target
  "circuit"). Both empirical G2 criteria met; the upgrade to causal language in
  frozen claims is a human gate (D020). Full write-up:
  `docs/causal_intervention_report.md`.

### Blocking (before submission)
- **Frozen test pass.** Everything is dev-only; **C10 is `UNTESTED`** in
  `docs/CLAIMS_LEDGER.md`. The primary table must be reproduced on the frozen test
  split.
- **Novelty re-check (I21).** Refresh the landscape scan and position explicitly vs
  AdaEAGLE / Judge Decoding / WhiFlash (and anything newer) before freezing "C10
  unoccupied".
- **Claims-ledger + manuscript.** Move C10 to a supported claim; build the two
  headline figures (I18) and the anonymous manuscript (I19); keep G2-gated language.
- **Uncommitted work.** The capture-sampling fix, `pair`/`data_dir`/`c_reg`/
  `domain_control` threading, length/Tier-2 + payoff, and this report are not yet
  committed.

### To do (next steps, highest leverage first)
1. **G2 interventions (I15) — v1 + Llama PASS (empirical G2 met).** Forward-hook
   steering disrupts acceptance ~2–10× norm-matched controls, dose-dependent, beyond
   entropy, at all 4 layers, **replicated on both model families**
   (`docs/causal_intervention_report.md`). Remaining: a **human trips G2** (D020) to
   use causal language in frozen claims; firm-up — finer layer sweep, a
   projection-removal (ablation) variant alongside steering, larger cap_test.
2. **Minimal cheap sufficient statistic.** Does a small low-dim subspace (a few dims
   / one layer) recover most of `raw`'s first-token lift? Turns a predictive result
   into a *deployable* one (today's cheap variants don't clear the bar).
3. **Frozen test pass** — converts dev → confirmed (evidence gate).
4. **Target-appropriate layer sweep.** Layers 6/12/18/24 were tuned for the 0.5B
   *draft* (24 layers); sweep the 28/32-layer *targets* to find where the signal
   peaks (supports G2 localization).
5. **Manuscript + novelty freeze** — I18 / I19 / I21, claims-ledger audit,
   test-split confirmation, then submit per the staged policy (D010).

## 1b. Validated transfer — domain-controlled, replicated (supersedes §1a)

The single-pair §1a numbers are superseded by a **domain-controlled** binary
analysis on **domain-balanced** captures across three model/corpus settings.
(Prerequisite fix: `cap_prompts` + sorted truncation had captured only ~2 domains
per run — re-captured 2026-07-21 with domain-stratified sampling.) The baseline is
now `preround_hardened + domain` (one-hot), so the test is *"beyond entropy AND
domain."* All numbers `c_reg=0.1`, dev, prompt-grouped OOF + bootstrap CI.

| pair | #dom | base | raw comb | raw Δ | p≤0 | align Δ | drift Δ |
|---|---|---|---|---|---|---|---|
| Qwen v1 | 4 | 0.670 | 0.742 | **+0.0722** | 0.000 | +0.0137 | −0.0126 |
| Qwen v2 | 7 | 0.618 | 0.731 | **+0.1124** | 0.000 | +0.0226 | +0.0139 |
| Llama v1 | 4 | 0.696 | 0.765 | **+0.0685** | 0.000 | +0.0237 | −0.0466 |

(v2 raw Δ 95% CI [+0.095, +0.130]; all three are CI-clean, p(Δ≤0)=0.)

**The core finding replicates cleanly and robustly.** `raw_frontier` beats
entropy + domain by **+0.07–0.11 AUROC**, all CI-clean, domain-controlled, across
two model families (Qwen v1 +0.072 ≈ Llama v1 +0.069) and two corpora (v1 +0.072,
v2 +0.112). It is the sole *consistently* credible signal on all three. Because it
survives proper domain coverage + domain control on the 7-domain v2, the pre-round
signal is **not domain identification** — the earlier 2-domain worry was a
capture-sampling artifact, now fixed.

**Caveats (honest):**
- The **cheap/weak feature families** (align, drift, norm, lowrank) show only small
  (+0.01–0.02) lift and flicker in/out of the `credible` set across runs — so only
  `raw` is robust, and the near-zero-cost deployable variant stays marginal.
- **v2's baseline is genuinely weaker** (0.618: acceptance is harder to predict from
  entropy+domain there), so v2's larger delta is partly a weaker base — though
  `raw`'s absolute AUROC (0.73) is in the same band as v1/Llama.
- Still dev-only; the domain-controlled **length/Tier-2**, the frozen **test** pass,
  a **novelty re-check** (I21), and (optional) **G2 interventions** remain.

**Verdict:** the replication gate is **met** — domain-controlled, cross-model AND
cross-corpus. With near-zero deployed cost (§G3 microbench) and calibration, this is
a defensible, original honest-diagnostic contribution: *the target's cached
pre-round representation predicts speculative-decoding acceptance beyond entropy and
domain (+0.07–0.11 AUROC), at near-zero deployed cost, replicated across two model
families and two corpora.*

## 1c. Length is first-token-only — replicated, domain-controlled

Per-accepted-length survival probes P(A≥k), `raw_frontier`, domain-controlled,
`c_reg=0.1`, n_boot=2000, all three settings (dev):

| k | Qwen-v1 Δ | Qwen-v2 Δ | Llama Δ |
|---|---|---|---|
| **1 (first-token)** | **+0.073** [+0.059,+0.088] | **+0.103** [+0.085,+0.121] | **+0.065** [+0.046,+0.084] |
| 2 | +0.008 (null) | +0.045 (clean) | −0.007 (null) |
| 4 | −0.010 (null) | −0.004 (null) | −0.019 (worse) |
| 6 | −0.002 (null) | −0.011 (null) | −0.032 (worse) |
| 8 | +0.002 (null) | −0.018 (worse) | −0.021 (worse) |

The signal is concentrated at **k=1 (first-token)** on all three; for run-length
(k≥4) it is **null-to-significantly-harmful** — the representation adds nothing
beyond entropy+domain for long runs (base AUROC climbs to ~0.78–0.81 there, since
domain determines who gets long runs) and degrades it on v2/Llama. The **Tier-2
length controller** consequently does not benefit — on v2 it mostly *increases*
regret (CI-robust help only at one cost, 0.5). So the deployable systems use, if
any, is a **first-token skip** decision, not length selection — cleanly
differentiated from AdaEAGLE (which sets length from features, exactly where our
signal fails).

## 1a. Firmed-up + length-aware update (c_reg=0.1, n_boot=2000)

Re-run of the top candidate `raw_frontier` with the regularized/converged
(reproducible) fit, plus per-length survival probes and a length-aware controller.
These figures supersede §2–§5's exploratory `c_reg=1.0` numbers.

**Predictive lift is stronger and reproducible.** With proper regularization the
binary (k=1) lift rises to **+0.0740 AUROC, 95% CI [+0.051, +0.099]** (was +0.047
under the overfit `c_reg=1.0`); equal-capacity random 14k control ≈ 0.53.

**Tier-1 — per-length survival lift P(A≥k)** (dev, n=5321):

| k | P(A≥k) | base | combined | Δauroc | 95% CI | robust |
|---|---|---|---|---|---|---|
| 1 | 0.843 | 0.698 | 0.772 | +0.0740 | [+0.051, +0.099] | yes |
| 2 | 0.743 | 0.690 | 0.729 | +0.0387 | [+0.018, +0.062] | yes |
| 4 | 0.607 | 0.712 | 0.724 | +0.0110 | [−0.008, +0.032] | no (lone dip) |
| 6 | 0.512 | 0.700 | 0.722 | +0.0217 | [+0.005, +0.040] | yes |
| 8 | 0.429 | 0.702 | 0.720 | +0.0181 | [+0.001, +0.037] | yes |

**Revised length story.** The lift is *strongest* at first-token (k=1) but is
**CI-clean at 4 of 5 thresholds** (only k=4 dips) — so it is **not purely
first-token**: a weaker but real (~+0.02) signal persists for run-length. This
supersedes the earlier "first-token, not length" reading; the k=4 dip is likely
multiple-comparisons noise.

**Tier-2 — length-controller throughput regret** (choose L∈{skip,1,2,4,6,8} from
predicted survival; realized-throughput regret vs a clairvoyant oracle; lower is
better):

| cost_draft | reg_base | reg_comb | Δregret | 95% CI | helps_ci |
|---|---|---|---|---|---|
| 0.20 | 0.352 | 0.349 | −0.0024 | [−0.010, +0.005] | no |
| 0.30 | 0.329 | 0.323 | −0.0051 | [−0.014, +0.003] | no |
| **0.50** | 0.246 | 0.238 | **−0.0087** | **[−0.015, −0.0025]** | **yes** |
| ≥1.0 | 0.000 | 0.000 | 0.000 | — | no (degenerate skip) |

The representation-driven length controller beats the scalar-baseline controller
**CI-robustly at one realistic cost (0.5)**, small magnitude (−0.009); the trend is
coherent (growing to cost 0.5) but a single robust point warrants the same
multiple-comparisons caution. At cost ≥ 1 the length decision degenerates to skip.

**Net:** the firmed-up picture is modestly *more* positive — a larger, reproducible
predictive lift (+0.074), a signal that is **not** purely first-token, and a weak
but CI-robust length-controller win at a realistic cost. Still modest, dev-only,
one-pair. The binary re-run at `c_reg=0.1` and the v2/Llama transfer remain pending.

## 1. What was measured

The load-bearing gap this closed: I10 captured only the **draft** residual stream
at proposal-generating positions (a within-round signal). The pre-round headline
(C10/I23) needs the **target** verified-context **frontier** representation — the
state that exists **before** a round drafts, available at ~zero marginal cost as a
byproduct of the previous verify. A new capture (`capture_frontier_activations`,
D023) produced it.

- **Capture:** teacher-forced target residual stream at layers 6/12/18/24 at the
  frontier (last-committed) position of every proposal-bearing round, labeled by
  that round's own realized acceptance. Run `sweep-2026-07-11T203836`, split
  `all`: **10,694 frontier rows × 4 layers** over 240 prompts (120 dev + 120 test).
- **Scoring:** each seed-library `FeatureSpec` is fit as **incremental lift over
  the frozen pre-round baseline** `preround_hardened` (recent-acceptance EMA +
  previous-round target frontier entropy/margin), under **prompt-grouped
  GroupKFold OOF**, with **equal-capacity controls** (norm-matched + random of the
  same dimensionality), a **prompt-grouped bootstrap CI** on the AUROC delta,
  **global Platt recalibration** of the decision metrics, and a **draft-cost-ratio
  sweep** of calibrated decision-regret with per-cost bootstrap CIs.

Dev fit: **n = 5,321 rows, positive (accept) rate = 0.8435.**

## 2. Predictive result (G1) — a real signal beyond entropy

Frozen baseline in this fit (dev, per-round): **base AUROC = 0.6979.** (The
`preround_hardened` figure of ~0.73 in the ledger was measured on test /
per-drafted-token; the honest comparison is the within-fit delta, not the
cross-setting number.)

| candidate | Δ AUROC | 95% CI | P(Δ≤0) | random control |
|---|---|---|---|---|
| **raw_frontier** (14336-d) | **+0.0465** | **[+0.0212, +0.0738]** | **0.0000** | 0.529 |
| lowrank_k16 (64-d) | +0.0120 | — | — | 0.685 |
| norm_summary (12-d) | +0.0059 | — | — | 0.692 |
| align (2-d) | −0.0005 | — | — | 0.695 |
| drift (14340-d) | −0.0579 | — | — | 0.511 |

- **The full target-frontier representation carries real, control-surviving
  predictive information about acceptance that pre-round entropy does not.**
  `raw_frontier` reaches combined AUROC 0.7444 (Δ +0.0465, CI excludes 0), while an
  **equal-capacity random 14336-d control collapses to 0.529** — the gap is signal,
  not capacity. Cross-run the delta ranges **+0.047 to +0.059** (both runs CI-clean).
- ⇒ **Entropy is not a sufficient statistic for pre-round acceptance.** This is the
  positive finding. The C10 cell (lossless, pre-round, baseline-controlled) was
  unoccupied in the landscape scan (I21), so this is differentiated from the
  generic "a probe predicts acceptance" that the differentiation requirements rule
  out.
- The **cheap / deployable variants add little**: `norm` +0.006, `lowrank` +0.012.

## 3. Calibration — the ECE blow-up was a fixable artifact

`raw_frontier`'s uncalibrated combined ECE was **0.1423** (≈10× the base's 0.017).
Global Platt recalibration (a single monotonic map on the OOF scores — AUROC
preserved exactly, leakage-safe per D019) restored it to **0.0164**, essentially
the base's 0.0173. So the ranking lift is genuine information, **not** overconfidence.

## 4. Systems result (G3 decision proxy) — real but narrow

Calibrated decision-regret sweep for `raw_frontier` across draft cost (regret is
the coarse binary skip/draft proxy; **wall-clock is authoritative, G3**). Δregret =
combined − base; negative = helps.

| cost_draft | tau | Δregret | 95% CI | P(Δ≥0) | CI-robust |
|---|---|---|---|---|---|
| 0.05–0.20 | ≤0.17 | 0.0000 | [0,0] | 1.000 | no |
| 0.30 | 0.231 | +0.0003 | [−0.0005,+0.0016] | 0.671 | no |
| 0.50 | 0.333 | −0.0006 | [−0.0025,+0.0018] | 0.284 | no |
| **1.00** | 0.500 | **−0.0094** | **[−0.0151,−0.0041]** | 0.000 | **yes** |
| **2.00** | 0.667 | **−0.0237** | **[−0.0403,−0.0084]** | 0.003 | **yes** |
| **4.00** | 0.800 | **−0.0545** | **[−0.0899,−0.0197]** | 0.001 | **yes** |
| 9.00 | 0.900 | −0.0402 | [−0.0912,+0.0074] | 0.048 | no |

- **At cheap draft cost (≤0.2 — the idealized-serving regime): no help.** Every
  calibrated model defaults to "always draft" (regret saturates at
  `cost_draft × reject_rate`), so no pre-round signal moves a decision.
- **At cost_draft ≥ 1 (tau ≥ 0.5): `raw_frontier` gives a CI-robust regret
  reduction** — ~6% relative at cost 1.0, growing to cost 4.0. This is the
  credible systems signal. The launch-bound eager harness sits near cost_draft ≈ 1
  (draft ≈ verify), so *in that regime* the reduction is real.

## 5. The controls / CIs did their job (and one caution)

CI-robust regret help (`helps_ci`) by candidate:

| candidate | Δ AUROC | CI-robust help at cost_draft |
|---|---|---|
| raw_frontier | +0.047 | **1.0, 2.0, 4.0** (contiguous) |
| lowrank_k16 | +0.012 | 4.0 only |
| norm_summary | +0.006 | 4.0 only |
| align | −0.001 | **NONE** |
| drift | −0.058 | 1.0 only |

- **`align` (no ranking lift) → zero CI-robust help.** The CI correctly demolished
  its point-estimate "helps" (which fired at 4 costs). Validation that the machinery
  is not fooled by noise.
- **Caution — `drift` (negative AUROC) is CI-robust at a single cost (1.0).** This
  exposes a real subtlety: **single-threshold regret is not monotonic in AUROC** —
  a globally-worse ranker can win at one operating point. The credible bar is
  therefore **real AUROC lift (CI-clean) AND CI-robust over a cost *range***, which
  **only `raw_frontier` meets**. (The `beats_controls` flag alone is also
  insufficient: `drift` has `beats_controls=True` yet sits below base — read it only
  together with `beats_baseline`.)

## 6. Threats to validity (open, must resolve before any claim)

1. **Wall-clock not yet measured (the real G3).** §4 is the decision *proxy* on the
   *expensive* signal (`raw` = full residual stream). The decisive test charges the
   feature's own extraction+probe cost inside deployed timing. Not yet run.
2. **AUROC magnitude not reproducible.** `raw`'s delta was +0.0592 (run 1) vs
   +0.0465 (run 3) — same data, seed, folds. Cause: a 14336-feature `lbfgs`
   logistic that does not converge, so BLAS threading changes the result. Fix:
   stronger regularization / single-threaded BLAS / dimensionality reduction.
3. **Coarse binary label.** Regret uses `accept` (≥1 accepted), not expected
   accepted **length** vs cost. A length-aware payoff could differ.
4. **Layer indices** (6/12/18/24) were chosen for the 24-layer *draft*; on the
   28-layer *target* these are early/mid layers (24 ≠ final). A target-tuned layer
   sweep is unexplored.
5. **Single pair, dev only.** No transfer (v2 / Llama) and no frozen test pass.

## 7. Decisive vs open

| Question | Status |
|---|---|
| Real pre-round signal beyond entropy (dev) | **Decisive positive** (control-clean, CI-clean, not a calibration artifact) |
| Its magnitude | Not reproducible yet (fix the fit) |
| Reduces decision *proxy* regret | **CI-robust** for `raw` at cost_draft ≥ 1 (expensive-draft regime); none at cheap-draft |
| Net wall-clock win (G3) | **Untested** — the decisive systems experiment |
| Replicates (v2 / Llama / test) | Untested; **C10 `UNTESTED`** |

**Bottom line:** a **positive scientific result** (a real, differentiated pre-round
diagnostic signal beyond entropy) plus a **narrow, not-yet-deployable positive
systems signal** (CI-robust regret reduction confined to the expensive-draft regime
and to the full-representation variant). Even the worst case is a publishable
honest-diagnostic result with a quantified mechanism–systems trade-off
(RESEARCH_SPEC: beating the controller baseline is not required for an honest
diagnostic paper).

## 8. Reproduce

Artifacts are immutable; every number above is script-generated. Code at the
committing revision. Frozen bar and controls are in `cas.autoresearch.eval`.

```bash
# 1. capture the target-frontier representation (GPU; dev+test in one artifact)
modal run --detach modal_app.py::capture_frontier \
  --run-id sweep-2026-07-11T203836 --split all
# 2. score the seed library on dev (CPU): incremental lift + controls + CI +
#    recalibration + cost-ratio sweep
modal run --detach modal_app.py::autoresearch \
  --run-id sweep-2026-07-11T203836 --eval-split dev
# 3. read the leaderboard / cost-sweep from the committed JSON
modal run modal_app.py::autoresearch_show \
  --run-id sweep-2026-07-11T203836 --eval-split dev
```

Output: `/artifacts/analysis/sweep-2026-07-11T203836/autoresearch_dev.json`.
Orchestration (Workflow scriptPath): `src/cas/autoresearch/generator_critic.js`.

## 9. Next steps (priority order)

1. **Wall-clock G3 test** — charge `raw`'s extraction+probe cost per round; does the
   §4 reduction survive real latency?
2. **Stabilize the fit** so the AUROC magnitude is reproducible.
3. **Stricter `credible_systems` flag** (AUROC-CI-clean **and** CI-robust over a
   cost range) so automated runs are not fooled by the `drift`-type single-point
   anomaly.
4. **Transfer** (v2 / Llama) and length-aware regret.
5. **Log to `CLAIMS_LEDGER`** as dev-only / tentative; C10 stays `UNTESTED`.
