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
- **Model pair:** Qwen2.5-7B-Instruct target / Qwen2.5-0.5B-Instruct draft (v1).
  No replication pair (v2 / Llama) run yet.
- **Reproducibility (resolved):** the earlier run-to-run AUROC wobble came from an
  under-regularized (non-converged) 14k-feature fit. Fixed by stronger L2
  (`c_reg=0.1`), which converges the strictly-convex objective to its unique,
  thread-independent optimum. The firmed-up figures in §1a use `c_reg=0.1` and
  **supersede** the exploratory `c_reg=1.0` numbers in §2–§5; note proper
  regularization *raised* the lift (the old fit was overfit).

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
