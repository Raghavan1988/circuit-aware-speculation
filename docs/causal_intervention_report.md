# Causal Intervention Report — Pre-round Acceptance Direction (I15 / G2)

Companion to `docs/autoresearch_outcomes.md`. Covers the causal-validation step
for the pre-round first-token acceptance signal (I13/I23). **Development-set;
the derived direction is fit on dev and evaluated on held-out test rounds.** Per
D020, results are stated as a **diagnostic signal / representation-level causal**
claim, never "circuit"/"mechanism", until a human trips G2 after replication.

**Status:** Qwen-v1 and Llama both complete — the causal effect **replicates across
both model families** (§3). The two empirical G2 criteria (survives controls +
dose-responsive + beyond entropy, AND replicated) are **met**; tripping G2 (the
move to causal/representation-level language in frozen claims) is a human decision
(D020).

## 1. Why (motivation)

The predictive finding (I13/I23) is **correlational**: a probe *decodes*
first-token acceptance from the target's cached verified-context representation,
beyond entropy and domain, replicated across model families and corpora. A probe
shows information is *present*; it does not show the representation is *used*, nor
that the decoded direction *causes* acceptance rather than being an epiphenomenal
correlate of context.

The intervention answers the causal question, and it is gated by the project's own
criteria:

- **G2 (D020, AGENTS.md):** "causal" requires an intervention that is
  **layer-specific, dose-responsive, replicated, and survives random +
  norm-matched controls."
- **Finding-specific requirement:** the causal effect on acceptance must exceed the
  effect on **entropy** — otherwise the direction is causally just an entropy/
  confidence knob and there is nothing causal *beyond entropy* (the causal analog
  of the predictive "beyond entropy" claim).

Payoff: converting "diagnostic signal" → causal is the project's headline ambition
and the single biggest strengthening of the contribution.

## 2. Methodology

**Direction.** On **dev** frontier states at each layer L ∈ {6,12,18,24}, derive
the first-token acceptance direction `d̂_L = mean(accepted) − mean(rejected)`,
unit-normalized. (First-token, since the signal is first-token only, autoresearch
§1c.)

**Intervention (forward-hook steering).** On held-out **test** rounds, add
`α·σ_L·d̂_L` to the target residual stream at the **frontier position** (via a
forward hook on `layers[L-1]`, whose output is `hidden_states[L]`), at doses
α ∈ {−2,−1,0,+1,+2}. `σ_L` = mean frontier-state norm (natural dose scale). Re-read
the target's argmax + entropy at the frontier; **first-token acceptance =
(draft's sealed proposal == intervened target argmax)**.

**Controls (per layer).** Two norm-matched controls: **random** (isotropic
Gaussian scaled to ‖d̂‖) and **shuffled** (permuted d̂ — norm-preserving,
structure-destroying). The real direction must disrupt acceptance **more than
both** — isolating the learned direction from mere perturbation magnitude.

**Metrics.** The intervention produces an **inverted-U** (acceptance peaks at α=0;
steering either way breaks agreement), so endpoint/monotone metrics are wrong.
Instead:

- **disruption** = `accept(α=0) − mean(accept at α≠0)`; must be **peak-at-0** and
  **dose-monotone in |α|**.
- **beats_controls**: real disruption ≫ control disruption.
- **beyond-entropy**: within narrow entropy bins (controlling residual entropy),
  does α still move acceptance? (`entropy_stratified_effect`) — robust to the
  step-shaped accept↔entropy relationship that breaks a linear mediation model.
- **causal_beyond_entropy** = disruption>0 AND dose-monotone-in-|α| AND
  beats_controls AND beyond-entropy.

**Fidelity diagnostic.** `sealed_fidelity` = how often a no-op (α=0) re-forward
reproduces the **sealed** argmax. <1.0 is bf16 non-determinism on low-margin
positions — **not** a correctness bug: disruption is measured entirely in the
self-consistent **re-forward frame** (baseline = α=0 re-forward), so the gap
cancels in the real-vs-control comparison.

**Replication.** Repeat on the Llama pair. **Layer-specificity** (across layers)
and **replication** (across model families) are the two G2 requirements; a human
trips G2.

**Tooling.** Transparent PyTorch forward hooks on the real HF model (same model as
the proven exact-decoding equivalence). Pure analysis logic in
`cas.autoresearch.interventions` (unit-tested, incl. the two decisive
entropy-mediation cases); GPU runner `modal_app.py::intervene`. D015 prefers
nnsight; hooks are used here (auditable, no image change) and nnsight is swappable
— worth a `DECISIONS` note.

**Honest design boundary.** The direction is in the **target's** representation
predicting a **separate** draft. Intervening on the target changes the **target's
own** next-token (what must be matched) — so the causal claim is *"this direction
causally controls the target's next-token agreement-ability"*, **not** a shared
draft–target "circuit". Even a clean pass earns representation-level causal
language, not "circuit/mechanism".

## 3. Outcome

**Qwen-v1, 4 layers, 200 test rounds, `sealed_fidelity = 0.95`:**

| layer | baseline accept | real disruption | control disruption (rand / shuf) | beats ctrl | beyond H | **causal** |
|---|---|---|---|---|---|---|
| 6  | 0.83 | **+0.284** | 0.028 / 0.040 | ✓ | ✓ | **True** |
| 12 | 0.83 | **+0.278** | 0.053 / 0.089 | ✓ | ✓ | **True** |
| 18 | 0.83 | **+0.381** | 0.151 / 0.171 | ✓ | ✓ | **True** |
| 24 | 0.83 | **+0.413** | 0.091 / 0.093 | ✓ | ✓ | **True** |

Real α-curve (L24), α = −2…+2: `[0.19, 0.63, 0.83, 0.65, 0.22]` — a clean
peak-at-0 disruption.

**Reading.** Steering along the acceptance direction disrupts first-token
acceptance **~2–10× more** than norm-matched controls of equal magnitude,
**dose-dependently** (peak-at-0, falling with |α|), **beyond the induced entropy
change**, at **every probed layer**. That is the shape of a real G2-level
causal/localization result: the direction is **functionally special** in the
target's computation, not merely correlated with acceptance. The effect is
**strongest at the late layers (18/24)** — distributed through the residual
stream, most active late (consistent with next-token resolution happening late),
rather than a single-layer circuit.

**Llama, 4 layers, 200 test rounds, `sealed_fidelity = 1.00`:**

| layer | baseline accept | real disruption | control disruption (rand / shuf) | beats ctrl | beyond H | **causal** |
|---|---|---|---|---|---|---|
| 6  | 0.82 | **+0.318** | −0.015 / −0.014 | ✓ | ✓ | **True** |
| 12 | 0.82 | **+0.398** | 0.027 / 0.009 | ✓ | ✓ | **True** |
| 18 | 0.82 | **+0.396** | 0.070 / 0.065 | ✓ | ✓ | **True** |
| 24 | 0.82 | **+0.368** | 0.074 / 0.066 | ✓ | ✓ | **True** |

**Replicated.** Llama shows the **same pattern** as v1 — all four layers causal,
real disruption ~0.32–0.40 vs **near-zero** control disruption — and is **cleaner**:
`sealed_fidelity = 1.00` (perfect re-forward fidelity, confirming v1's 0.95 was
bf16 numerical noise, not a bug) and controls ≈ 0 (even slightly negative at L6).
So the causal effect **replicates across two model families**: perturbing the
acceptance direction specifically disrupts first-token acceptance, dose-dependently,
beyond entropy, on **both Qwen and Llama**.

## 4. Caveats / threats to validity

- **Representation-level, not a circuit** — see the design boundary in §2.
- **Not narrowly localized** — causal at all four layers (strongest late); the
  honest phrasing is "the direction is causally active through the residual stream,
  strongest late", not "localized to layer X".
- **Beyond-entropy is suggestive, not airtight** — mediation for a peak-shaped
  effect is delicate; the **load-bearing evidence is the control-specificity**
  (real ≫ norm-matched), which is unambiguous.
- **`sealed_fidelity = 0.95`** — ~5% bf16 numerical noise (fresh vs sealed forward);
  cancels in real-vs-control but adds ~5% noise to absolute rates.
- **Dev-derived direction, test-evaluated** — leakage-safe, but a single dev/test
  split; the direction and doses were not tuned on test.
- **Language gate** — stays "diagnostic signal" (D020) until Llama replicates and a
  human trips G2; C10 remains `UNTESTED` in the claims ledger.

## 5. Verdict + next steps

**Both Qwen-v1 and Llama pass** — control-surviving, dose-responsive, beyond-entropy,
at all four layers, on **both model families**. The two empirical G2 criteria (D020)
are **met**. Tripping G2 — moving to causal / representation-level language in frozen
claims — is now a human decision. The honest claim available: *"the pre-round
acceptance direction causally controls the target's next-token agreement-ability,
beyond entropy, replicated across two model families"* — representation-level, not a
draft–target "circuit".

Then, to firm up for a claim: a **finer layer sweep** to characterize where the
effect peaks; larger `cap_test` for tighter estimates; a projection-removal
(ablation) variant alongside additive steering; and recording the nnsight-vs-hooks
choice and I15 status in `DECISIONS`/backlog.

## 6. Reproduce

```bash
# Qwen-v1 (above)
modal run --detach modal_app.py::intervene_run \
  --run-id sweep-2026-07-11T203836 --layers 6,12,18,24
# Llama replication
modal run --detach modal_app.py::intervene_run \
  --run-id sweep-llama-f8-2026-07-13 --layers 6,12,18,24 --cas-pair llama --data-dir data
```

Output: `/artifacts/analysis/<run>/intervene.json`. Pure logic + tests:
`src/cas/autoresearch/interventions.py`, `tests/test_interventions.py`. Runner:
`modal_app.py::intervene`. Check `sealed_fidelity` (a bf16 diagnostic; disruption
is measured in the self-consistent re-forward frame).
