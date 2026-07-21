# Generator–Critic Autoresearch Plan

Execution method for the *signal-discovery* portion of this project: a
generator–critic autoresearch loop that searches for an original, gate-clearing
internal signal of draft–target acceptance.

- **Scope:** this is a *method* for running issues **I13** (incremental
  information) and **I23** (pre-round prediction from cached representations),
  feeding tripwires into **I15** (interventions / G2) and **I14** (controller /
  G3). It introduces **no new scientific scope** — it uses the locked contract,
  the locked corpus (D022), the locked tooling (D015, D021), and the
  measurement-first ordering (D018). No `DECISIONS.md` entry is required unless
  it changes an interface or adds an artifact family.
- **Authority:** subordinate to `AGENTS.md` + `docs/` (scientific law),
  the gates in `RESEARCH_SPEC.md`, and `docs/CLAIMS_LEDGER.md`. Where this file
  and the contract disagree, the contract wins.
- **Naming (D008):** "generator–critic autoresearch" and "eval-gated
  hill-climbing" describe a public methodology attributed to an individual, not a
  venue, organization, lab, or owner goal. If this file is ever bundled into an
  anonymized release, genericize the attribution line below.

Lineage: this follows the Karpathy framing of autoresearch — an LLM does not
*have* the research idea; it runs a fast, parallel search against an
un-gameable evaluation, and the "discovery" is a verified point in a search
space. Taste, thesis, and gate decisions stay human.

---

## 1. The core bet

The parts of this problem that are cheapest to automate — probe sweeps, the
acceptance atlas, a bandit length controller — are exactly the parts
`RESEARCH_SPEC.md` § *Differentiation requirements* forbids as a contribution.
So a naive sweep produces non-novel components. This loop is pointed instead at
the one region where a *searchable* original contribution can still live:

> **A new, near-zero-cost, pre-round signal of acceptance that (a) adds
> predictive information over the full surface-baseline stack, (b) has a
> characterized failure geometry, and (c) is cheap enough that a controller
> built on it can net a real wall-clock win.**

The invention is the *signal itself* (a function of already-cached
verified-context representations) plus *the map of where it breaks* — not a
framing sentence. That is what a search can find and a critic can verify.

---

## 2. The eval is the moat

Autoresearch quality equals eval quality. A gameable eval turns a search into a
generator of plausible-but-wrong findings. This project's contract already *is*
the un-gameable eval; the loop is frozen against it **before** any search runs.

**Predictive eval (primary, → G1, I13/I23).** Incremental lift of a candidate
signal over the frozen surface-baseline stack
`{target frontier entropy, top-1/top-2 margin, recent acceptance, token
category, domain, generation phase, position}`:

- Metrics: AUROC, AUPRC, Brier, ECE, and decision-regret (D018).
- Targets: next-round acceptance **and** accepted-run-length bins
  (survival/hazard formulation, D018/D019).
- Splits: **prompt-grouped, out-of-fold** only. Token-level random splits are
  prohibited (AGENTS.md). Calibration folds keep prompt groups intact and use
  out-of-fold base scores (D019).
- Labels: same-round counterfactual full-information labels — a max-length round
  labels all shorter actions up to the first rejection; terminal/capped rows are
  excluded from nominal-yield fitting; never rolled into multi-round trajectories
  (D018/D019).
- Fit **only** on development data; freeze before the test pass.

**Deployment eval (→ G3, I14).** A pre-round signal earns its name only if it is
near-free. Measure the candidate's *marginal wall-clock cost* on the compiled
timing path (D021), separately from its offline predictive value (I23 criterion).
A signal that predicts well but costs a draft-forward-pass to compute has failed
its own premise.

**The eval is frozen and versioned before the loop starts. The generator never
edits the eval.**

---

## 3. What the loop may invent (the search space)

Ranked; the first is primary.

- **A. New pre-round signal class (primary, I23).** A feature computed from
  cached verified-context representations *before* any draft compute. Seeds:
  - *draft–target representational alignment* — recast "will the draft agree"
    as a geometric alignment between the two models' residual streams at the
    current verified context (both share the tokenizer);
  - *divergence velocity / curvature* — the rate at which the residual stream is
    drifting across recent positions (a dynamical precursor, orthogonal to the
    static entropy/margin baseline);
  - *induction / copy-head activity* — the strongest "circuit" lead; the atlas
    already flags repeated/copied spans, and induction heads are a named circuit
    (language stays G2-gated, see §6);
  - *surprise budget* — cumulative divergence since the last resync;
  - *minimal low-rank projections* of a selected layer's frontier state.
- **B. Failure-geometry cartography (the "explain when it fails" clause).** The
  structure of transfer/collapse across domain × token-category × generation
  phase × model-pair. A characterized failure law is itself an original finding.
- **C. Minimal sufficient statistic.** The smallest, cheapest, transferable
  representation that still suffices for the pre-round decision — a reusable
  artifact (touches G4).
- **D. Mechanism→action coupling (I14).** Whether the *magnitude* of a surviving
  signal maps directly to the optimal proposal length, collapsing predict-and-
  decide into one cheap op — the intended contribution per D002.

**Off-limits to the loop** (human/taste/gate territory): reframing the thesis;
tripping any gate; upgrading correlational language to mechanistic/"circuit"
language (D020); selecting anything on test latency or test labels.

---

## 4. Roles

- **Generator.** Proposes candidate signals as *typed, executable feature specs*
  — each a pure function of `(cached verified-context representations, trace
  fields)` returning a per-round feature vector — accompanied by (i) a
  falsifiable hypothesis, (ii) an a-priori cost class (near-zero / cheap /
  draft-priced), and (iii) which seed family (§3.A) it extends or how it departs.
  It reads the running candidate ledger to avoid re-proposing killed signals.
- **Critic (adversarial verifier).** Its job is to *kill* each survivor. A
  candidate is refuted unless it clears **all** of:
  1. incremental lift over the full baseline stack survives prompt-grouped OOF;
  2. lift is not explained by any single baseline covariate (entropy, margin,
     position, category, domain) via ablation;
  3. it is not label leakage (feature must be available *before* the action;
     canonical same-round outcome fields are rejected, D019);
  4. it beats a **norm-matched and a random control feature of equal
     dimensionality** (not just the zero-feature baseline);
  5. the lift replicates on a held-out domain **and** the second model pair (I17)
     or an equivalently strong transfer setting.
  Use k independent refuters with distinct lenses (leakage, capacity, transfer);
  majority-refute kills. Default to refuted under uncertainty.
- **Referee (human + script).** Trips gates. The loop never claims — it ranks
  and hands evidence up. Only a human moves a survivor from "candidate" to a
  ledger claim, and only a human trips G1/G2/G3.

---

## 5. The loop

Round structure (pipeline, no barrier except the dedup step):

1. **Generate** N candidate feature specs (generator, seeded by §3.A and the
   ledger of what's been killed).
2. **Fit** each pre-round on development OOF folds; compute incremental lift over
   the frozen baseline stack (§2).
3. **Dedup** against all previously *seen* candidates (not just confirmed ones —
   dedup vs. seen so critic-rejected signals don't reappear every round).
4. **Critic** — k adversarial refuters per fresh survivor (§4); majority-refute
   kills.
5. **Characterize** survivors: failure-geometry sweep (domain × category × phase
   × model-pair) + deployed-cost measurement on the compiled path (§2).
6. **Record** every candidate (survived *and* killed) to the candidate ledger;
   promote nothing automatically.
7. **Loop-until-dry:** stop after K consecutive rounds produce no fresh survivor,
   or on token/compute budget. Simple "run M rounds" caps miss the tail.

All agents operate on **saved I07 traces / I10 activations / I12 probe
artifacts** — they never re-run the decode engine (raw artifacts stay immutable,
AGENTS.md). Capture is already offline/eager (D015); timing uses the compiled
path (D021).

---

## 6. Honesty firewall (anti-gaming)

Non-negotiable; these are guardrails *on* the loop, not tasks *for* it:

- Eval frozen and versioned before search; generator cannot edit it.
- Prompt-grouped OOF everywhere; no token-level splits.
- Controls must have equal capacity (norm-matched + random, matched
  dimensionality) — a lift over "nothing" is not a lift.
- Counterfactual labels are same-round only; terminal rows excluded.
- No selection on test latency or test labels, ever.
- Every reported number is script-generated from immutable artifacts; nothing is
  hand-typed into tables/figures.
- Negative results and killed candidates are logged to `docs/CLAIMS_LEDGER.md`;
  the pre-registered negative outcome for the incremental study is publishable
  (D018).
- **Mechanistic language stays gated (D020, G2).** A predictive survivor is a
  "diagnostic signal" / "representation," never a "circuit" or "mechanism,"
  until interventions (I15) are layer-specific, dose-responsive, replicated, and
  survive norm-matched controls.
- A survivor is a *candidate*, not a claim, until it clears the critic **and**
  replicates across a model pair / transfer setting.

---

## 7. Gate mapping and tripwires

- **G1 (predictive validity, I13/I23):** the loop's primary product — a ranked,
  critic-survived set of pre-round signals with incremental lift and failure
  maps, or a stable replicated negative result. Either outcome is publishable
  under the measurement-first plan (D018).
- **G2 tripwire (I15):** fires only if a survivor is *localized* (concentrated at
  specific layers with a clean, low-dimensional direction). Interventions then
  test causality; only then may "circuit/mechanism" language appear (D020).
- **G3 tripwire (I14):** fires only if a survivor's deployed marginal cost (§2)
  is low enough that the compiled-path controller nets a wall-clock win after
  *all* overhead. Acceptance-rate improvement alone is insufficient (G3).
- Consistent with D018: G2/G3 are exploratory ride-along bets; the measurement
  backbone (the atlas, the incremental study, the failure map) is the ≥80%-odds
  deliverable and does not depend on a positive signal being found.

---

## 8. Workflow sketch

> Runnable orchestrator: `src/cas/autoresearch/generator_critic.js` — invoke via
> the Workflow tool's `scriptPath`. It lives beside the `cas.autoresearch` package
> it drives (not under `.claude/`), so the whole loop is tracked as part of the
> artifact (D024).

Run as a background `Workflow` over saved artifacts (illustrative skeleton;
actual feature specs and artifact paths resolved at run time):

```javascript
export const meta = {
  name: 'generator-critic',
  description: 'Search for a gate-clearing pre-round acceptance signal (I13/I23)',
  phases: [
    { title: 'Generate' },
    { title: 'Fit+Eval' },
    { title: 'Critic' },
    { title: 'Characterize' },
  ],
}

const seen = new Set(), survived = []
let dry = 0
while (dry < 2 && (!budget.total || budget.remaining() > 80_000)) {
  // 1. Generate candidate feature specs (seeded by §3.A + the killed ledger)
  const cands = await agent(GEN_PROMPT(Array.from(seen)),
    { phase: 'Generate', schema: FEATURE_SPEC_LIST })

  // 2+3. Fit pre-round on dev OOF, incremental lift over baseline; dedup vs seen
  const fresh = (await parallel(cands.specs.map(s => () =>
    agent(FIT_PROMPT(s), { phase: 'Fit+Eval', schema: LIFT_RESULT }))))
    .filter(Boolean).filter(r => !seen.has(r.key))
  if (!fresh.length) { dry++; continue }
  dry = 0; fresh.forEach(r => seen.add(r.key))

  // 4. Adversarial critic — k refuters/lens; majority-refute kills
  const judged = await parallel(fresh.map(r => () =>
    parallel(['leakage', 'capacity', 'transfer'].map(lens => () =>
      agent(REFUTE_PROMPT(r, lens), { phase: 'Critic', schema: VERDICT })))
      .then(v => ({ r, keep: v.filter(Boolean).filter(x => !x.refuted).length >= 2 }))))
  const kept = judged.filter(j => j.keep).map(j => j.r)

  // 5. Failure-geometry + deployed-cost characterization for survivors
  const characterized = await parallel(kept.map(r => () =>
    agent(CHARACTERIZE_PROMPT(r), { phase: 'Characterize', schema: FAILURE_MAP })))
  survived.push(...characterized.filter(Boolean))
}
return { survived }   // candidates, NOT claims — a human trips the gate
```

Dedup vs. `seen`, not `survived`, so critic-rejected signals do not reappear each
round (else the loop never converges).

---

## 9. Integration and pre-flight

- Before running: claim **I13** and/or **I23** in `docs/ISSUE_BACKLOG.md` with
  owner + `IN_PROGRESS` (AGENTS.md pre-flight). Both are currently `OPEN` with no
  owner; check for concurrent ownership first.
- Outputs update ledger cells **C10** (pre-round prediction) and the incremental-
  information claim family; killed candidates and the negative outcome are logged
  too.
- Reproducible command, artifact IDs, and code revision accompany every produced
  number (AGENTS.md definition of done).
- No public or manuscript name uses "circuit"/"mechanism" for a loop survivor
  before G2 (D020).

---

## 10. What this loop cannot do

It cannot decide that any survivor *is* the contribution — that is the taste call,
and it stays with the owner. The loop supplies verified evidence at high volume;
the thesis, the gate trips, and the mechanistic wording remain human.
