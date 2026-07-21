/*
 * generator_critic.js — Workflow orchestration for the generator-critic
 * autoresearch loop (D023; docs/generator_critic.md §5/§6/§8).
 *
 * WHAT THIS IS
 *   The deterministic orchestration for eval-gated hill-climbing over candidate
 *   PRE-ROUND acceptance signals for issues I13 (incremental information) and
 *   I23 (pre-round prediction from cached representations). A generator proposes
 *   typed FeatureSpecs from a fixed seed library; a fit CLI scores each as
 *   incremental lift over a frozen baseline; adversarial critics try to kill the
 *   survivors; a characterizer maps where each surviving signal transfers/breaks.
 *
 *   This file is AUTHORED FOR LATER INVOCATION as a background Workflow. It is
 *   not a standalone Node program: `export const meta`, top-level `await`, and a
 *   top-level `return` are part of the Workflow scripting DSL (the runtime wraps
 *   the body in an async function). Running `node --check` on it will therefore
 *   report a syntax error on `export`/`return` — that is expected, not a defect.
 *
 * ARTIFACTS (agents never re-run the decode engine — AGENTS.md immutability)
 *   All scoring reads SAVED artifacts through the fit CLI: I07 traces,
 *   I10/D023 target-frontier activations (/artifacts/probes/<run_id>/frontier/),
 *   and I12/frontier probe artifacts. The generator emits specs; the Fit+Eval
 *   agent shells out to scripts/fit_autoresearch.py, which resolves paths and
 *   returns metrics. No arbitrary code executes inside the loop (D023.3).
 *
 * HONESTY FIREWALL (guardrails ON the loop, not tasks FOR it — §6)
 *   - The eval is frozen and versioned BEFORE the search; the loop never edits it.
 *   - Scoring is prompt-grouped, out-of-fold (OOF) only — no token-level splits.
 *   - Every candidate must beat NORM-MATCHED and RANDOM controls of equal
 *     dimensionality, not just the zero-feature baseline; the frozen bar is the
 *     `preround_hardened` baseline (~0.73 AUROC).
 *   - Selection happens on DEV only; the test split stays frozen. No selection on
 *     test latency or test labels, ever.
 *   - Critics default to REFUTED under uncertainty; a survivor is a CANDIDATE,
 *     not a claim — a human trips G1/G2/G3.
 *   - Mechanistic / "circuit" language stays G2-gated (D020, RESEARCH_SPEC G2):
 *     a predictive survivor is a "diagnostic signal" / "representation", never a
 *     "circuit" or "mechanism", until interventions (I15) pass. No prompt or
 *     output in this file upgrades correlational language to mechanistic.
 */

export const meta = {
  name: 'generator-critic',
  description: 'Search for a gate-clearing pre-round acceptance signal (I13/I23) as ranked CANDIDATES, not claims.',
  phases: [
    { title: 'Generate' },
    { title: 'Fit+Eval' },
    { title: 'Critic' },
    { title: 'Characterize' },
  ],
}

// --------------------------------------------------------------------------
// Schemas (plain JS objects; JSON-Schema shape). agent(prompt, { schema })
// returns the validated object.
// --------------------------------------------------------------------------

// One candidate feature spec from the fixed seed library (D023.3): a typed,
// executable descriptor — NOT free-form code.
const FEATURE_SPEC = {
  type: 'object',
  additionalProperties: false,
  properties: {
    name: { type: 'string' },
    family: { type: 'string', enum: ['raw', 'lowrank', 'drift', 'align', 'norm'] },
    layers: { type: 'array', items: { type: 'integer' } },
    params: { type: 'object' },
    hypothesis: { type: 'string' },
    cost_class: { type: 'string', enum: ['near-zero', 'cheap', 'draft-priced'] },
  },
  required: ['name', 'family', 'layers', 'params', 'hypothesis', 'cost_class'],
}

// Generator output: a list of candidate FeatureSpecs.
const FEATURE_SPEC_LIST = {
  type: 'object',
  additionalProperties: false,
  properties: {
    specs: { type: 'array', items: FEATURE_SPEC },
  },
  required: ['specs'],
}

// A block of predictive metrics (reused for base / combined / controls).
const METRIC_BLOCK = {
  type: 'object',
  properties: {
    auroc: { type: 'number' },
    auprc: { type: 'number' },
    brier: { type: 'number' },
    ece: { type: 'number' },
    regret: { type: 'number' },
  },
  required: ['auroc', 'auprc', 'brier', 'ece', 'regret'],
}

// Incremental-lift deltas of combined-over-base.
const DELTA_BLOCK = {
  type: 'object',
  properties: {
    auroc: { type: 'number' },
    auprc: { type: 'number' },
    brier: { type: 'number' },
    ece: { type: 'number' },
    regret: { type: 'number' },
  },
  required: ['auroc'],
}

// Fit+Eval output: the parsed JSON printed/saved by scripts/fit_autoresearch.py.
const LIFT_RESULT = {
  type: 'object',
  properties: {
    name: { type: 'string' },
    base: METRIC_BLOCK,
    combined: METRIC_BLOCK,
    control_random: METRIC_BLOCK,
    control_norm: METRIC_BLOCK,
    deltas: DELTA_BLOCK,
    beats_baseline: { type: 'boolean' },
    beats_controls: { type: 'boolean' },
    delta_auroc_ci: { type: 'array', items: { type: 'number' } },
    n: { type: 'integer' },
    pos_rate: { type: 'number' },
  },
  required: [
    'name', 'base', 'combined', 'control_random', 'control_norm', 'deltas',
    'beats_baseline', 'beats_controls', 'delta_auroc_ci', 'n', 'pos_rate',
  ],
}

// One adversarial refuter's verdict. Default refuted=true under uncertainty.
const VERDICT = {
  type: 'object',
  properties: {
    refuted: { type: 'boolean' },
    reason: { type: 'string' },
    lens: { type: 'string', enum: ['leakage', 'capacity', 'transfer'] },
  },
  required: ['refuted', 'reason', 'lens'],
}

// Characterization of a survivor: where the signal transfers/breaks + cost.
const FAILURE_MAP = {
  type: 'object',
  properties: {
    by_domain: { type: 'object' },
    by_category: { type: 'object' },
    by_phase: { type: 'object' },
    by_model_pair: { type: 'object' },
    deployed_cost_note: { type: 'string' },
    summary: { type: 'string' },
  },
  required: [
    'by_domain', 'by_category', 'by_phase', 'by_model_pair',
    'deployed_cost_note', 'summary',
  ],
}

// --------------------------------------------------------------------------
// Prompt builders. Agents are varied by index/round, never by randomness
// (Date.now/Math.random are forbidden and throw).
// --------------------------------------------------------------------------

const FAMILIES = ['raw', 'lowrank', 'drift', 'align', 'norm']

// Rotate which seed family the generator is nudged to emphasize each round, so
// successive rounds explore the space instead of re-proposing the same specs.
function emphasisForRound(round) {
  return FAMILIES[round % FAMILIES.length]
}

function GEN_PROMPT(seenNames, n, round) {
  const seenBlock = seenNames.length
    ? seenNames.map((x) => '- ' + x).join('\n')
    : '(none yet)'
  return [
    'You are the GENERATOR in a generator-critic autoresearch loop searching for',
    'a new, near-zero-cost, PRE-ROUND signal of draft-target acceptance computed',
    'from already-cached verified-context representations (I13/I23).',
    '',
    'Propose exactly ' + n + ' candidate FeatureSpecs. Each spec is a TYPED entry',
    'from the fixed seed library — NOT free-form code. Allowed families:',
    '  raw     : raw residual-stream components at chosen layers',
    '  lowrank : minimal low-rank projections of a selected layer frontier state',
    '  drift   : divergence velocity / curvature of the residual stream',
    '  align   : draft-target representational alignment (shared tokenizer)',
    '  norm    : norm / surprise-budget summaries',
    '',
    'Hard constraints (the honesty firewall — you cannot relax these):',
    '- The frozen bar to beat is the `preround_hardened` baseline (~0.73 AUROC).',
    '  A candidate must add INCREMENTAL lift over that full baseline stack.',
    '- Features must be available BEFORE the round drafts (no post-draft / no',
    '  same-round outcome fields — that is label leakage).',
    '- Give each spec a name, family, layers (int[]), params (object), a',
    '  falsifiable hypothesis, and an a-priori cost_class',
    '  (near-zero | cheap | draft-priced).',
    '- Use "diagnostic signal" / "representation" language only. Do NOT call any',
    '  candidate a "circuit" or "mechanism" (G2-gated, D020).',
    '',
    'This round, lean toward the "' + emphasisForRound(round) + '" family for',
    'novelty, but you may mix families.',
    '',
    'Do NOT re-propose any signal already seen (killed or surviving) — pick',
    'genuinely new names and parameterizations distinct from these:',
    seenBlock,
  ].join('\n')
}

// Slugify a spec name for a filesystem-safe artifact path.
function slug(name) {
  return String(name).replace(/[^a-zA-Z0-9_-]/g, '_')
}

function FIT_PROMPT(spec, runId) {
  const specJson = JSON.stringify(spec)
  const outPath = 'artifacts/autoresearch/' + runId + '/' + slug(spec.name) + '.json'
  // The fit CLI reads saved I07/I10/frontier artifacts, fits the pre-round
  // feature on prompt-grouped OOF dev folds, and reports incremental lift over
  // the frozen baseline plus equal-dimensionality norm/random controls.
  const cmd =
    "PYTHONPATH=src python scripts/fit_autoresearch.py --run " + runId +
    " --eval dev --spec-json '" + specJson + "' --out " + outPath
  return [
    'You are the FIT+EVAL executor. Run EXACTLY this command in your shell and',
    'do nothing else that mutates artifacts:',
    '',
    '  ' + cmd,
    '',
    'The CLI reads the saved I07 traces / I10 & frontier activations / probe',
    'artifacts (it does NOT re-run the decode engine) and prints/saves a JSON',
    'object. Parse that JSON and return it verbatim. It has keys: base, combined,',
    'control_random, control_norm (each {auroc,auprc,brier,ece,regret}), deltas,',
    'beats_baseline, beats_controls, delta_auroc_ci, n, pos_rate. Ensure the',
    'returned object also carries name="' + spec.name + '". Do not edit the eval,',
    'do not touch test folds, do not hand-edit any number.',
  ].join('\n')
}

const LENS_INSTRUCTION = {
  leakage: [
    'LENS = leakage. Try to KILL this finding by showing the lift is label',
    'leakage: is the feature actually available only AFTER the draft, or does it',
    'encode a same-round outcome field? Any post-draft availability => refuted.',
  ].join('\n'),
  capacity: [
    'LENS = capacity. Try to KILL this finding by showing the lift is merely',
    'extra capacity: does it beat the NORM-MATCHED and RANDOM controls of EQUAL',
    'dimensionality, or would a same-dimension nuisance feature match it? If the',
    'edge over equal-capacity controls is not clear => refuted.',
  ].join('\n'),
  transfer: [
    'LENS = transfer. Try to KILL this finding by showing it will not generalize:',
    'would the lift survive a HELD-OUT domain AND the second model pair (I17)? If',
    'transfer is doubtful => refuted.',
  ].join('\n'),
}

function REFUTE_PROMPT(result, lens) {
  return [
    'You are an ADVERSARIAL CRITIC. Your job is to REFUTE (kill) this candidate',
    'pre-round acceptance signal, not to praise it. DEFAULT TO refuted=true under',
    'any uncertainty; only set refuted=false if the finding clearly survives your',
    'lens.',
    '',
    LENS_INSTRUCTION[lens],
    '',
    'Candidate name: ' + result.name,
    'Fit+Eval result (JSON): ' + JSON.stringify(result),
    '',
    'Return {refuted, reason, lens} with lens="' + lens + '". Use "diagnostic',
    'signal" language only — never "circuit"/"mechanism" (G2-gated, D020).',
  ].join('\n')
}

function CHARACTERIZE_PROMPT(result) {
  return [
    'You are the CHARACTERIZER. This candidate survived the adversarial critic.',
    'Using the saved artifacts and the fit result, map WHERE the signal transfers',
    'and WHERE it breaks. Return a FAILURE_MAP: by_domain, by_category, by_phase,',
    'by_model_pair (each an object keyed by slice -> lift/status), a',
    'deployed_cost_note describing its marginal wall-clock cost status on the',
    'compiled timing path (near-zero / cheap / draft-priced), and a short summary.',
    '',
    'Candidate name: ' + result.name,
    'Fit+Eval result (JSON): ' + JSON.stringify(result),
    '',
    'Remember: this is a CANDIDATE, not a claim. Use "diagnostic signal" /',
    '"representation" language only — never "circuit"/"mechanism" (G2-gated, D020).',
  ].join('\n')
}

// --------------------------------------------------------------------------
// The loop (§5 / §8). Pipeline, no barrier except the dedup step.
// --------------------------------------------------------------------------

const N = 8 // candidate specs proposed per round
const RUN_ID = (args && args.run_id) || 'sweep-2026-07-11T203836'

const seen = new Set()   // every candidate name EVER seen (killed or survived)
const survived = []      // characterized survivors — CANDIDATES, not claims
let dry = 0              // consecutive rounds with no fresh candidate
let round = 0

// Loop-until-dry: stop after 2 consecutive dry rounds, or when the token budget
// falls below the reserve. Plain "run M rounds" caps miss the tail (§5.7).
while (dry < 2 && (!budget.total || budget.remaining() > 80_000)) {
  log('Round ' + round + ' | seen=' + seen.size + ' | survived=' + survived.length +
      ' | remaining=' + budget.remaining())

  // 1. Generate candidate feature specs (seeded by §3.A + the seen ledger).
  phase('Generate')
  const gen = await agent(GEN_PROMPT(Array.from(seen), N, round), {
    label: 'generate-r' + round,
    phase: 'Generate',
    schema: FEATURE_SPEC_LIST,
  })
  const proposed = (gen && Array.isArray(gen.specs)) ? gen.specs : []

  // Dedup vs SEEN (not vs survived) BEFORE fitting, so critic-rejected signals
  // do not get re-fit every round (§8 warning).
  const toFit = proposed.filter((s) => s && s.name && !seen.has(s.name))
  if (!toFit.length) {
    dry++
    log('No fresh candidates proposed; dry=' + dry)
    round++
    continue
  }

  // 2+3. Fit each fresh pre-round spec on dev OOF folds via the fit CLI; the CLI
  // computes incremental lift over the frozen baseline + equal-dim controls.
  phase('Fit+Eval')
  const fitted = (await parallel(toFit.map((s, i) => () =>
    agent(FIT_PROMPT(s, RUN_ID), {
      label: 'fit-r' + round + '-' + i,
      phase: 'Fit+Eval',
      schema: LIFT_RESULT,
    }))))
    .filter(Boolean)                       // thunk errors -> null
    .filter((r) => r.name && !seen.has(r.name)) // re-dedup vs seen (§8)

  if (!fitted.length) {
    dry++
    log('No fresh Fit+Eval results; dry=' + dry)
    round++
    continue
  }
  // Mark ALL fitted candidates seen — survived or not — so nothing reappears.
  dry = 0
  fitted.forEach((r) => seen.add(r.name))

  // Only candidates that clear the frozen bar AND the equal-capacity controls
  // proceed to the critic.
  const passed = fitted.filter((r) => r.beats_baseline && r.beats_controls)
  if (!passed.length) {
    log('No candidate passed Fit+Eval this round.')
    round++
    continue
  }

  // 4. Adversarial critic — 3 refuters per passed candidate, distinct lenses;
  // majority-refute kills (keep only if >=2 of 3 say NOT refuted).
  phase('Critic')
  const lenses = ['leakage', 'capacity', 'transfer']
  const judged = (await parallel(passed.map((r) => () =>
    parallel(lenses.map((lens) => () =>
      agent(REFUTE_PROMPT(r, lens), {
        label: 'refute-' + slug(r.name) + '-' + lens,
        phase: 'Critic',
        schema: VERDICT,
      })))
      .then((verdicts) => {
        // Failed/absent verdicts are null -> not counted as "not refuted",
        // which preserves default-to-refuted under uncertainty.
        const notRefuted = verdicts.filter(Boolean).filter((v) => !v.refuted).length
        return { r, keep: notRefuted >= 2 }
      }))))
    .filter(Boolean)
  const kept = judged.filter((j) => j.keep).map((j) => j.r)
  log('Critic kept ' + kept.length + ' of ' + passed.length + ' candidate(s).')

  // 5. Failure-geometry + deployed-cost characterization for survivors.
  phase('Characterize')
  const characterized = (await parallel(kept.map((r) => () =>
    agent(CHARACTERIZE_PROMPT(r), {
      label: 'characterize-' + slug(r.name),
      phase: 'Characterize',
      schema: FAILURE_MAP,
    }).then((fm) => ({ name: r.name, lift: r, failure_map: fm })))))
    .filter(Boolean)

  survived.push(...characterized)
  round++
}

// The survivors are CANDIDATES, not claims. A human trips G1/G2/G3 and moves a
// candidate to a ledger claim (docs/CLAIMS_LEDGER.md); the loop promotes nothing
// automatically. No "circuit"/"mechanism" language is applied pre-G2 (D020).
return { survived, seen_count: seen.size }
