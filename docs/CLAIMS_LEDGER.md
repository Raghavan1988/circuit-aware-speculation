# Claims Ledger

Allowed statuses: `UNTESTED`, `SUPPORTED`, `PARTIAL`, `REFUTED`, and `RETIRED`.

No claim may move to `SUPPORTED` without experiment identifiers, applicable settings, uncertainty, and known counterexamples. A manuscript claim must be no broader than this record.

| ID | Proposed claim | Status | Required evidence | Experiment IDs | Counterexamples / limits |
|---|---|---|---|---|---|
| C01 | Draft hidden states contain acceptance information beyond entropy, margin, history, and domain. | PARTIAL (negative-leaning) | Prompt-grouped incremental comparison on held-out prompts | I10/I12 dev probe, run sweep-2026-07-11T203836 (2026-07-12) | **Not supported for a LINEAR probe on the Qwen pair:** hidden-only AUROC peaks 0.803 (layer 18) vs surface 0.870; hidden⊕surface adds ≤ +0.006 AUROC (layers 18/24 only, not shown significant). Cheap surface signals are a strong, hard-to-beat baseline. Scope: linear probe, 4 layers, 120 dev prompts, 42k tokens. Nonlinear probes / other layers / the Llama pair (I17) untested. See Run log 2026-07-12 |
| C02 | Acceptance information becomes accessible at identifiable draft-model layers. | UNTESTED | Layerwise probes replicated across domains and a second model setting | — | — |
| C03 | Rejection-associated directions have a controlled effect on draft–target divergence or acceptance. | PARTIAL | Dose-response intervention with random and norm-matched controls | I15 intervene, held-out test rounds of `sweep-2026-07-11T203836` (Qwen-v1) and `sweep-llama-f8-2026-07-13` (Llama); `analysis/<run>/intervene.json` (2026-07-22) | **First-token acceptance only** (does not extend to run-length, per the I23 survival negative). Steering the first-token acceptance direction in the TARGET's cached frontier representation disrupts acceptance ~2–10× more than norm-matched random/shuffled controls, dose-dependently, beyond induced entropy, at all 4 layers, both families. Scope: representation-level control of the target's next-token agreement-ability — NOT a draft–target "circuit"; language upgrade is a human gate (D020). See causal_intervention_report.md + 2026-07-22 note |
| C04 | Acceptance behavior differs systematically across token categories and generation phases. | SUPPORTED — CATEGORY axis only (frozen test 3/3, 2026-07-22); phase axis REFUTED-as-inconsistent | Acceptance atlas with paired uncertainty and annotation validation | `c04_atlas_{dev,test}.json` on all three settings (machinery 80ee48b; pre-registration 968c425); structural-vs-entity contrast on test: overall +0.195/+0.210/+0.118 (p=0), within-domain CI-clean in 15/15 cells incl. the pre-registered expected-null (Llama-code +0.067) | **Category axis only:** the phase half is null on Qwen (spread ≤0.9pp both corpora) and monotone-increasing on Llama only (descriptive; length/mix confound possible) — no cross-family phase claim. Weakest cell: Qwen-v1 summ (+0.054, p=0.027, CI lo −0.002). Domain-grain prior (arXiv:2604.14682) reproduced as CONTROL; our axis is category-within-domain. Labels: overlapping categories v1.0.0, counterfactual fixed_8 `target_match`. Llama `reasoning_transition` is not low (Qwen-only); Llama `special` ≈0.007 anomaly recorded descriptively |
| C05 | Selective speculation with a skip action reduces wasted compute relative to adaptive-length baselines. | UNTESTED | Held-out comparison including all overhead | — | — |
| C06 | The circuit-aware controller improves net latency over the best global fixed policy. | UNTESTED | Paired held-out wall-clock study with uncertainty | — | **Harness-dependent (2026-07-12, T3.4):** the routing headroom this claim needs is ~5% on the eager launch-bound harness (best fixed action = skip) and only reaches ~25–46% under a serving-grade fused+graph-captured draft. Any net-latency claim must state the execution mode. See Run log 2026-07-12 |
| C07 | Any controller advantage persists against the best per-domain fixed policy. | UNTESTED | Per-domain held-out comparison | — | — |
| C08 | The signal or controller transfers under domain and traffic shift without full retuning. | UNTESTED | Shift study with calibration drift and latency regret | — | — |
| C09 | The principal finding replicates outside the primary Qwen pair. | UNTESTED | Compatible Llama pair or approved Qwen-ratio fallback | — | — |
| C10 | Next-round acceptance is predictable from cached verified-context representations before any draft compute, at deployable marginal cost. | SUPPORTED (first-token scope; frozen test 3/3 under BOTH protocols, 2026-07-22) | Prompt-grouped held-out comparison against post-draft signals (entropy, margin, learned head), with deployed-path overhead measured (I23) | I13/I23 frozen test pass, two protocols. Within-test OOF (`autoresearch_test_domctl.json`): Δauroc +0.0755/+0.0918/+0.0542 (Qwen-v1/v2/Llama, pass-1 archived; redo reproduces to ±0.001). STRICT dev→test transfer, nothing fit on test (`autoresearch_frozen_transfer_domctl.json`): **+0.0555 [+0.0414,+0.0700] / +0.1027 [+0.0830,+0.1242] / +0.0393 [+0.0225,+0.0565]**; all p(Δ≤0)=0, controls-clean, domain-controlled. Causal I15 (2026-07-22 note). Decision-proxy `credible_systems` is protocol-sensitive on Qwen-v1 (strict: robust at one cost only) — wall-clock systems claim stays G3-gated. Novelty re-check 2026-07-22 (I21): **cell still unoccupied** across 12 newly-verified + 4 previously-verified adjacent works; re-run citing sweep at manuscript freeze (see landscape.md §I21 2026-07-22) | **Narrowed 2026-07-11 (I21 R2):** AdaEAGLE (2412.18910; pre-draft length from target verified-context features, EAGLE, uncalibrated, no skip/baselines); Judge Decoding (2501.19309; target-embedding judge **during verify**, relaxes losslessness); WhiFlash (2606.07710; pre-draft drafter routing). Full C10 cell (lossless, calibrated, baseline-controlled, independent drafter, skip-capable) still unoccupied. Baselines must include free frontier entropy/margin at last verified position (cf. 2606.30265). SemanticSpec (2602.03708) does not scoop (mid-verify semantic-relaxed probes). See landscape.md + ledger notes |
| C11 | The identified acceptance representation transfers beyond independent drafts to a modern speculator family. | UNTESTED | Cross-speculator evaluation; extension work, only after the core evidence gate (D009) | — | — |

## Landscape verification notes (I21)

### 2026-07-10 — Planning-pass four-paper verification (Grok)

- **Primary confirmed:** arXiv:2603.01639 (Learning to Draft / LTD — RL draft+verify throughput), arXiv:2605.02888 (SpecKV — draft confidence under compression), arXiv:2604.14682 (acceptance dynamics across cognitive domains), arXiv:2606.30265 (theory of acceptance certificates).
- **Mirror-found item:** arXiv:2606.30265 — **primary archive confirmed** at https://arxiv.org/abs/2606.30265 (v1, 29 Jun 2026). Not a phantom.
- **C04 verdict:** 2604.14682 is domain-grain (+ coarse early/late position bins, weak entropy–α). Does **not** pre-empt token-category/phase atlas. Status stays UNTESTED; prior work cited as control.
- **C10 / theory:** 2606.30265 constrains a pure-theory Track B option (certificates for local acceptance events) but is not an internal localization result and does not implement pre-round prediction.
- **Living table:** `docs/landscape.md` (authoritative). PLAN.md §3 edit proposed there as a note only.
- Logged by Grok, 2026-07-10.

### 2026-07-10 — I11 annotation tooling (not claim evidence)

- Implemented `cas.annotate` (CATEGORY_SET_VERSION=v1.0.0, PHASE_SET_VERSION=v1.0.0): overlapping categories + absolute generation-phase bins; pure stream function for the I06 seam.
- Stratified golden-sample agreement: run `PYTHONPATH=src python -m pytest tests/test_annotate.py -q -s` — agreement is script-printed, not hand-typed into this ledger as a scientific result.
- This does **not** move C04; acceptance-atlas numbers require real decode traces (I07+).
- Logged by Grok, 2026-07-10.

### 2026-07-11 — Literature-sweep impacts on C01/C10 (Claude; 25-agent verified sweep, details in CLAUDE_IDEAS.md)

- **C01 instrument scooped, science claim open:** DSpark (arXiv:2607.05147,
  2026-07-06, full text verified) publishes a calibrated **linear** head on
  draft hidden states for per-token acceptance (AUROC 0.81–0.90, ECE ≈1% after
  per-position temperature scaling). C01 must be framed as the controlled
  incremental-information study (vs combined entropy+margin+history+domain
  baseline, prompt-grouped splits, layer localization) — no published work has
  run that comparison; that negative finding survived checks across ~70 papers.
- **C10 narrowed, not scooped — new counterexamples to record:** AdaEAGLE
  (arXiv:2412.18910; pre-draft next-round length regression from the target's
  last-verified-token hidden state — same input class/timing, EAGLE setting,
  uncalibrated, no baselines, no skip); Judge Decoding (arXiv:2501.19309;
  linear head on target embeddings judging draft acceptability — during verify,
  relaxes losslessness); WhiFlash (arXiv:2606.07710; target-hidden-state
  pre-draft drafter routing). C10's pre-round, lossless, calibrated,
  baseline-controlled, independent-drafter cell remains unoccupied as of
  2026-07-11. Adjacent activity is monthly — supports staged-release urgency
  (D010).
- **C10 baseline hardening required before freeze:** target next-token entropy
  and top-1/top-2 margin at the last verified position are free byproducts of
  the verification pass and, per acceptance theory (arXiv:2606.30265), carry
  the governing variable. C10's baseline must include them (schema fields
  proposed for I06), else the claim is vulnerable to a one-line rebuttal.
- **ID/venue corrections for I21 queue:** Not-a-Bandit primary ID is
  2510.20064 (2506.00285 resolves to a different paper); DSpark's verified
  paper text does not mention SGLang — the landscape's "SGLang release"
  attribution needs re-verification.
- Logged by Claude, 2026-07-11.

### 2026-07-11 — I21 round-2 landscape corrections + G2 SemanticSpec (Grok)

- **ID correction applied:** Not-a-Bandit primary arXiv ID is **2510.20064**
  (title matches). Prior landscape ID **2506.00285** is *Lazy Heuristic Search
  for Solving POMDPs…* (cs.RO) — unrelated; removed as Not-a-Bandit cite.
  PLAN.md §3 still needs Claude's one-line ID fix if it retained 2506.00285.
- **DSpark system attribution corrected:** arXiv:2607.05147 abstract deploys
  confidence-scheduled speculative decoding in the **authors' production
  serving stack under live user traffic** (vs MTP-1 baseline). Full text does
  not ship DSpark as an SGLang feature; prior landscape "DSpark in SGLang"
  row was wrong. Corrected in `docs/landscape.md`. Phase-2 engine reasoning
  that cited SGLang+DSpark should be re-read by PLAN owner.
- **Sweep additions primary-verified and tabled:** AdaEAGLE (2412.18910),
  Judge Decoding (2501.19309), WhiFlash (2606.07710), C2T (2502.13652),
  Sequoia (2402.12374), DISCO (2405.04304), AdaEDL (2410.18351),
  DSDE (2509.01083), TurboSpec (2406.14066), CaDDTree (2606.01813),
  plus methodology context 2509.10625 / 2606.14530 / 2506.08572.
- **G2 SemanticSpec (2602.03708) full-text verdict:** Probes multi-layer
  hidden states of **draft and target** during **verification of semantic
  sequences** to estimate semantic probability; accepts with a
  semantic-aware (non-token-exact) rule. Does **not** run incremental
  information vs entropy/margin/history/domain. Does **not** predict
  next-round acceptance from cached verified context before draft compute.
  **C01 science claim (controlled incremental-info study): not scooped.**
  **C10 pre-round lossless cell: not scooped.** Cite as adjacent internal-
  state speculation control under a relaxed correctness contract.
- **G3 deployed practice:** vLLM = static K + batch-size-tiered Dynamic SD
  (`num_speculative_tokens_per_batch_size`) + historical queue auto-disable;
  SGLang = EMA-of-accept-length adaptive step tiers (EAGLE/EAGLE3 only);
  TensorRT-LLM = static `max_draft_len`, no dynamic length path in one-model
  docs. Cheap systems baselines for I13/I14; documented in landscape
  §Deployed practice with doc links.
- **G4 C04 positioning:** Atlas must (1) reproduce domain-marginal acceptance
  as control matching 2604.14682, then (2) add category×phase within domain
  as the new axis. Full note in landscape.md §C04 domain-control positioning.
- **C10 claims-table cell updated** to name the three narrowing papers.
- Living table: `docs/landscape.md` (authoritative). PLAN.md edits proposed
  as notes only.
- Logged by Grok, 2026-07-11.

### 2026-07-12 — I10/I12 acceptance probe: cheap surface signals beat a linear hidden-state probe (C01 negative)

Teacher-forced draft residual-stream capture (I10) at layers 6/12/18/24 over 120
dev prompts of run `sweep-2026-07-11T203836` (42,568 tokens; token-to-activation
**align_rate 0.9916** vs the sealed proposals — the ~0.8% miss is bf16 cache-vs-
no-cache flips, dropped). Regularized logistic probes, prompt-grouped GroupKFold,
standardized hidden features (I12). Artifacts: `/artifacts/probes/…/acts_L*.npy`,
`metadata.parquet`, `probe_results.json`.

| feature set | AUROC |
|---|---|
| surface stack (entropy, margin, history, offset, pos) — the bar | **0.870** |
| hidden-only layer 6 / 12 / 18 / 24 | 0.734 / 0.769 / **0.803** / 0.789 |
| hidden⊕surface layer 6 / 12 / 18 / 24 | 0.864 / 0.868 / **0.876** / 0.873 |

- **Finding:** a linear probe on draft hidden states does **not** beat the cheap
  surface baseline (best hidden 0.803 ≪ surface 0.870), and adds at most **+0.006
  AUROC** when combined (late layers only; significance not established). For this
  pair, the free signals (entropy/margin/history) already capture the linearly-
  accessible acceptance information. Contextualizes DSpark (2607.05147): a linear
  acceptance head is matched here by signals that cost nothing to compute.
- **Layer trend (C02 direction, weak):** linear accessibility rises with depth
  (0.734 → 0.803, peak layer 18) then dips at the final layer — consistent with
  mid/late emergence, but never enough to beat surface.
- **Boundaries:** linear probe only (nonlinear untested); 4 layers; single pair;
  dev split. Does not touch C04 (atlas, latency- and probe-independent).
- Logged by Claude, 2026-07-12.

### 2026-07-12 — I17 Llama replication: draft-repo access still gated (blocked)

`CAS_PAIR=llama modal run …::verify` with the `huggingface-token` secret
attached. **401 on `meta-llama/Llama-3.2-1B-Instruct`** (the draft) —
"Access to model … is restricted." The error is specific to the 1B draft, so
either that repo's gated access is not yet granted on the token's account
(separate from Llama-3.1-8B) or the secret's token lacks it. No SHAs resolved;
config Llama revisions remain None (unpinned). Pair-switch wiring (CAS_PAIR,
HF secret on GPU fns) is in place and ready once access is granted.
- Logged by Claude, 2026-07-12.

### 2026-07-12 — RQ2 adaptive length: entropy-stop beats best fixed length on efficiency (positive, generalizes)

Offline replay of length policies over the sealed `fixed_8` counterfactual labels
(`scripts/eval_length_policies.py`; every round carries per-position draft entropy
and match, so any length policy is evaluable exactly without re-running a model).
Latency-independent metrics; threshold tuned on dev, **evaluated held-out on test**
with the frozen value. Artifacts: `analysis/…/rq2_length_policies_{dev,test}.json`.

Held-out **test** (19,074 rounds), serving-cost basis (draft forward = 0.1 x verify):

| policy | yield (emit/round) | wasted/emit | eff (serving) |
|---|---|---|---|
| best fixed = fixed_8 | 4.324 | 1.081 | 2.402 |
| **entropy-stop (tau=2.0, dev-frozen)** | 3.854 | **0.408** | **2.672 (+11.2%)** |
| history-EMA | 3.324 | 0.434 | 2.415 (+0.5%) |
| oracle (per-round best) | 4.324 | 0.017 | 3.228 |

- **Finding:** the SVIP-style entropy-stop controller beats the best fixed length
  by **+11.2%** serving-efficiency on held-out test (dev +11.3% — transfers), and
  cuts **wasted draft tokens by ~62%** (0.408 vs 1.081 per emitted token) while
  keeping ~89% of the yield. It captures ~33% of the oracle headroom over best
  fixed. History-EMA alone barely helps (+0.5%): entropy is the effective content
  signal for length. This is the roadmap RQ2 / Phase-1 result.
- **Scope / honest caveat:** this is under a **serving-realistic draft cost** (the
  small draft is genuinely cheap). Under the current **launch-bound** harness
  (eff_launch column, all forwards equal cost) skip still dominates, consistent
  with the M3 stop finding. So the Phase-1 win is real in the deployment regime
  but contingent on the deferred StaticCache fast-draft path. Not a measured
  wall-clock number; evidence toward C05/C06 scoped to the serving-cost model.
- Logged by Claude, 2026-07-12.

### 2026-07-12 — RQ3 draft-routing feasibility: code specialist ties the general draft (no-go, well-powered)

Apples-to-apples acceptance probe (`modal_app.py::probe_draft`): the Qwen2.5-7B-
Instruct target's greedy continuation is generated once per prompt, then each
candidate draft's per-token argmax agreement (drift-free acceptance) is scored on
the identical continuation. `Qwen2.5-Coder-0.5B-Instruct` shares the target
tokenizer exactly (151,665 vocab, base-ids aligned) — exact spec decoding is
possible. Artifacts: `analysis/rq3_draft_probe_code.json`.

Code, all **164** HumanEval prompts (18,471 tokens/draft, prompt-grouped):

| draft | code acceptance |
|---|---|
| general Qwen2.5-0.5B-Instruct | 0.8746 |
| code Qwen2.5-Coder-0.5B-Instruct | 0.8745 |

Paired diff (Coder − general) = **−0.0024, 95% CI [−0.0077, +0.0035]** (bootstrap
over 164 prompts). **CI includes 0 → statistically indistinguishable.**

- **Finding:** an off-the-shelf code specialist does **not** beat the general
  instruct draft on code against a general instruct target. Mechanism: acceptance
  tracks how well the draft matches the *target's* distribution, and the general
  0.5B-Instruct is already family-matched to the 7B-Instruct target; domain
  expertise in the draft does not add. This is a well-powered no-go (the 40-prompt
  pilot showed the same direction, underpowered).
- **Consequence for RQ1/RQ3:** with a single general target, the draft-routing
  axis has no headroom here, so the joint controller (RQ1) collapses to the length
  controller (RQ2) on this pair. Empirically supports deferring draft routing
  (D003/D009). The higher-value specialists would be target-distilled EAGLE heads
  (deferred C11), not off-the-shelf domain models. Math arm + full draft×domain
  matrix in progress (`rq3_draft_matrix.json`).
- **Metric note:** drift-free teacher-forced acceptance (0.87) is a block-length-1
  upper bound; the sweep's 0.523 code number is over 8-token blocks where the
  draft drifts. The general-vs-specialist comparison is apples-to-apples on the
  identical metric, so the tie is valid.
- Logged by Claude, 2026-07-12.

### 2026-07-12 — RQ3 exhaustive draft x domain matrix: specialization does not help, size does (definitive no-go)

Full draft x domain acceptance matrix (`modal_app.py::draft_matrix`, 644 prompts,
drift-free per-token acceptance vs the shared Qwen2.5-7B-Instruct greedy
continuation; drafts loaded one at a time). All five drafts share the target
tokenizer (aligned). Artifact: `analysis/rq3_draft_matrix.json`.

| draft | chat | code | math | summ |
|---|---|---|---|---|
| general 0.5B | 0.738 | 0.898 | 0.910 | 0.672 |
| Coder 0.5B | 0.675 | 0.899 | 0.877 | 0.603 |
| **general 1.5B** | **0.789** | **0.917** | **0.928** | **0.736** |
| Math 1.5B | 0.662 | 0.862 | 0.919 | 0.554 |
| Coder 1.5B | 0.753 | 0.917 | 0.917 | 0.695 |

Size-matched paired diffs (prompt-grouped bootstrap 95% CI):
- Coder-0.5B − general-0.5B @ code: −0.0033, CI [−0.0087, +0.0018], n=164 → **tie**
- Coder-1.5B − general-1.5B @ code: −0.0037, CI [−0.0091, +0.0008], n=164 → **tie**
- Math-1.5B − general-1.5B @ math: −0.0088, CI [−0.0128, −0.0050], n=200 → **SIGNIFICANT (specialist WORSE on its own domain)**

- **Findings:** (1) code specialists **tie** the size-matched general draft on
  code (both sizes); (2) the math specialist is **significantly worse than the
  size-matched general even on math**, and much worse off-domain (drifted from the
  target's distribution); (3) **draft SIZE dominates specialization** — general-1.5B
  is the best-or-tied draft in every domain and beats general-0.5B everywhere; (4)
  the **oracle router ≈ general-1.5B in all four domains**, so routing among
  off-the-shelf specialists yields ~zero benefit over the general draft of the
  right size.
- **Verdict:** RQ3 (draft routing) is a **well-powered no-go across all domains** on
  this pair; RQ1 (joint) collapses to RQ2 (length). For exact-match spec decoding
  with a general instruct target, acceptance is governed by draft-vs-target
  distribution match and draft capacity, not draft domain expertise. Empirically
  validates deferring draft routing (D003/D009); the only promising specialist
  direction is target-distilled EAGLE heads (deferred C11), not domain models.
- Logged by Claude, 2026-07-12.

### 2026-07-13 — T5.3 controller comparison + T5.4 error taxonomy (roadmap A2/A4)

Same replay protocol as the RQ2 entry (sealed fixed_8 labels, serving-cost basis
draft=0.1×verify, held-out test, 19,074 rounds; stream order = sorted request_id).
Artifacts: `analysis/…/rq2_length_policies_test.json`, `t5_4_taxonomy_test.json`.

**T5.3 — online bandits converge to best-fixed; the contextual rule wins:**

| controller | eff (serving) | wasted/emit |
|---|---|---|
| best fixed (L=8) | 2.4022 | 1.081 |
| UCB1 over lengths | 2.4015 | 1.075 |
| epsilon-greedy (mean of 3 seeds) | 2.3661 | ~1.05 |
| entropy-stop (tau=2.0, dev-frozen) | **2.6715** | **0.408** |

- Stationary context-free bandits (UCBSpec/epsilon-greedy style) learn the best
  single arm and therefore **tie the best fixed length by construction** — they
  cannot exceed it. The +11.2% win comes from **per-round context** (draft
  entropy), not from online arm selection. Controller ranking: entropy-stop >
  UCB ≈ eps-greedy ≈ best fixed > history-EMA.
- Caveat: the replay stream is domain-sorted, so per-quartile efficiency trends
  reflect domain composition, not regret; a windowed non-stationary bandit could
  at most recover the domain-level gain (~+4.4%, see routing-opportunity note),
  still well below the contextual +11.2%.

**T5.4 — error taxonomy for the frozen entropy controller (test):**

| domain | over-draft rounds | over tok/round | under-draft rounds | under tok/round |
|---|---|---|---|---|
| math | 48.0% | 2.44 | 4.6% | 0.15 |
| code | 33.9% | 1.53 | 13.8% | 0.46 |
| chat | 34.1% | 1.26 | 20.2% | 0.55 |
| summ | 34.1% | 1.22 | 23.7% | 0.63 |

- Failure modes are domain-skewed: on math the controller mostly **over-drafts**
  (confident but wrong continuations); on summarization/chat it mostly
  **under-drafts** (entropy over-fires and stops runs that would have matched).
  Headroom to the oracle (2.67 → 3.23) lives in exactly these cells.
- **Entropy calibration is cleanly monotone** (P(accept) 0.969 at H≤0.25 down to
  0.164 at H>6, n=152k positions) — the signal the threshold rule relies on is
  well-behaved; no miscalibration pathology.
- **Candidate-set ablation (A4):** a coarse {0,1,4,8} menu retains ~97-98% of the
  value (oracle 3.135 vs 3.228; entropy-stop 2.622 vs 2.668) — a 4-arm menu is
  nearly free of loss, which simplifies deployment.
- Batch interference: not measurable (batch-1 harness) — stated, not omitted.
- Also fixed this date: the sweep split-stamp bug (`assignment.get(prompt_id)` →
  `prompt_hash`, modal_app run_policy) so future sweeps stamp splits correctly.
- **RQ2 headline uncertainty (added same date):** prompt-grouped bootstrap
  (2,000 resamples over 322 held-out test prompts, `rq2_ci_test.json`):
  entropy-stop vs best fixed relative efficiency delta **+11.2%, 95% CI
  [+10.3%, +12.1%]**, P(delta<=0) = 0/2000. Wasted-per-emitted CIs disjoint:
  stop 0.408 [0.388, 0.430] vs fixed 1.081 [1.008, 1.157].
- Logged by Claude, 2026-07-13.

### 2026-07-13 — I17 Llama replication unblocked; cross-family equivalence gate passes (fp32 118/118)

- **Access resolved:** the Modal `huggingface-token` secret exposed `HUGGINGFACE_TOKEN`,
  but `huggingface_hub`/`transformers` read `HF_TOKEN` — root cause of the
  2026-07-12/13 gated-repo 401s despite an approved gate. Fixed by mapping the var
  at container import + refreshing the token (owner). `hfcheck`: whoami=Raghavan1988,
  both repos ACCESS OK; real download + forward succeeded.
- **Tokenizer compatibility (cross-family):** Llama-3.1-8B-Instruct and
  Llama-3.2-1B-Instruct share the 128,256-vocab Llama-3 tokenizer; `load_pair` did
  not raise → exact-match speculative decoding is valid on this pair. Revisions
  pinned: target `0e9e39f2…`, draft `9213176726f5…` (config.py, D014).
- **Equivalence gate:** `run_tests(dtype=float32, pair=llama)` on A100-80GB →
  **118 passed, 0 failed** (261.7s), including the GPU bit-identity tests
  (fixed lengths, skip, stop-rule, eos, D018 fields, fp-divergence). The exact
  greedy engine is lossless on the Llama pair, replicating the fp32 Qwen result.
- **Scope:** confirms the ENGINE cross-family. The science replication (atlas,
  RQ2 adaptive length) follows from the Llama fixed_8 sweep
  (`sweep-llama-f8-2026-07-13`, v1 corpus, bf16, max_new=256). RQ3 draft-routing is
  NOT replicated cross-family (no tokenizer-compatible Llama code/math specialist
  draft pool) — stated boundary, not fudged.
- Logged by Claude, 2026-07-13.

### 2026-07-13 — RQ3 replicated on the representative v2 corpus (7 axes): no-go confirmed and strengthened

Draft x domain matrix on corpus v2 (`rq3_draft_matrix_data_v2.json`, 1,494 prompts,
7 axes incl. the new translation / qa_rag / structured). Same 5 Qwen drafts vs the
7B target; drift-free acceptance; prompt-grouped paired 95% CIs.

- **general-1.5B is the best draft in EVERY axis** (oracle router = general-1.5B for
  all 7 domains), including the new ones: qa_rag 0.691, structured 0.875, translation 0.714.
- Size-matched specialist vs general: Coder-0.5B−general-0.5B @ code −0.0031
  [−0.0067, +0.0004] (tie); Coder-1.5B−general-1.5B @ code −0.0062 [−0.0094, −0.0032]
  (**significantly worse**); Math-1.5B−general-1.5B @ math −0.0088 [−0.0128, −0.0050]
  (**significantly worse**).
- The math specialist collapses on off-domain content (qa_rag 0.461 vs 0.691,
  structured 0.678 vs 0.875, translation 0.492 vs 0.714 vs the same-size general):
  domain-tuning drifts the draft away from the general target's distribution.
- **Verdict:** RQ3 draft-routing no-go holds — and is stronger — on the
  representative corpus. Draft SIZE dominates specialization across all 7 axes;
  routing among off-the-shelf specialists yields ~zero benefit.
- Logged by Claude, 2026-07-13.

## Evidence record template

When updating a claim, append:

```text
Claim:
Status:
Experiment IDs and code revision:
Models, datasets, and split:
Estimate and uncertainty:
Controller/tracing overhead included:
Known counterexamples:
Interpretation boundary:
Updated by/date:
```

## Run log: failed runs and negative results

Faithful record of failed/aborted runs per AGENTS.md. Numbers are copied from
immutable run logs, never hand-estimated.

### 2026-07-10 — I03 equivalence gate failed on the primary bf16 pair (unresolved → pending fp32 confirmation)

- **Run:** `CAS_GPU=A100 modal run modal_app.py::run_tests` (Modal app
  `ap-l3nL9Xr4BBy67rX0ML33vG`), Qwen2.5-7B-Instruct `a09a3545` target /
  Qwen2.5-0.5B-Instruct `7ae55760` draft, dtype **bfloat16**, eager attention,
  A100-40GB, transformers 4.46.3 / torch 2.5.1+cu124.
- **Result:** 18 passed, 7 failed in 604s. `test_bit_identical_to_greedy`
  failed for all L∈{1,2,3,4,6,8}; `test_fp_divergence_rate` = **6/12 = 0.50**
  sequence-level (ceiling 0.05). `test_skip_action_equivalent` (L=0) **passed**.
- **Root cause (identified, not a cache bug):** skip (single-token target
  forwards) is exact; every L>0 round runs the target over `gap+proposals` in one
  forward. The parallel-verify vs. sequential-decode arithmetic difference
  (D014.2) is ~1e-2 in **bf16**, not the ~1e-7 the <0.1% expectation assumed.
  Back-solving 50%/sequence over 96 tokens ⇒ **~0.7% per-token** argmax flip; one
  flip derails the rest of the greedy continuation. Cache/commit bookkeeping was
  re-derived by hand for k=0, 0<k<L, and k=L and is correct.
- **fp32 confirmation (RESOLVED):** `CAS_GPU=A100 modal run
  modal_app.py::run_tests --dtype float32` (Modal app `ap-ECNFNMgvfLdXnXdFBQYMSQ`),
  in-container dtype verified `EngineConfig.target.dtype='float32'`. **25 passed,
  0 failed**; `test_fp_divergence_rate` = 0/12. The cache/commit/rollback and
  batched-verify logic are therefore algorithmically exact; the bf16 gate failure
  was floating-point argmax-tie noise, not a bug. (A prior attempt setting
  `CAS_DTYPE` as a local env var did NOT reach the Modal container and silently
  re-ran bf16 — hence dtype is now passed as a function arg with an in-container
  assert.)
- **Remaining (reporting/calibration, not code):** (a) change
  `test_fp_divergence_rate` to a per-token rate with logged top-1/top-2 margins at
  each flip; (b) record the measured bf16 per-token rate as a dated D014 addendum
  (the "<0.1%" figure was fp32-calibrated). I03 algorithmic correctness is
  confirmed; the bf16 losslessness statement is "lossless up to logged argmax
  ties at the measured rate."
- Logged by Claude, 2026-07-10.

### 2026-07-11 — Engine review fixes + fp32 re-gate + I07 smoke characterization

- **Adversarial review (22 agents) of the D018 engine wiring** confirmed 5
  defects, one **critical and pre-existing**: tokens after an accepted mid-round
  eos were committed while greedy stops at eos — token-identity violation on any
  eos-terminating request (the prior gate's raw prompts rarely reached eos in 96
  tokens, hence never caught). Fixed same day (emit loop breaks at first
  committed eos; termination labeled from committed tokens; per-round
  start_output_pos snapshot; frontier signals moved inside measured "tracing"
  time; TokenTrace `accepted` split from counterfactual `target_match`). No
  results are invalidated: none existed. **Re-gate: fp32 73/73 passed** (Modal
  app `ap-85dmbyOiFYsL6Y1VxlTTRp`), including a new chat-template eos
  regression test, stop-rule equivalence at both extremes, and D018 field
  checks.
- **I07 smoke (cap=2, bf16, max_new=256, run `sweep-2026-07-11T172406`+fix):**
  all 9 policy trace runs sealed through the full writer path. Per-request
  equivalence vs the target-only reference: skip 2/2 identical; drafted
  policies 10/12 diverged — **consistent with the 2026-07-10 bf16
  characterization** (~0.7%/token ⇒ expected ≈83% sequence divergence at 256
  tokens; observed 83%). Divergences are fp argmax ties, flagged per-request in
  `equivalence_status`; per-round acceptance labels are exact in-context ground
  truth regardless of trajectory divergence.
- First smoke attempt OOMed (38GB): sweep-runner target-only helper lacked
  `torch.no_grad()`; fixed, plus allocator hygiene. Recorded as a failed run
  per AGENTS.md; no artifacts were sealed by the failed attempt.
- Logged by Claude, 2026-07-11.

### 2026-07-10 — I08/I09 baseline policy tooling (not claim evidence)

- Implemented resettable, pure-Python entropy-stop and rolling-acceptance
  policies plus a UCBSpec-style length-arm policy in `cas.policies`.
- UCBSpec deviations from arXiv:2505.15141: arms are the locked draft lengths
  including skip rather than heterogeneous hyperparameter specifications under
  one shared maximum; decoding is greedy exact-match; the generation cap is
  enforced by the engine; deterministic arm-order tie-breaking is used. The
  implementation retains round-robin initialization, accepted-plus-bonus output
  length as reward, and the published self-normalized confidence radius. The
  paper's stopping-time guarantee is not claimed for the changed arm definition.
- Unit tests demonstrate cold-start and post-initialization selection, threshold
  equality, rolling history, skip handling, and request reset. Reproduce with
  `PYTHONPATH=src python -m pytest tests/test_policies.py -q`.
- This is tooling only and does not change C05–C08. Held-out quality, latency,
  cold-start, and steady-state evidence require engine integration and real
  traces after I06/I07.
- Logged by Codex, 2026-07-10.

### 2026-07-11 — I09 round-2 repair and accepted-length policy tooling (not claim evidence)

- The round-1 raw-emitted-token selector is retained as `UCBSpecNaive`. The
  repaired `UCBSpecPolicy` requires an immutable development-measured cost for
  every action and puts both empirical rewards and confidence radii in emitted-
  tokens-per-cost units. No timing constants are embedded in policy code.
- Relative to arXiv:2505.15141, the disclosed draft-length-arm, greedy decoding,
  finite-cap, deterministic tie-break, and stopping-time caveats remain. The
  repaired reward/radius units are an additional intentional deviation; the
  published guarantee is not claimed for it.
- Added pure-stdlib accepted-length scaffolding: conditional continuation
  probabilities, monotone survival, expected-yield/cost selection including
  skip, first-rejection counterfactual labels, label-aware prompt-grouped Platt
  calibration, and explicit rejection of terminal/capped nominal-yield rows.
  A short row labels longer actions after an observed rejection; only a short
  all-accepted row is censored. Features are contractually pre-action.
- Added a design-only probe-as-prior UCB interface. A low-confidence prior has
  exactly zero pseudo-count and matches the realized-history fallback. It is not
  fitted or integrated with the engine.
- Reproduce the 32 deterministic unit tests with `PYTHONPATH=src python -m
  pytest tests/test_policies.py tests/test_survival.py -q`. Synthetic
  convergence fixtures are policy tests, not experimental measurements. No
  status changes to C05–C08 are warranted; I14 and held-out performance remain
  untested.
- Logged by Codex, 2026-07-11.

### 2026-07-12 — M3 oracle headroom + draft launch-bound characterization (negative/measurement)

Full sweep `sweep-2026-07-11T203836` (bf16, max_new=256, 644 prompts, all 8
policies sealed). Analysis reports on the `cas-artifacts` volume:
`analysis/…/t3_report.json`, `t3_4_bench_eager.json`, `t3_4_bench_sdpa.json`.
Every number below is script-generated from the sealed traces / bench, not
hand-entered.

- **M3 oracle headroom = STOP on this harness.** Counterfactual per-round
  best-action vs best fixed action (D018.3), costs measured per policy:
  headroom **compute-basis 1.68% / full-basis 2.66%**, best fixed action =
  **skip (L=0)** — below the D018 ~5% tripwire. On the eager harness, adaptive
  draft-length routing barely beats "don't speculate."
- **Root cause is a launch-bound draft forward, not acceptance.** T3.4
  micro-benchmark (single A100, median of 25): the 0.5B draft costs **≈24
  ms/token** and this is **invariant across configs** A (per-token
  entropy+margin+argmax with host syncs), B (signals off), C (signals batched,
  no per-token sync), D (pure floor) — so the earlier "instrumentation/sync
  overhead" hypothesis is **REFUTED**. Draft (0.5B, 24 layers) ≈ target verify
  (7B, 28 layers, ~30 ms): cost tracks **layer count, not parameters**. Draft
  forward over 1 vs 5 tokens is ~24 vs ~25 ms → dominated by **fixed
  per-forward kernel-launch overhead**. HBM floor for 0.5B ≈ 0.5 ms (1 GB ÷ ~2
  TB/s) vs 24 ms measured ⇒ **launch-bound, not memory-bound.**
- **SDPA does not fix it (T3.4b).** Fused-attention draft still **≈25 ms/token**
  → the overhead is whole-forward kernel dispatch (projections, MLP, norms,
  RoPE, residuals), not the attention kernels. The fix requires CUDA-graph /
  compiled capture, not an attention-backend flag.
- **The STOP is harness-specific, not fundamental.** CPU cost-sensitivity on the
  same sealed match vectors: at a hypothetical draft cost of 8 / 4 / 2 ms/token
  the headroom is **46% / 38% / 26%** and best fixed action moves to L=2/6/8. So
  the routing opportunity is real; it is masked by launch-bound eager execution.
  Consistent with D014 (eager is capture-friendly, not production-absolute).
- **Impact on claims:** C06 (and C05) net-latency advantage is **harness-
  dependent** — limit added to the C06 row. C04 (acceptance atlas) is
  **latency-independent and unaffected**: 38 category×phase cells, prompt-
  bootstrap CIs, `code_delimiter`/`operator`/`number` ≈0.85–0.88 vs
  `named_entity`/`reasoning_transition` ≈0.58–0.67 (label = counterfactual
  `target_match`). Surface baseline ladder (I13): `surface_stack` AUROC 0.84,
  pre-draft `preround_hardened` 0.73 — the bar any hidden-state probe (C01) must
  beat. These stand regardless of the timing result.
- **Data-quality note (recoverable, no re-run):** sealed
  `RequestSummary.split` is all `"unknown"` — the sweep stamped
  `assignment.get(prompt_id)` but the split map is keyed by `prompt_hash`
  (modal_app.py:276). Recovered at analysis time via prompt_hash + split
  manifest. Atlas/baselines/oracle unaffected. Fix the stamping and re-stamp
  before publishing the dataset (T6.1).
- Follow-up: measurement-only dual-mode harness authorized (D021) to characterize
  the deployed-regime headroom and unblock an honest M3 re-decision.
- **Dual-mode (D021) measured — no drop-in flag fixes the launch-bound draft:**
  `t3_4_bench_sdpa_default.json` and the reduce-overhead attempt.
  (a) `torch.compile(mode="reduce-overhead")` (CUDA graphs) **errors at runtime** —
  "accessing tensor output of CUDAGraphs that has been overwritten by a subsequent
  run" (in `apply_rotary_pos_emb` / `get_seq_length`): CUDA-graph static output
  buffers are incompatible with HF `DynamicCache`, whose KV tensors persist and
  grow across steps. (b) `torch.compile(mode="default")` (inductor fusion, no CUDA
  graphs): draft **≈31 ms/token — no improvement over eager (~24 ms), actually
  worse**, and the run trips `torch._dynamo` `cache_size_limit` (8) because the
  variable per-step sequence length forces constant recompilation; the resulting
  cost profile is polluted (skip verify 38.7 ms vs L=1 verify 16.8 ms — internally
  inconsistent), so its 13–16% "headroom" is a recompilation artifact, not a real
  gain. **Conclusion:** on the current dynamic-cache engine, neither fused
  attention nor either compile mode circumvents launch-bound. The real fix is a
  fixed-shape **`StaticCache`** decode path — which simultaneously enables stable
  compilation (one shape) and CUDA-graph replay — i.e. serving-grade plumbing,
  deferred to Tier-2/G4 (D009/D010). The dual-mode seam itself works and is in
  place; it just has nothing to compile *to* until a static-cache path exists.
- Logged by Claude, 2026-07-12.

## Low-hanging-fruit analyses (LHF #1-6), offline on sealed fixed_8 traces

Six CPU-only analyses over the sealed Qwen-v1 (`sweep-2026-07-11T203836`, 19,074
test rounds) and Llama (`sweep-llama-f8-2026-07-13`, 15,500 test rounds) fixed_8
counterfactual traces. All numbers script-generated (`scripts/lhf_analysis.py`,
`modal_app.py::lhf`), dev-tuned / test-reported, serving-cost efficiency
(draft priced 0.1x verify). Both families give the SAME qualitative picture;
Llama's absolute efficiency is higher (3.17 vs 2.67 tok/round-cost).

- **#2 Calibration-optimal stopping (positive, sharpens RQ2).** On a fine tau
  grid (0.1-6.0) the entropy-stop optimum is tau*=2.0 on dev for BOTH families,
  and the RQ2 headline tau=2.0 is within 0.0% of that test optimum. So tau=2.0 is
  not a lucky pick, it is the calibrated optimum. P(accept | draft entropy) is
  monotone decreasing (Qwen 0.969 at entropy<=0.25 down to 0.16 at >6; Llama
  0.993 down to ~0.35), i.e. draft entropy is a well-calibrated acceptance
  predictor. Per-position economic breakeven is p_accept=0.091 (=0.1/1.1);
  the block-level entropy stop beats the myopic per-position rule because one
  high-entropy position predicts the rest of the block also fails.
- **#1 Pre-round gate / C10 (bounds the pre-round-only story).** Choosing L
  before drafting from the PREVIOUS round's frontier entropy (dev-tuned bin->L
  map) beats best-fixed by only +3.54% (Qwen) / +1.13% (Llama), i.e. it recovers
  ~1/3 to ~1/2 of the within-round entropy-stop gain (-6.89% / -5.20% vs
  within-round). The dev-tuned map is sensible (high frontier entropy -> shorter
  L). **The bulk of the controllable headroom needs the ONLINE within-round draft
  signal, not a pre-round prediction.** skip_frac=0: the pre-round gate never
  elected a lossless full skip -> no purchase for a pre-round skip predictor
  (consistent with the ~2% routing headroom).
- **#5 Skip economics (negative, kills learned-skip).** Under 0.1x serving cost,
  allowing L=0 (pure-AR fallback) slightly HURTS: eff_with_skip < eff_no_skip by
  -0.71% (Qwen) / -0.31% (Llama), despite the tau=2 controller electing skip
  22.6% / 16.2% of rounds. The draft is cheap enough that always drafting >=1
  dominates; a learned skip gate is not a source of gains.
- **#3 Headroom attribution (negative for content-static).** A category-clairvoyant
  static length lookup (know the upcoming token's category, use its dev-optimal L)
  captures only 12.5% (Qwen) / 4.4% (Llama) of the entropy-stop gain over
  best-fixed. **~88-96% of RQ2's win is carried by the dynamic entropy signal, not
  by token-category identity** -> reinforces the RQ3 no-go (domain/category routing
  has little headroom).
- **#6 Block-breaker atlas (descriptive).** The first rejected token in a block is
  overwhelmingly a leading-space word token: whitespace-prefixed 77%, content_word
  50%, function_word ~30% of breaks (multi-labeled) in BOTH families; punctuation,
  numbers, named entities, code delimiters are each <14%. Block termination is
  driven by ordinary lexical divergence at word starts, not by structural/boundary
  tokens.
- **#4 Zero-shot controller transfer (positive).** Qwen's dev-optimal tau (2.0)
  applied to Llama is within 0.0% of Llama's OWN dev-optimal tau (also 2.0): the
  entropy-stop controller transfers cross-family with no re-tuning. The
  calibration curve, block-breaker profile, and attribution all replicate across
  Qwen and Llama.
- **Unifying result:** essentially all controllable speculative-decoding headroom
  lives in the ONLINE, within-round draft-entropy signal. Pre-round, category-
  static, and skip alternatives each recover only a small fraction (or are
  negative), and the tau=2.0 stop is provably calibration-optimal and transfers
  zero-shot across model families.
- Logged by Claude, 2026-07-13. Artifacts: volume
  `analysis/lhf_sweep-2026-07-11T203836.json`, `analysis/lhf_sweep-llama-f8-2026-07-13.json`.

## RQ2 + LHF replication on corpus v2 (third corpus, positive; magnitude corpus-dependent)

Ran the RQ2 bootstrap CI and the six LHF analyses on the sealed v2 fixed_8 sweep
(`sweep-v2-f8-2026-07-13`, Qwen pair, corpus v2 = 1,494 prompts / 7 axes, 5,926
test rounds / 222 test prompts). Script-generated (`modal_app.py::rq2ci`, `::lhf`);
artifacts `analysis/sweep-v2-f8-2026-07-13/rq2_ci_test.json`,
`analysis/lhf_sweep-v2-f8-2026-07-13.json`.

- **RQ2 headline replicates, smaller magnitude.** entropy-stop (tau=2.0, dev-frozen)
  vs best fixed: eff 2.9073 vs 2.7047 = **+7.49%, 95% CI [+6.04%, +9.06%],
  P(delta<=0)=0/2000** (prompt-grouped, 222 prompts). Wasted draft tokens cut
  ~49% (stop 0.432 [0.402, 0.464] vs fixed 0.849 [0.769, 0.936], disjoint).
  Three-corpus range now **+6.7% (Llama), +7.5% (v2), +11.2% (v1)** - always
  positive; the broader/more-diverse v2 gives a smaller relative gain (more of the
  corpus is already high-acceptance). tau grid optimum on v2 is tau*=1.8 (dev);
  tau=2.0 is within -0.24% of it.
- **The unifying result gets STRONGER on the diverse corpus.** Category-clairvoyant
  static length lookup captures only **0.2%** of the entropy-stop gain over
  best-fixed on v2 (vs 12.5% Qwen-v1 / 4.4% Llama) - i.e. ~99.8% of the win is the
  online within-round entropy signal, essentially none is token-category identity.
  Pre-round gate +0.82% vs best-fixed / -6.21% vs online; skip hurts (-0.37%);
  calibration monotone (0.976 at H<=0.25 -> 0.101 at H>6); block-breakers =
  whitespace 74.6% / content_word 47.1% / function_word 26.1% (lexical word-starts).
  All six LHF findings replicate on the third corpus.
- **Bug fixed in passing:** `rq2_ci` / `eval_policies` / `taxonomy` wrote to
  `/artifacts/analysis/{run_id}/...` without creating the subdir, so the first v2
  `rq2ci` raised FileNotFoundError (v1 worked only because a prior capture had made
  the dir). Added `os.makedirs(..., exist_ok=True)` before all three writes;
  re-ran and it sealed.
- Logged by Claude, 2026-07-13.

## Autoresearch pre-round signal (I13/I23/C10) + causal validation (I15), 2026-07-22

Generator-critic autoresearch loop (D023). Method + full numbers in
`docs/autoresearch_outcomes.md` and `docs/causal_intervention_report.md`; every
number script-generated from immutable artifacts.

- **Predictive (dev, domain-controlled, replicated).** A pre-round FIRST-TOKEN
  acceptance signal from the target's cached verified-context frontier
  representation beats `preround_hardened + domain` (entropy+margin+history+domain)
  by **+0.072 (Qwen-v1, 4 domains), +0.112 (Qwen-v2, 7 domains), +0.069 (Llama)**,
  all CI-clean, beating equal-capacity random/norm-matched controls. Replicated
  across two model families AND two corpora. Near-zero deployed cost (G3 microbench:
  ~16 µs probe, 0.015% of a round). Only the full representation clears the bar;
  cheap variants (lowrank/norm/align/drift) are marginal/inconsistent.
- **Negative / boundary (recorded per AGENTS.md).** FIRST-TOKEN only: per-length
  survival probes give null-to-significantly-worse lift at k≥4 on all three
  settings; a length-aware controller does not benefit (mostly hurts). Does not
  extend to run-length or length-selection — differentiates from AdaEAGLE (which
  sets length from features).
- **Corrected exclusion.** An initial capture-sampling bug (`cap_prompts` + sorted
  truncation) captured only ~2 domains/run, inflating v2 via a weak
  summarization-dominated baseline (base AUROC 0.60) and making domain-control
  vacuous. Fixed with domain-stratified sampling (D025); re-captured (4/7/4 domains)
  and re-run — the finding survived, confirming it is NOT domain identification. The
  pre-fix v2 "stronger" reading was an artifact.
- **Reproducibility.** The exploratory `c_reg=1.0` fit was overfit/non-convergent
  (AUROC wobbled run-to-run); `c_reg=0.1` (D025) converges the strictly-convex probe
  to a unique, thread-independent optimum and RAISES the lift (v1 k=1 +0.047→+0.074).
- **Causal (I15, held-out test rounds, replicated).** Forward-hook steering of the
  first-token acceptance direction disrupts acceptance **~2–10× more than
  norm-matched random/shuffled controls**, dose-dependently (peak-at-0), beyond the
  induced entropy change, at all 4 layers, on BOTH Qwen-v1 (`sealed_fidelity` 0.95)
  and Llama (1.00). The two empirical G2 criteria are met. Honest scope:
  representation-level causal control of the target's next-token agreement-ability,
  NOT a draft–target "circuit"; the LANGUAGE upgrade is a human gate (D020).
- **C10 status:** dev-strong + causally validated, but the frozen PREDICTIVE test
  pass is pending; C10 stays `UNTESTED` for the frozen table until then.
- Logged by Claude, 2026-07-22.

### 2026-07-22 — C10 frozen predictive TEST pass: 3/3 settings PASS (C10 → SUPPORTED, first-token scope)

The deliberate one-shot frozen test pass for the pre-round first-token acceptance
signal (I13/I23, D023/D025). Protocol pre-registered before unblinding: candidate
= `raw_frontier` ONLY (the sole dev survivor; single-spec scoring avoids multiple
comparisons on test), layers 6/12/18/24, `c_reg=0.1`, `domain_control=True`
(baseline = `preround_hardened + domain`), seed 0, prompt-grouped OOF +
equal-capacity norm-matched/random controls + prompt-grouped bootstrap CI —
machinery identical to dev (cas.autoresearch.eval, unchanged). Pass criterion per
setting: Δauroc > 0, CI-clean (p(Δ≤0)<0.05), beats controls; replication = all
three settings. Artifacts: `analysis/<run>/autoresearch_test_domctl.json`.

| setting | n test rows | base | combined | Δauroc | 95% CI | p(Δ≤0) | ctrl AUROC |
|---|---|---|---|---|---|---|---|
| Qwen-v1 (4 dom) | 11,982 | 0.6593 | 0.7348 | **+0.0755** | [+0.0625, +0.0884] | 0.000 | 0.550 |
| Qwen-v2 (7 dom) | 7,191 | 0.6246 | 0.7165 | **+0.0918** | [+0.0727, +0.1108] | 0.000 | 0.534 |
| Llama (4 dom) | 9,610 | 0.7126 | 0.7669 | **+0.0542** | [+0.0386, +0.0708] | 0.000 | 0.559 |

- **PASS 3/3.** The held-out test lifts match dev closely (dev +0.072/+0.112/+0.069)
  — no dev-overfitting signature. Equal-capacity random/norm controls sit at
  ~0.53–0.56 (below base): the lift is signal, not capacity. Platt-calibrated ECE
  stays at baseline level (comb 0.016–0.018) — the ranking lift is decision-usable,
  not a calibration artifact. `credible_systems` (AUROC-CI-clean + CI-robust
  regret reduction over ≥2 costs) fires on all three (CI-robust at cost_draft
  1.0–9.0 on both Qwen settings, 2.0–4.0 on Llama; still no help in the cheap-draft
  regime ≤0.2 — unchanged boundary).
- **Integrity disclosure:** a pre-existing `autoresearch_test_domctl.json` for
  Qwen-v2 (full 5-spec library) was found on the volume before this pass — the v2
  test split had been evaluated at least once earlier without a repo record
  (likely a stray `--eval-split test` during the autoresearch loop). Its
  `raw_frontier` numbers (+0.0906 [+0.0709,+0.1110]) agree with the pre-registered
  re-run (+0.0918) to ~0.001 (earlier code state); conclusion unchanged.
  Candidate/hyperparameter selection was documented as dev-only (D025,
  autoresearch_outcomes.md) before this pass, so selection is untainted; the v1
  and Llama test files did not exist before today.
- **C10 → SUPPORTED, scope-limited:** (a) FIRST-TOKEN acceptance only — per-length
  survival is null-to-harmful at k≥4 (dev, 3/3; ledger note above), so no
  run-length or length-selection claim; (b) only the FULL frontier representation
  clears the bar (cheap variants marginal — deployable-cheap variant open);
  (c) marginal cost from the G3 microbench (~16 µs probe, 0.015% of a round,
  dev-measured) — a full serving-path wall-clock claim stays G3-gated;
  (d) counterexamples/limits vs AdaEAGLE / Judge Decoding / WhiFlash unchanged
  (see C10 row); novelty re-check (I21) still due before manuscript freeze.
- Logged by Claude, 2026-07-22.

### 2026-07-22 (later) — C10 frozen pass REDONE + strict dev→test transfer: 3/3 PASS under both protocols

Owner-directed redo of the frozen pass, plus a STRICTER protocol added the same
day. Artifacts: live `autoresearch_test_domctl.json` (redo) and NEW
`autoresearch_frozen_transfer_domctl.json` per run; the first-pass and the
pre-existing v2 files are preserved under `analysis/<run>/archive/`.

**(1) Redo of the pre-registered within-test-OOF pass — conclusions reproduce;
numeric wobble bounded.** Same code, same artifacts, fresh containers:

| setting | pass-1 Δauroc (archived) | redo Δauroc | conclusion flags |
|---|---|---|---|
| Qwen-v1 | +0.0755 | +0.0761 | identical (win, controls, credible 3/3) |
| Qwen-v2 | +0.0918 | +0.0918 | identical |
| Llama | +0.0542 | +0.0530 | identical |

Reported AUROC reproduces to **~±0.001 across containers** (BLAS reduction-order
noise in the 14k-feature lbfgs fit; binned ECE wobbles more, e.g. v1 combined
0.018→0.024). This is ~10× smaller than the CI half-widths (~±0.013), so no
conclusion is sensitive to it. **Correction to the earlier integrity note:** the
pre-existing v2 file's +0.0906 vs the pre-registered +0.0918 is consistent with
this same container wobble — the "earlier code state" attribution was
over-specific. **D025 refinement:** `c_reg=0.1` gives a unique optimum at the
objective level, but the reported AUROC is reproducible to ~±0.001 across
containers, not bit-identical.

**(2) NEW strict frozen dev→test transfer (`score_spec_frozen` /
`frozen_transfer_lift`, commits cb2392d+06c62d2).** The within-test-OOF protocol
never *selects* on test but does *fit* fold models on test rows; the strict
protocol fits NOTHING on test — scaler, probe, equal-capacity controls,
imputation means, domain vocabulary, and Platt calibrator are all dev-frozen and
the untouched test rows are scored exactly once. Pre-registered criterion
unchanged (Δauroc>0, grouped-CI p(Δ≤0)<0.05, beats controls, 3/3):

| setting | n_dev→n_test | base | combined | Δauroc | 95% CI | p(Δ≤0) | ctrl | credible_sys |
|---|---|---|---|---|---|---|---|---|
| Qwen-v1 | 11,586→11,982 | 0.6603 | 0.7158 | **+0.0555** | [+0.0414, +0.0700] | 0.000 | 0.542 | **no** (robust at cost 2.0 only) |
| Qwen-v2 | 6,931→7,191 | 0.6259 | 0.7285 | **+0.1027** | [+0.0830, +0.1242] | 0.000 | 0.528 | yes (1.0–9.0) |
| Llama | 9,659→9,610 | 0.7129 | 0.7523 | **+0.0393** | [+0.0225, +0.0565] | 0.000 | 0.559 | yes (2.0, 4.0) |

- **Predictive PASS 3/3 under the strictest protocol** — the C10 `SUPPORTED`
  status stands on the stronger footing. Deltas move v1 −0.021 / Llama −0.014 /
  v2 +0.011 vs within-test OOF — small shifts in both directions, no
  overfitting signature.
- **Honest boundaries surfaced by the strict protocol:** (a) on Qwen-v1 the
  calibrated decision-regret proxy is CI-robust at only ONE cost (2.0), so the
  `credible_systems` flag fails there — the decision-proxy robustness is
  protocol-sensitive; the authoritative systems claim was and remains G3-gated
  wall-clock, untouched by this. (b) Calibration transfer is imperfect: combined
  test ECE 0.021–0.037 (dev-OOF-trained Platt applied to a full-dev-fit score
  distribution — mismatch documented in `frozen_transfer_lift`'s docstring);
  ranking metrics are unaffected.
- Unit tests: `tests/test_autoresearch_frozen.py` (5 tests: transfer detects a
  real signal, noise fails its dev-frozen control, determinism, Platt-helper
  equivalence, dev-frozen imputation/vocabulary), plus the existing eval tests
  unchanged — `PYTHONPATH=src python -m pytest tests/test_autoresearch_frozen.py
  tests/test_autoresearch_eval.py -q`.
- Logged by Claude, 2026-07-22.

### 2026-07-22 — I21 novelty re-check for the C10 freeze: cell UNOCCUPIED (adversarially-verified sweep)

Multi-agent verified literature sweep (100 agents; 24 claims, all surviving 3-0
adversarial verification; 12 papers verified on primary arXiv pages, most
full-text). Full per-paper table + criterion matrix in `docs/landscape.md`
§"I21, 2026-07-22".

- **Verdict: the C10 cell is unoccupied as of 2026-07-22** — no verified work
  combines pre-round timing × target cached-frontier representation × lossless
  exact-match × calibration × control vs free entropy/margin/history × independent
  drafter × skip decision × first-token target. Coverage = 12 papers verified this
  round ∪ the four adjacent works primary-verified in the earlier I21 rounds
  (WhiFlash, SemanticSpec, 2604.14682, Not-a-Bandit — no surviving claims in this
  round's harness, an attrition artifact, not a refutation; the 2026-07-10/11
  in-repo verifications stand).
- **AdaEAGLE (2412.18910) confirmed as the primary differentiation target, 3/8
  criteria:** LDLP consumes exactly C10's signal class and timing (verbatim §3.2:
  embedding + final-layernorm target hidden state at the last validated-prefix
  token, pre-draft) but predicts LENGTH via uncalibrated scalar L1 regression, is
  deliberately logit-free (no entropy/margin baselines anywhere), has no skip
  action, and uses the EAGLE trained head. Length-classification is a losing
  ablation in their own §4.3.2 — convergent with our finding that the frontier
  signal is first-token, not length.
- **Theory grounding (2606.30265):** margin certificates for strict greedy
  formalize the top-1/top-2 margin as carrying the governing variable → C10's
  margin-baseline control is theory-mandated. Open manuscript question logged: does
  the certificate imply a ceiling on probe headroom over the margin baseline?
- **New rows added to landscape.md:** HVD (2606.01019), SpecLA (2607.16673, v1
  18 Jul 2026 — newest in window), Speculative Verification (2509.24328), AdaSpec
  (2503.05096, ex-SpecServe; name-collision warning vs 2510.19779). All fail the
  cell on ≥5 criteria each.
- **Caveats:** citing-papers sweeps for AdaEAGLE/Judge/WhiFlash produced no
  surviving claims — a very recent citer could exist unseen; **re-run the citing
  sweep at manuscript freeze**. DSpark head-input wording differs between our two
  full-text reads (hidden-states vs confidence features) — re-quote before quoting;
  not decisive for C10. One refuted claim recorded for transparency: AdaSpec does
  NOT articulate the pre-round gap (0-3) — never cite it as naming C10's gap.
- Logged by Claude, 2026-07-22.

### 2026-07-22 — C04 atlas dev pass (3 settings) + PRE-REGISTRATION for the frozen test pass

Split-aware, domain-controlled atlas (`atlas_c04`, commit 80ee48b) on the DEV
split of all three settings (Qwen-v1 149,872 tokens / Qwen-v2 207,256 / Llama
123,904; labels = counterfactual per-position `target_match`; categories
v1.0.0 overlapping, phases v1.0.0 absolute bins; prompt-grouped bootstrap
CIs). Artifacts: `analysis/<run>/c04_atlas_dev.json`.

**Dev findings.** (a) Domain marginals reproduce the domain-grain axis of
arXiv:2604.14682 (control achieved; e.g. v1 summ 0.657 < chat 0.700 < code
0.813 < math 0.875). (b) The pre-specified structural-vs-entity contrast
(HIGH = code_delimiter/operator/number pooled vs LOW =
named_entity/reasoning_transition pooled) is positive and CI-clean OVERALL in
all three settings (+0.190 / +0.219 / +0.089) and WITHIN 14 of 15 domain
cells (all 4 v1, all 7 v2, Llama chat/math/summ); the one null is Llama-code
(−0.006 [−0.063, +0.037], base acceptance 0.913 — ceiling). (c) The category
ordering's tails are cross-family stable (high: code_delimiter/operator/
number/clause_boundary/punctuation; named_entity below the marginal in all
three); reasoning_transition is low on Qwen only. (d) **Phase axis is
null-to-inconsistent:** flat on both Qwen settings (max−min ≤ 0.015,
overlapping CIs); monotone increasing on Llama only (0.788→0.827→0.849) — no
cross-family phase structure. (e) Descriptive anomaly, not a claim: Llama
`special` accepts at 0.007 (n=883) — likely chat-template/special-token
handling divergence between the pairs; investigate separately.

**PRE-REGISTERED test criteria (fixed before unblinding test; evaluated
per-setting by script over `c04_atlas_test.json`):**
- **T1 (overall contrast):** HIGH−LOW contrast positive, p(Δ≤0)<0.05, all 3
  settings.
- **T2 (beyond domain):** within-domain contrast positive + CI-clean in all 4
  Qwen-v1 domains, all 7 Qwen-v2 domains, and Llama chat/math/summ.
  **Llama-code pre-registered as expected-null:** pass requires its test CI to
  include 0 or sit positive (no significant negative reversal).
- **T3 (tail stability):** in each setting, each HIGH category's test rate ≥
  the setting's overall test marginal; named_entity's ≤ the marginal.
- **T4 (phase, expected null):** on both Qwen settings, phase spread
  (max−min) ≤ 0.03; no cross-family phase claim regardless of outcome (the
  Llama monotone trend is recorded descriptively; potential length/mix
  confound).
- **Outcome mapping:** T1–T3 pass 3/3 → C04 → SUPPORTED for the CATEGORY axis,
  scope-limited (phase axis null/inconsistent; Llama-code ceiling exclusion;
  overlapping labels v1.0.0; fixed_8 counterfactual labels; 2 families, 2
  corpora). Any failure → PARTIAL with the specific failing cells recorded.
- Logged by Claude, 2026-07-22 (before the test unblinding).

### 2026-07-22 — C04 frozen TEST pass: PASS 3/3 on all pre-registered criteria (C04 → SUPPORTED, category axis)

One-shot test evaluation (`c04_atlas_test.json` × 3), verdicts computed by
script against the pre-registered T1–T4 exactly as committed in 968c425.

| criterion | Qwen-v1 (152,592 tok) | Qwen-v2 (209,160) | Llama (124,000) |
|---|---|---|---|
| T1 overall contrast | **+0.1947** [+0.172,+0.216] p=0 | **+0.2104** [+0.193,+0.228] p=0 | **+0.1177** [+0.103,+0.134] p=0 |
| T2 within-domain | 4/4 CI-clean | 7/7 CI-clean | 3/3 CI-clean + code |
| T3 tail stability | PASS (marg 0.742) | PASS (0.688) | PASS (0.833) |
| T4 phase spread (Qwen ≤0.03) | 0.0084 PASS | 0.0036 PASS | 0.0589 (descriptive) |

- **PASS 3/3 → C04 → SUPPORTED, category axis, scope-limited.** The
  structural-vs-entity acceptance structure (code_delimiter/operator/number ≫
  named_entity/reasoning_transition) is real, domain-controlled, and replicates
  across both model families and both corpora on frozen test data.
- **The pre-registered expected-null reversed positive:** Llama-code came in
  **+0.0674 [+0.0304, +0.1097]** — the dev null was noise at a 0.913 acceptance
  ceiling, and the criterion (no negative reversal) is exceeded: 15/15 domain
  cells are CI-clean positive on test.
- **Weakest cell, reported:** Qwen-v1 summ +0.0544, p(Δ≤0)=0.027 (percentile CI
  lo −0.002) — passes the pre-registered p<0.05 criterion but is the margin to
  watch in any re-run.
- **Phase axis outcome:** decisively flat on Qwen (test spreads 0.8pp/0.4pp —
  tighter than the 3pp criterion); Llama monotone (prefix 0.794 → mid 0.824 →
  late 0.853) recorded descriptively only. C04's "generation phases" half is
  narrowed out of the claim; absolute-position phase bins carry no
  cross-family acceptance structure at this grain.
- Consequences: I18's atlas-evidence half is done (figures remain); the atlas +
  contrast tables are the natural core of the companion analysis manuscript.
  C04 claim language stays representation-free (no G2 implications — this is
  behavioral structure, not internal-state evidence).
- Logged by Claude, 2026-07-22.
