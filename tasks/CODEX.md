# Parallel task assignment — Codex

You are **Codex**, working concurrently with **Claude** and **Grok** in this
repository. This file is your work order. It does not override the repository
contract; `AGENTS.md` binds you exactly as it binds every agent.

## 0. Read first (do not skip)

1. `AGENTS.md` — the operating contract.
2. `docs/RESEARCH_SPEC.md`, `docs/EXPERIMENT_CONTRACT.md`, `docs/DECISIONS.md`
   (especially D008 naming, D009 gates, D014 engine, D015 tooling).
3. `PLAN.md` §3 (competitive landscape — your baselines have published sources).
4. `docs/ISSUE_BACKLOG.md` — your issues are **I08** and **I09**; acceptance
   criteria are there. Set their status to `IN_PROGRESS` (owner `Codex`) as your
   first commit.

## 1. Session state you are joining (2026-07-10)

- **Steps 1–2 are done and proven.** Exact greedy speculative decoding is
  token-identical to target-only greedy (fp32 equivalence gate: 25/25). The
  engine lives in `src/cas/spec_decode.py`; the pure accept/rollback math in
  `src/cas/commit.py`.
- **Data is frozen.** Prompt-grouped dev/test splits over 644 prompts (code 82,
  math 100, chat 40, summ 100 per side) via `src/cas/data/`.
- **Trace writer (I06) is Claude's in-progress work** in `src/cas/trace/`. Do not
  touch it; you will consume its records only through the interfaces below.
- Primary pair: `Qwen/Qwen2.5-7B-Instruct` (target) / `Qwen/Qwen2.5-0.5B-Instruct`
  (draft), revisions pinned in `src/cas/config.py`.

## 2. Your scope — I08 and I09 (contract policies 5, 6, 7)

Implement the baseline **length-control policies** as pure, unit-tested Python.
No GPU, no model loading, no trace files needed for your unit tests — these are
functions of the per-round context and cheap signals.

### I09 — Bandit length policy (contract policy 7)

- Match a documented **BanditSpec-style** formulation (arXiv:2505.15141) that
  treats the per-round draft length as the arm; **record every deviation** from
  the published formulation in a module docstring and in `docs/CLAIMS_LEDGER.md`.
- Report **cold-start and steady-state behavior separately** (your tests should
  demonstrate both regimes).
- Fits the existing engine interface directly (see §3): a stateful callable
  `RoundContext -> L`.

### I08 — Simple adaptive policies (contract policies 5, 6)

- **Recent-acceptance policy:** choose `L` from a running acceptance-rate state.
  Fits the `RoundContext -> L` interface directly.
- **Entropy stop rule (SVIP-style, arXiv:2411.18462):** stop drafting when the
  draft's next-token entropy exceeds a threshold. This is a *per-token* rule, so
  it uses the separate `StopRule` interface in §3 (Claude owns wiring it into the
  engine draft loop; you deliver the tested rule object).
- **Freeze test-time hyperparameters** (thresholds are constants chosen on dev
  data, not tuned at test). **Unit-test threshold boundaries and state reset
  between requests** — a policy must fully reset when a new request starts.

## 3. Interface contract (stable — code against exactly this)

From `src/cas/spec_decode.py` and `src/cas/config.py` (read-only for you):

```python
# cas.config
ACTION_LENGTHS: tuple[int, ...] = (0, 1, 2, 3, 4, 6, 8)   # 0 == skip
# cas.spec_decode
ActionPolicy = Callable[[RoundContext], int]              # returns L in ACTION_LENGTHS
@dataclass
class RoundContext:
    round_id: int
    generated_so_far: int
    last_round: RoundTrace | None      # cas.trace.records.RoundTrace (has
                                       # accepted_prefix_len, realized_draft_len,
                                       # draft_entropy, draft_top1_margin, ...)
```

For the entropy stop rule, implement against this signature (Claude will add the
matching consult-point in the engine; **do not edit the engine yourself**):

```python
@dataclass
class StopContext:
    draft_index: int            # 0-based position within the current draft
    cur_entropy: float          # entropy of the token about to be proposed
    cur_margin: float           # top1-top2 margin of that token
    proposed_so_far: tuple[int, ...]
StopRule = Callable[[StopContext], bool]   # True -> stop drafting now
```

Put `StopContext`/`StopRule` in your package and add a one-line note to
`docs/DECISIONS.md` proposing them as the engine's stop-rule seam; Claude will
ratify and wire it. Policies must be **stateful callable objects** exposing a
`reset()` method, not closures, so per-request reset is testable.

## 4. Where your code goes (your files — no collisions)

- `src/cas/policies/__init__.py`
- `src/cas/policies/recent_acceptance.py`   (I08)
- `src/cas/policies/entropy_stop.py`         (I08)
- `src/cas/policies/bandit.py`               (I09)
- `tests/test_policies.py`                   (all of the above)

**Off-limits (Claude/Grok own these):** `src/cas/trace/`, `src/cas/spec_decode.py`,
`src/cas/commit.py`, `src/cas/config.py`, `src/cas/models.py`, `src/cas/timing.py`,
`src/cas/data/`, `src/cas/signals.py`, `modal_app.py`, `src/cas/annotate/`,
`docs/landscape.md`. If you need a change in any of these, request it via a note
in `docs/DECISIONS.md` — do not edit them.

## 5. Environment / how to test

- **Local env only** — your work needs no GPU and no Modal. Run:
  `PYTHONPATH=src python -m pytest tests/test_policies.py -q`
- The local base conda torch is broken/old; **do not import torch** in your
  policy modules or tests. Signals arrive as plain floats via `RoundContext` /
  `StopContext`, so pure-stdlib policies are both possible and required (matches
  `cas.commit`, which is deliberately torch-free).
- GPU/Modal end-to-end runs are **Claude-driven and metered**; do not launch
  them. Integration of your policies into the decode loop happens after I06.

## 6. Hard rules (from AGENTS.md — non-negotiable)

- Never present synthetic/illustrative values as results; record cold-start and
  steady-state behavior from **your unit tests**, not hand-picked numbers.
- **Naming (D008):** cite prior work by arXiv ID/title only. Do **not** name any
  conference, journal, external lab/organization, or owner goals anywhere.
- Record deviations from the published bandit formulation and any negative
  findings in `docs/CLAIMS_LEDGER.md`.
- **Definition of done (AGENTS.md §"Definition of done"):** acceptance criteria
  met; `tests/test_policies.py` passes; a reproducible command documented in the
  module or a short `README`; backlog status + affected claims updated; interface
  additions (`StopRule`) recorded in `docs/DECISIONS.md` before downstream use.

## 7. Coordination

- Set I08/I09 to `IN_PROGRESS` (owner `Codex`) before you start; `DONE` when the
  definition of done is met.
- Do not rewrite Claude's or Grok's files. Reconcile any interface conflict
  through a dated `docs/DECISIONS.md` note.
- Re-verify any repository fact before asserting it in a durable doc — files
  change under you as the other two agents work.
