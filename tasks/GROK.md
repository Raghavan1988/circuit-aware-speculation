# Parallel task assignment — Grok

You are **Grok**, working concurrently with **Claude** and **Codex** in this
repository. This file is your work order. It does not override the repository
contract; `AGENTS.md` binds you exactly as it binds every agent.

## 0. Read first (do not skip)

1. `AGENTS.md` — the operating contract.
2. `docs/RESEARCH_SPEC.md`, `docs/EXPERIMENT_CONTRACT.md`, `docs/DECISIONS.md`
   (especially **D008 naming/disclosure**, D009 gates).
3. `PLAN.md` §3 (competitive landscape — the starting point for I21).
4. `docs/ISSUE_BACKLOG.md` — your issues are **I21** and **I11**; acceptance
   criteria are there. Set their status to `IN_PROGRESS` (owner `Grok`) first.

## 1. Session state you are joining (2026-07-10)

- Steps 1–2 (exact decoding + equivalence) are done and proven (fp32 gate 25/25).
- Data is frozen: prompt-grouped dev/test splits over 644 prompts (code 82,
  math 100, chat 40, summ 100 per side) via `src/cas/data/`. You can regenerate
  the prompt list locally with `PYTHONPATH=src python -c "from cas.data.ingest
  import ingest_all; rows=ingest_all(); print(len(rows))"` (CPU, downloads public
  datasets; no GPU).
- Claude owns the trace writer (I06, `src/cas/trace/`); Codex owns the baseline
  policies (I08/I09, `src/cas/policies/`). Stay out of both.

## 2. Your scope

### I21 — Landscape verification + living comparison table (research; web)

- **Verify against primary sources** the four planning-pass additions in
  `PLAN.md` §3 / backlog I21: arXiv:2603.01639 (adaptive drafting via RL),
  arXiv:2605.02888 (draft confidence under KV compression), arXiv:2604.14682
  (task-conditioned acceptance dynamics), arXiv:2606.30265 (theory of
  acceptance). One was found via a **mirror site** and needs primary-archive
  (arXiv) confirmation — flag which, with the resolved link.
- **Re-scan** for newer adjacent work (adaptive/learned/bandit speculation,
  acceptance prediction, per-request draft control, ragged verify). Note anything
  that overlaps the thesis, especially claim **C04** (token-category/phase
  acceptance) and the pre-round-prediction bet (C10, I23).
- Maintain a **living comparison table** in `docs/landscape.md` (already started):
  columns for what each work adapts, its signal, whether it is mechanistic, and
  its relevance/contrast here. Keep `PLAN.md` §3 and `docs/landscape.md`
  consistent (landscape.md is authoritative and yours; propose PLAN.md edits via
  a note rather than large rewrites, since PLAN.md is Claude's).
- **Record the impact on claims** (notably C04, and any novelty threat) in
  `docs/CLAIMS_LEDGER.md` **before** any novelty claim is frozen.

### I11 — Token-category annotation (CPU code)

- Implement **versioned, overlapping** token categories plus a **generation-phase**
  label, per `docs/TRACE_SCHEMA.md` (token trace: "overlapping token-category
  labels and generation-phase label") and RESEARCH_SPEC's acceptance-atlas needs.
- **Preserve ambiguity** — categories may overlap; do **not** force mutually
  exclusive labels. Emit a *set* of labels per token, plus one phase label.
- **Validate a stratified manual sample** — draw a stratified sample across the
  four domains, hand-check labels, and report agreement; keep the sample and the
  check reproducible (scripted, seeded).
- Version the category set (e.g. `CATEGORY_SET_VERSION`) so later schema changes
  are traceable (TRACE_SCHEMA invariant 7).

## 3. Where your code/docs go (your files — no collisions)

- `docs/landscape.md`                         (I21 — living table; yours)
- `src/cas/annotate/__init__.py`
- `src/cas/annotate/categories.py`            (I11 — overlapping categories)
- `src/cas/annotate/phases.py`                (I11 — generation-phase label)
- `tests/test_annotate.py`                    (I11 — incl. the stratified check)
- Ledger/backlog updates as required.

**Off-limits (Claude/Codex own these):** `src/cas/trace/`, `src/cas/spec_decode.py`,
`src/cas/commit.py`, `src/cas/config.py`, `src/cas/models.py`, `src/cas/timing.py`,
`src/cas/signals.py`, `src/cas/data/`, `modal_app.py`, `src/cas/policies/`,
`PLAN.md` (propose edits, don't rewrite). Request changes to any of these via a
dated `docs/DECISIONS.md` note.

## 4. Integration seam for I11 (so it drops into the trace later)

The token trace (Claude's I06) will carry per-token category/phase fields. Design
your annotator as a **pure function of the token stream**, not of trace files:

```python
# proposed signature — put it in cas.annotate and note it in docs/DECISIONS.md
def annotate_token(token_id: int, piece: str, position: int, context_pieces: list[str]) -> AnnotatedToken: ...
@dataclass
class AnnotatedToken:
    categories: frozenset[str]   # overlapping; may be empty
    phase: str                   # e.g. "prefix" | "mid" | "late" (define + version)
    category_set_version: str
```

Claude will call this from the writer once I06 lands; keep it torch-free and
tokenizer-light (take decoded `piece` strings as input so you don't load models).

## 5. Environment / how to test

- **Local, CPU only.** `PYTHONPATH=src python -m pytest tests/test_annotate.py -q`.
- `sklearn` (1.3.0) and `pyarrow` (11.0.0) are available locally if useful for the
  stratified-sample analysis; **do not import torch** (local torch is broken/old).
- No GPU, no Modal — those are Claude-driven and metered. Your work needs neither.

## 6. Hard rules (from AGENTS.md — non-negotiable)

- **Naming/disclosure (D008):** cite prior work by arXiv ID/title only. Do **not**
  name any conference, journal, external organization/lab, or the owner's
  strategic goals in any repository file — including `docs/landscape.md`. Refer to
  venues generically ("the target journal"). This is the rule most likely to trip
  a landscape scan; watch it.
- Never present estimated/illustrative values as results. The stratified-sample
  agreement number must come from the scripted check, not be hand-typed.
- Record landscape impacts on claims (esp. C04) and any novelty threat in
  `docs/CLAIMS_LEDGER.md`; use "representation"/"diagnostic signal", not
  "circuit"/"mechanism", until the G2 gate passes.
- **Definition of done (AGENTS.md):** acceptance criteria met; tests pass
  (I11); reproducible command documented; backlog status + affected claims
  updated; interface additions recorded in `docs/DECISIONS.md` before downstream
  use.

## 7. Coordination

- Set I21/I11 to `IN_PROGRESS` (owner `Grok`) before starting; `DONE` when the
  definition of done is met.
- Do not rewrite Claude's or Codex's files. Reconcile conflicts via a dated
  `docs/DECISIONS.md` note.
- Re-verify any repository fact before asserting it in a durable document — the
  other two agents change files under you as they work.
