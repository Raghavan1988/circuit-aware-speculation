# CLAUDE.md

Guidance for Claude Code sessions in this repository.

## Read order (before changing anything)

1. `AGENTS.md` — the operating contract. It binds all agents, including Claude.
2. `docs/RESEARCH_SPEC.md` (including the "Decision gates" G1–G5 and
   "Differentiation requirements" sections), `docs/EXPERIMENT_CONTRACT.md`,
   `docs/DECISIONS.md` (D001–D011 govern scope, naming, gates, release).
3. `PLAN.md` — execution calendar, gates pointer, budget, competitive landscape.
4. `docs/ISSUE_BACKLOG.md` — pick one unblocked issue, set owner + status first.

(The two agent opinion files from 2026-07-10 planning were migrated into the
files above and deleted; see D012.)

## Non-negotiables (summary; AGENTS.md is authoritative)

- Never present smoke/synthetic/illustrative values as results; never hand-type
  numbers into tables or figures — every reported number is script-generated from
  immutable raw artifacts.
- Prompt-grouped splits only; token-level random splits are prohibited.
- All end-to-end latency numbers include controller, tracing, sync, and routing
  overhead; device-synchronized timing boundaries.
- Scope is locked by `docs/EXPERIMENT_CONTRACT.md`; deviations need a dated
  `docs/DECISIONS.md` entry. Draft routing and modern-speculator integration are
  deferred from the core (D003, D009).
- Naming policy (D008): no conference/journal names, no external organization or
  lab names, no owner strategic goals in any repository file. Venues appear as
  dates or generic descriptions only.
- Release timing follows the staged policy (D010): science gates the preprint;
  the impact gate (G4) gates the broader artifact release.
- "Mechanism"/"circuit" wording is gated by G2; use "representation" or
  "diagnostic signal" until interventions pass.
- Record negative results and failed runs in `docs/CLAIMS_LEDGER.md`.

## Environment facts (verified 2026-07-10, re-verify if stale)

- Local GPU: RTX 4090 Laptop, 16 GB — suitable for tests/CPU work, not main runs.
- Local base conda env is broken for this work (CPU-only torch 2.0.1,
  transformers 4.29). Use a dedicated pinned env; never install into base.
- Main experiments run on Modal or RunPod (A100/H100 class). `~/.modal.toml`
  exists on the local machine. No provider credentials in the repo, ever.
- Hugging Face: the user's cached token is invalid (401 on gated repos). The
  primary Qwen2.5-7B/0.5B pair is ungated and works without auth. The Llama
  replication pair is gated — ask the user to run `huggingface-cli login` before
  starting issue I17.
- `/home/raghavan/13_Raghavan_Content_Aware_Speculation_Control` is read-only
  reference material (see PLAN.md §8 for what is worth mining).

## Working style for this repo

- Small changes scoped to one backlog issue; update backlog status and the
  claims ledger in the same change.
- Multiple agents work here (Claude and Codex). Do not rewrite the other agent's
  files; add or extend, and reconcile conflicts via `docs/DECISIONS.md`.
- Re-verify any repository fact before asserting it in a durable document — the
  other agent works concurrently and files change between turns.
