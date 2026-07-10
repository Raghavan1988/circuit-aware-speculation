# CLAUDE.md

Guidance for Claude Code sessions in this repository.

## Read order (before changing anything)

1. `AGENTS.md` — the operating contract. It binds all agents, including Claude.
2. `docs/RESEARCH_SPEC.md`, `docs/EXPERIMENT_CONTRACT.md`, `docs/DECISIONS.md`.
3. `PLAN.md` — Claude's execution schedule, decision gates, venue timeline, and
   the competitive-landscape scan (dated 2026-07-10).
4. `docs/ISSUE_BACKLOG.md` — pick one unblocked issue, set owner + status first.

## Non-negotiables (summary; AGENTS.md is authoritative)

- Never present smoke/synthetic/illustrative values as results; never hand-type
  numbers into tables or figures — every reported number is script-generated from
  immutable raw artifacts.
- Prompt-grouped splits only; token-level random splits are prohibited.
- All end-to-end latency numbers include controller, tracing, sync, and routing
  overhead; device-synchronized timing boundaries.
- Scope is locked by `docs/EXPERIMENT_CONTRACT.md`; deviations need a dated
  `docs/DECISIONS.md` entry. Draft routing and EAGLE-3 are deferred — do not start
  them while core work (exact decoding, baselines, traces, probes, selective
  speculation) is incomplete.
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

## Writing style for this repo

- Do not name any conference, journal, or lab/organization in files Claude
  authors here; refer to venues generically (see PLAN.md §6).
- Small changes scoped to one backlog issue; update backlog status and the
  claims ledger in the same change.
- Multiple agents work in this repo (Claude and Codex). Do not rewrite the other
  agent's files; add or extend, and reconcile conflicts via `docs/DECISIONS.md`.
