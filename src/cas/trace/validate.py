"""Automatic validation of TRACE_SCHEMA.md invariants (issue I06).

Pure stdlib on purpose (mirrors cas.commit): every invariant is checkable on
any machine, in CI, and inside the writer before a single byte is persisted.
Raises TraceValidationError with a precise message; never silently repairs.

Invariants encoded (TRACE_SCHEMA.md "Validation invariants"):
  1. foreign identifiers resolve to exactly one parent record;
  2. accepted prefix length <= realized draft length;
  3. per-request token totals equal sums over rounds;
  4. latency components non-negative; total bounds components within tolerance;
  5. (train/test fitting separation is enforced at analysis time, not here);
  6. target-only records contain no draft statistics;
  7. schema_version present (writer stamps it from records.SCHEMA_VERSION).

Plus the D014/D018 structural checks the schema implies for the exact greedy
engine: target_argmax length, accepted-prefix consistency with the per-position
match vector, and emitted-token reconstruction (accepted prefix + correction).
"""
from __future__ import annotations

from .records import RoundTrace, RequestSummary, RunMetadata

# Names whose summaries must contain no draft statistics (invariant 6).
TARGET_ONLY_POLICIES = frozenset({"target_only", "target-only"})


class TraceValidationError(ValueError):
    """A schema invariant failed; the offending record is named in args."""


def _fail(msg: str) -> None:
    raise TraceValidationError(msg)


def validate_round(rt: RoundTrace) -> None:
    """Structural invariants for one decode round."""
    where = f"round {rt.request_id}/{rt.round_id}"
    if rt.realized_draft_len != len(rt.proposed_token_ids):
        _fail(f"{where}: realized_draft_len {rt.realized_draft_len} != "
              f"len(proposed) {len(rt.proposed_token_ids)}")
    if not (0 <= rt.accepted_prefix_len <= rt.realized_draft_len):
        _fail(f"{where}: accepted_prefix_len {rt.accepted_prefix_len} outside "
              f"[0, {rt.realized_draft_len}]")  # invariant 2
    if not rt.emitted_token_ids:
        _fail(f"{where}: every round must emit at least one token")
    if len(rt.emitted_token_ids) != rt.accepted_prefix_len + 1:
        _fail(f"{where}: emitted {len(rt.emitted_token_ids)} != accepted+1 "
              f"{rt.accepted_prefix_len + 1}")
    for name, ns in rt.latency_ns.items():
        if ns < 0:
            _fail(f"{where}: negative latency component {name}={ns}")

    # D018: target argmax covers each drafted position + the bonus position.
    if rt.target_argmax_ids:
        want = rt.realized_draft_len + 1
        if len(rt.target_argmax_ids) != want:
            _fail(f"{where}: target_argmax_ids has {len(rt.target_argmax_ids)} "
                  f"entries, want realized+1 = {want}")
        match = rt.per_position_match()
        k = 0
        for m in match:
            if m:
                k += 1
            else:
                break
        if k != rt.accepted_prefix_len:
            _fail(f"{where}: accepted_prefix_len {rt.accepted_prefix_len} != "
                  f"longest match prefix {k}")
        expect = (tuple(rt.proposed_token_ids[:k])
                  + (rt.target_argmax_ids[k],))
        if tuple(rt.emitted_token_ids) != expect:
            _fail(f"{where}: emitted {rt.emitted_token_ids} != accepted prefix "
                  f"+ correction {expect}")
    if rt.first_rejection_pos is not None:
        if rt.first_rejection_pos != rt.accepted_prefix_len + 1:
            _fail(f"{where}: first_rejection_pos {rt.first_rejection_pos} "
                  f"inconsistent with accepted {rt.accepted_prefix_len}")
    elif rt.accepted_prefix_len != rt.realized_draft_len:
        _fail(f"{where}: no rejection recorded but accepted "
              f"{rt.accepted_prefix_len} < drafted {rt.realized_draft_len}")


def validate_request(
    summary: RequestSummary,
    rounds: list[RoundTrace],
    latency_tolerance: float = 0.05,
) -> None:
    """Cross-record invariants for one request (invariants 1, 3, 4, 6)."""
    where = f"request {summary.request_id}"

    # Target-only requests legitimately have no rounds (pure autoregressive
    # reference path); everything else must have at least one.
    if not rounds:
        if summary.policy_name not in TARGET_ONLY_POLICIES:
            _fail(f"{where}: no rounds recorded for non-target-only policy "
                  f"{summary.policy_name!r}")
        if summary.total_drafted or summary.total_accepted or summary.total_rejected:
            _fail(f"{where}: target-only carries draft totals")  # invariant 6
        if summary.n_rounds != 0:
            _fail(f"{where}: n_rounds {summary.n_rounds} != 0 with no rounds")
        return

    for rt in rounds:
        if rt.request_id != summary.request_id:
            _fail(f"{where}: round {rt.round_id} has foreign request_id "
                  f"{rt.request_id!r}")  # invariant 1
        validate_round(rt)

    ids = [rt.round_id for rt in rounds]
    if ids != list(range(len(rounds))):
        _fail(f"{where}: round_ids not contiguous from 0: {ids}")

    # Rounds tile the output contiguously: round 0 starts after the prefill
    # token; each next round starts within (prev start, prev start + emitted]
    # (an early eos/max_new break may commit fewer than emitted, and only the
    # final round may break early).
    if rounds and rounds[0].start_output_pos != 1:
        _fail(f"{where}: first round starts at "
              f"{rounds[0].start_output_pos}, expected 1")
    for prev, cur in zip(rounds, rounds[1:]):
        lo = prev.start_output_pos + 1
        hi = prev.start_output_pos + len(prev.emitted_token_ids)
        if not (lo <= cur.start_output_pos <= hi):
            _fail(f"{where}: round {cur.round_id} start_output_pos "
                  f"{cur.start_output_pos} outside [{lo}, {hi}]")

    drafted = sum(rt.realized_draft_len for rt in rounds)
    accepted = sum(rt.accepted_prefix_len for rt in rounds)
    if summary.total_drafted != drafted:
        _fail(f"{where}: total_drafted {summary.total_drafted} != sum over "
              f"rounds {drafted}")  # invariant 3
    if summary.total_accepted != accepted:
        _fail(f"{where}: total_accepted {summary.total_accepted} != {accepted}")
    if summary.total_rejected != drafted - accepted:
        _fail(f"{where}: total_rejected {summary.total_rejected} != "
              f"{drafted - accepted}")
    if summary.n_rounds != len(rounds):
        _fail(f"{where}: n_rounds {summary.n_rounds} != {len(rounds)}")

    # Output count: 1 prefill token + per-round emissions, possibly truncated
    # by max_new_tokens inside the final round only.
    emitted = 1 + sum(len(rt.emitted_token_ids) for rt in rounds)
    excess = emitted - summary.output_tokens
    limit = len(rounds[-1].emitted_token_ids) if rounds else 1
    if excess < 0 or excess >= max(limit, 1) + 1:
        _fail(f"{where}: output_tokens {summary.output_tokens} irreconcilable "
              f"with per-round emissions {emitted} (excess {excess})")

    # invariant 4: synchronized total bounds recorded components.
    parts = summary.prefill_ns + summary.decode_ns
    if summary.end_to_end_ns and parts > summary.end_to_end_ns * (1 + latency_tolerance):
        _fail(f"{where}: components {parts}ns exceed end_to_end "
              f"{summary.end_to_end_ns}ns beyond tolerance")
    for v, name in ((summary.prefill_ns, "prefill_ns"),
                    (summary.decode_ns, "decode_ns"),
                    (summary.ttft_ns, "ttft_ns"),
                    (summary.end_to_end_ns, "end_to_end_ns")):
        if v < 0:
            _fail(f"{where}: negative {name}")

    # invariant 6: target-only requests carry no draft statistics.
    if summary.policy_name in TARGET_ONLY_POLICIES:
        if drafted or any(rt.proposed_token_ids for rt in rounds):
            _fail(f"{where}: target-only policy carries draft statistics")


def validate_run(
    meta: RunMetadata,
    summaries: list[RequestSummary],
    rounds_by_request: dict[str, list[RoundTrace]],
) -> None:
    """Run-level invariants (1, 7): keys resolve, ids unique, version stamped."""
    if not meta.schema_version:
        _fail("run metadata missing schema_version")  # invariant 7
    if not meta.run_id:
        _fail("run metadata missing run_id")
    seen: set[str] = set()
    for s in summaries:
        if s.run_id != meta.run_id:
            _fail(f"request {s.request_id}: run_id {s.run_id!r} != "
                  f"{meta.run_id!r}")
        if s.request_id in seen:
            _fail(f"duplicate request_id {s.request_id!r}")
        seen.add(s.request_id)
    for req_id in rounds_by_request:
        if req_id not in seen:
            _fail(f"rounds recorded for unknown request {req_id!r}")
    for s in summaries:
        validate_request(s, rounds_by_request.get(s.request_id, []))
