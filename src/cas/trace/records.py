"""Structured trace records emitted per decode round and per request.

Field names mirror docs/TRACE_SCHEMA.md so the Step-3 writer (I06) can serialize
them directly. This module intentionally holds only plain dataclasses (no torch)
so records can be constructed, asserted on, and later written without the model.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RoundTrace:
    """One draft/verify round (decode-round trace in the schema)."""

    request_id: str
    round_id: int
    start_output_pos: int          # index into generated tokens where this round begins
    requested_action: int          # L (0 == skip)
    realized_draft_len: int        # == requested_action for the exact engine
    proposed_token_ids: tuple[int, ...]
    accepted_prefix_len: int
    first_rejection_pos: int | None
    emitted_token_ids: tuple[int, ...]
    # cheap signals for the controller/probes, per proposed position
    draft_entropy: tuple[float, ...] = ()
    draft_top1_margin: tuple[float, ...] = ()
    # latency components in nanoseconds (authoritative only under CUDA timing)
    latency_ns: dict[str, int] = field(default_factory=dict)
    cache_len_before: int = 0
    cache_len_after: int = 0


@dataclass
class RequestSummary:
    """One prompt under one policy (request summary in the schema)."""

    run_id: str
    request_id: str
    dataset: str
    domain: str
    split: str
    prompt_hash: str
    policy_name: str
    prompt_tokens: int
    output_tokens: int
    termination_reason: str
    output_token_hash: str
    total_drafted: int
    total_accepted: int
    total_rejected: int
    n_rounds: int
    # timing (ns); reconstructed TTFT/TPOT recorded by the caller
    prefill_ns: int = 0
    decode_ns: int = 0
    ttft_ns: int = 0
    end_to_end_ns: int = 0
    peak_mem_bytes: int = 0
    # equivalence (I03): does this match target-only greedy for the same prompt?
    equivalence_status: str = "unchecked"   # "identical" | "diverged" | "unchecked"
    reference_output_hash: str = ""
    failure_reason: str = ""

    def accepted_per_drafted(self) -> float:
        return self.total_accepted / self.total_drafted if self.total_drafted else 0.0
