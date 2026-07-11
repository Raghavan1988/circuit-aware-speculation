"""Structured trace records emitted per decode round and per request.

Field names mirror docs/TRACE_SCHEMA.md so the Step-3 writer (I06) can serialize
them directly. This module intentionally holds only plain dataclasses (no torch)
so records can be constructed, asserted on, and later written without the model.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Bump on any schema change and add a migration note (TRACE_SCHEMA invariant 7).
SCHEMA_VERSION = "0.1.0"


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
    # --- D018 fields (engine wiring lands with task #2; defaults keep old
    # construction sites valid until then) -------------------------------
    # Target greedy argmax at every drafted position PLUS the bonus position
    # (len == realized_draft_len + 1; == 1 for skip). Strictly stronger than a
    # match vector: yields per-position counterfactual labels AND the target
    # token ids the token trace needs.
    target_argmax_ids: tuple[int, ...] = ()
    # Free verification byproducts at the frontier (last committed) position —
    # the distribution that produced this round's correction/bonus token.
    # Mandatory members of the hardened C10 baseline (D018.2).
    target_entropy_frontier: float | None = None
    target_margin_frontier: float | None = None
    # Controller context (schema: recent-acceptance state, per-action payoffs).
    recent_acceptance_state: float | None = None
    predicted_payoffs: dict[str, float] | None = None
    # Provenance slots (schema: intervention id; activation artifact id).
    intervention_id: str | None = None
    activation_artifact_id: str | None = None

    def per_position_match(self) -> tuple[bool, ...]:
        """Draft/target agreement at each drafted position (counterfactual
        labels for all shorter actions, D018.3). Empty if target argmax ids
        were not recorded (pre-D018 traces) or the round was a skip."""
        if not self.target_argmax_ids or not self.proposed_token_ids:
            return ()
        return tuple(
            p == t
            for p, t in zip(self.proposed_token_ids, self.target_argmax_ids)
        )


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


@dataclass
class RunMetadata:
    """One record per experiment invocation (run metadata in the schema).

    Every field is required by TRACE_SCHEMA.md; `command` must be pre-scrubbed
    of secrets by the caller. `quantization` is None for full-precision runs.
    """

    run_id: str
    created_at_utc: str          # ISO-8601, UTC
    git_commit: str
    config_hash: str
    command: str
    seed: int
    target_model_id: str
    target_revision: str
    draft_model_id: str          # "" for target-only runs
    draft_revision: str          # "" for target-only runs
    tokenizer_id: str
    tokenizer_revision: str
    dtype: str
    quantization: str | None
    device_name: str
    device_count: int
    driver_cuda_framework: dict[str, str]
    policy_name: str
    policy_version: str
    split_manifest_id: str
    schema_version: str = SCHEMA_VERSION


@dataclass
class TokenTrace:
    """One proposed (or target-emitted) token (token trace in the schema).

    Derived from a RoundTrace via `derive_token_traces`; carries the I11
    annotation seam fields (D016) and the activation artifact reference (I10).
    """

    run_id: str
    request_id: str
    round_id: int
    token_position: int          # absolute position in the generated output
    proposal_offset: int         # 0-based index within the round's draft
    draft_token_id: int
    target_token_id: int
    accepted: bool           # committed: position is inside the accepted prefix
    # Counterfactual per-position draft/target agreement (D018.3). Differs from
    # `accepted` after the first rejection: a later position may match the
    # target's counterfactual argmax yet was never emitted.
    target_match: bool = False
    draft_entropy: float | None = None
    draft_top1_margin: float | None = None
    draft_logprob: float | None = None
    divergence: float | None = None      # post-verification draft-target divergence
    categories: tuple[str, ...] = ()     # overlapping, may be empty (D016)
    phase: str = ""
    category_set_version: str = ""
    phase_set_version: str = ""
    activation_artifact_id: str | None = None
    activation_row_offset: int | None = None


def derive_token_traces(rt: RoundTrace, run_id: str) -> list[TokenTrace]:
    """Expand one round into per-token records (annotation fields left empty;
    the writer applies the I11 annotator when token pieces are available).

    Requires `target_argmax_ids` (D018); returns [] for skip rounds or
    pre-D018 rounds so callers can degrade gracefully.
    """
    if not rt.proposed_token_ids or not rt.target_argmax_ids:
        return []
    match = rt.per_position_match()
    out: list[TokenTrace] = []
    for i, (draft_id, target_id) in enumerate(
        zip(rt.proposed_token_ids, rt.target_argmax_ids)
    ):
        out.append(TokenTrace(
            run_id=run_id,
            request_id=rt.request_id,
            round_id=rt.round_id,
            token_position=rt.start_output_pos + i,
            proposal_offset=i,
            draft_token_id=draft_id,
            target_token_id=target_id,
            accepted=i < rt.accepted_prefix_len,
            target_match=match[i],
            draft_entropy=rt.draft_entropy[i] if i < len(rt.draft_entropy) else None,
            draft_top1_margin=(
                rt.draft_top1_margin[i] if i < len(rt.draft_top1_margin) else None
            ),
            activation_artifact_id=rt.activation_artifact_id,
            activation_row_offset=i if rt.activation_artifact_id else None,
        ))
    return out
