"""Surface-baseline feature construction (Task #4 / I13 groundwork).

Builds per-token feature rows for the cheap acceptance predictors — the
credited baseline every probe claim must beat (D018.1). Pure stdlib over
plain dict rows (rounds.parquet / tokens.parquet -> to_pylist()) so feature
semantics are unit-tested locally; the sklearn fitting lives in
scripts/fit_baselines.py and runs where sklearn is pinned.

Leakage rules encoded here, not left to callers:
  * history features use STRICTLY PRIOR rounds of the same request;
  * frontier (pre-round) target entropy/margin come from the PREVIOUS round's
    verification pass — the current round's frontier values are same-round
    outcome information and are never exposed as features;
  * domain is a categorical passthrough for one-hot encoding downstream.
"""
from __future__ import annotations

import json

HISTORY_EMA_ALPHA = 0.3  # frozen (D018: test-time hyperparameters are frozen)


def _lat(round_row) -> dict:
    lat = round_row.get("latency_ns") or {}
    return json.loads(lat) if isinstance(lat, str) else lat


def _match_prefix_rate(round_row) -> float:
    """Realized per-round acceptance fraction (accepted / drafted)."""
    drafted = round_row["realized_draft_len"]
    return (round_row["accepted_prefix_len"] / drafted) if drafted else 1.0


def feature_rows(rounds, domain_by_request=None) -> list[dict]:
    """Per-drafted-token feature/label rows from one policy's round records.

    Args:
        rounds: iterable of round dicts, ordered or orderable by
            (request_id, round_id), each carrying the D018 fields
            (proposed_token_ids, target_argmax_ids, draft_entropy,
            draft_top1_margin, target_entropy_frontier, target_margin_frontier).
        domain_by_request: optional {request_id: domain} map (from
            request_summaries); emitted as the `domain` column.

    Returns:
        One dict per drafted token with features, the acceptance label, and
        grouping keys (request_id for prompt-grouped splits).
    """
    domain_by_request = domain_by_request or {}
    by_req: dict = {}
    for r in rounds:
        by_req.setdefault(r["request_id"], []).append(r)
    out: list[dict] = []
    for req_id, rs in by_req.items():
        rs.sort(key=lambda r: r["round_id"])
        ema = None          # recent-acceptance EMA over prior rounds only
        prev_frontier_e = None
        prev_frontier_m = None
        for r in rs:
            proposed = r["proposed_token_ids"] or []
            targets = r["target_argmax_ids"] or []
            ents = r.get("draft_entropy") or []
            margs = r.get("draft_top1_margin") or []
            k = r["accepted_prefix_len"]
            for i, tok in enumerate(proposed):
                out.append({
                    "request_id": req_id,
                    "round_id": r["round_id"],
                    "proposal_offset": i,
                    "output_pos": r["start_output_pos"] + i,
                    "domain": domain_by_request.get(req_id, "unknown"),
                    # draft-side post-draft signals (exist only after drafting)
                    "draft_entropy": ents[i] if i < len(ents) else None,
                    "draft_margin": margs[i] if i < len(margs) else None,
                    # history: strictly prior rounds (None on round 0 -> the
                    # fitter imputes the dev-set prior and adds a missing flag)
                    "history_ema": ema,
                    # pre-round target-side signals from the PREVIOUS verify
                    "prev_target_entropy": prev_frontier_e,
                    "prev_target_margin": prev_frontier_m,
                    # label (committed acceptance) + counterfactual match
                    "accepted": i < k,
                    "target_match": (tok == targets[i]) if i < len(targets) else None,
                })
            rate = _match_prefix_rate(r)
            ema = rate if ema is None else (
                HISTORY_EMA_ALPHA * rate + (1 - HISTORY_EMA_ALPHA) * ema)
            prev_frontier_e = r.get("target_entropy_frontier")
            prev_frontier_m = r.get("target_margin_frontier")
    return out


# Feature sets for the incremental-information ladder (I13). Each entry is
# (name, feature columns); the fitter runs them in order and reports deltas.
FEATURE_SETS = (
    ("entropy_only", ("draft_entropy",)),
    ("entropy_margin", ("draft_entropy", "draft_margin")),
    ("history_only", ("history_ema",)),
    ("surface_stack", ("draft_entropy", "draft_margin", "history_ema",
                       "proposal_offset", "output_pos")),
    ("surface_stack_domain", ("draft_entropy", "draft_margin", "history_ema",
                              "proposal_offset", "output_pos", "domain")),
    # pre-round-only ladder (C10 baseline side: no draft signals exist yet)
    ("preround_history", ("history_ema",)),
    ("preround_hardened", ("history_ema", "prev_target_entropy",
                           "prev_target_margin")),
    ("preround_hardened_domain", ("history_ema", "prev_target_entropy",
                                  "prev_target_margin", "domain")),
)
