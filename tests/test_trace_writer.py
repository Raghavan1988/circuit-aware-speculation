"""I06: trace records, schema invariants, and the Parquet writer.

Rounds are constructed through cas.commit.verify_and_commit so the tested
records carry the real engine semantics rather than hand-faked values. These
are unit fixtures, not experimental results (AGENTS.md).
"""
import dataclasses

import pytest

from cas.commit import verify_and_commit
from cas.trace import (
    SCHEMA_VERSION,
    RequestSummary,
    RoundTrace,
    RunMetadata,
    TraceValidationError,
    TraceWriter,
    derive_token_traces,
    validate_request,
    validate_round,
)

pa = pytest.importorskip("pyarrow")
import pyarrow.parquet as pq  # noqa: E402


# ---- fixtures built from the real commit logic ------------------------------

def make_round(request_id, round_id, start_pos, proposals, target_argmax,
               **extra):
    res = verify_and_commit(list(proposals), list(target_argmax))
    return RoundTrace(
        request_id=request_id,
        round_id=round_id,
        start_output_pos=start_pos,
        requested_action=len(proposals),
        realized_draft_len=len(proposals),
        proposed_token_ids=tuple(proposals),
        accepted_prefix_len=res.accepted,
        first_rejection_pos=res.first_rejection,
        emitted_token_ids=res.emitted_ids,
        target_argmax_ids=tuple(target_argmax),
        latency_ns={"draft": 1000, "verify": 2000, "controller": 10},
        **extra,
    )


def make_request(request_id="req0", policy="fixed_4"):
    # round 0: partial accept (2 of 4); round 1: full accept (2 of 2)
    r0 = make_round(request_id, 0, 1, [5, 6, 7, 8], [5, 6, 99, 100, 101],
                    target_entropy_frontier=1.5, target_margin_frontier=0.4)
    r1 = make_round(request_id, 1, 1 + len(r0.emitted_token_ids),
                    [10, 11], [10, 11, 12])
    rounds = [r0, r1]
    drafted = sum(r.realized_draft_len for r in rounds)
    accepted = sum(r.accepted_prefix_len for r in rounds)
    out_tokens = 1 + sum(len(r.emitted_token_ids) for r in rounds)
    summary = RequestSummary(
        run_id="run0", request_id=request_id, dataset="unit", domain="code",
        split="dev", prompt_hash="abc", policy_name=policy,
        prompt_tokens=4, output_tokens=out_tokens,
        termination_reason="max_new_tokens", output_token_hash="h",
        total_drafted=drafted, total_accepted=accepted,
        total_rejected=drafted - accepted, n_rounds=len(rounds),
        prefill_ns=10_000, decode_ns=50_000, ttft_ns=10_000,
        end_to_end_ns=61_000,
    )
    return summary, rounds


def make_meta(**over):
    kw = dict(
        run_id="run0", created_at_utc="2026-07-11T00:00:00Z",
        git_commit="deadbeef", config_hash="cfg", command="pytest", seed=0,
        target_model_id="t", target_revision="r1", draft_model_id="d",
        draft_revision="r2", tokenizer_id="t", tokenizer_revision="r1",
        dtype="bfloat16", quantization=None, device_name="unit",
        device_count=1, driver_cuda_framework={"torch": "x"},
        policy_name="fixed_4", policy_version="1", split_manifest_id="m0",
    )
    kw.update(over)
    return RunMetadata(**kw)


# ---- records ----------------------------------------------------------------

def test_per_position_match_and_counterfactual_labels():
    rt = make_round("r", 0, 1, [5, 6, 7, 8], [5, 6, 99, 100, 101])
    assert rt.per_position_match() == (True, True, False, False)
    assert rt.accepted_prefix_len == 2
    # counterfactual: shorter actions' outcomes are derivable (D018.3)
    assert sum(rt.per_position_match()[:1]) == 1   # L=1 would accept 1
    assert sum(rt.per_position_match()[:3][:2]) == 2


def test_derive_token_traces():
    rt = make_round("r", 0, 3, [5, 6, 7], [5, 6, 9, 10],
                    draft_entropy=(0.1, 0.2, 0.3),
                    draft_top1_margin=(0.9, 0.8, 0.7))
    toks = derive_token_traces(rt, "run0")
    assert [t.accepted for t in toks] == [True, True, False]
    assert [t.token_position for t in toks] == [3, 4, 5]
    assert toks[2].target_token_id == 9
    assert toks[1].draft_entropy == 0.2


def test_skip_round_has_no_token_traces():
    rt = make_round("r", 0, 1, [], [42])
    assert derive_token_traces(rt, "run0") == []
    validate_round(rt)


def test_accepted_vs_target_match_diverge_after_rejection():
    """`accepted` is committed-prefix membership; `target_match` is the
    counterfactual per-position agreement (review 2026-07-11): position 2
    matches the target's counterfactual argmax but was never emitted."""
    rt = make_round("r", 0, 1, [5, 6, 7], [5, 9, 7, 12])
    assert rt.accepted_prefix_len == 1
    toks = derive_token_traces(rt, "run0")
    assert [t.accepted for t in toks] == [True, False, False]
    assert [t.target_match for t in toks] == [True, False, True]


def test_round_contiguity_invariant():
    """start_output_pos must tile the output; a gap fails validation."""
    summary, rounds = make_request()
    rounds[1] = dataclasses.replace(rounds[1], start_output_pos=99)
    with pytest.raises(TraceValidationError, match="start_output_pos"):
        validate_request(summary, rounds)


# ---- validation invariants ----------------------------------------------------

def test_validate_request_passes():
    summary, rounds = make_request()
    validate_request(summary, rounds)


def test_totals_mismatch_fails():
    summary, rounds = make_request()
    bad = dataclasses.replace(summary, total_accepted=summary.total_accepted + 1)
    with pytest.raises(TraceValidationError, match="total_accepted"):
        validate_request(bad, rounds)


def test_accepted_exceeding_drafted_fails():
    rt = make_round("r", 0, 1, [5, 6], [5, 6, 7])
    rt = dataclasses.replace(rt, accepted_prefix_len=3)
    with pytest.raises(TraceValidationError):
        validate_round(rt)


def test_tampered_emission_fails():
    rt = make_round("r", 0, 1, [5, 6], [5, 9, 7])
    rt = dataclasses.replace(rt, emitted_token_ids=(5, 6))
    with pytest.raises(TraceValidationError, match="emitted"):
        validate_round(rt)


def test_negative_latency_fails():
    rt = make_round("r", 0, 1, [5], [5, 6])
    rt.latency_ns["draft"] = -1
    with pytest.raises(TraceValidationError, match="negative latency"):
        validate_round(rt)


def test_target_only_with_draft_stats_fails():
    summary, rounds = make_request(policy="target_only")
    with pytest.raises(TraceValidationError, match="target-only"):
        validate_request(summary, rounds)


def test_target_only_without_rounds_passes():
    """The pure-autoregressive reference path records no rounds."""
    summary, _ = make_request(policy="target_only")
    summary = dataclasses.replace(summary, total_drafted=0, total_accepted=0,
                                  total_rejected=0, n_rounds=0)
    validate_request(summary, [])


def test_spec_policy_without_rounds_fails():
    summary, _ = make_request(policy="fixed_4")
    summary = dataclasses.replace(summary, n_rounds=0)
    with pytest.raises(TraceValidationError, match="no rounds"):
        validate_request(summary, [])


def test_foreign_request_id_fails():
    summary, rounds = make_request()
    rounds[1] = dataclasses.replace(rounds[1], request_id="other")
    with pytest.raises(TraceValidationError, match="foreign request_id"):
        validate_request(summary, rounds)


# ---- writer -------------------------------------------------------------------

def test_writer_roundtrip(tmp_path):
    w = TraceWriter(str(tmp_path / "run0"), make_meta())
    summary, rounds = make_request()
    w.add_request(summary, rounds)
    manifest = w.finalize()

    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["files"]["request_summaries.parquet"]["rows"] == 1
    assert manifest["files"]["rounds.parquet"]["rows"] == 2
    assert manifest["files"]["tokens.parquet"]["rows"] == 6  # 4 + 2 drafted

    rounds_tbl = pq.read_table(str(tmp_path / "run0" / "rounds.parquet"))
    got = rounds_tbl.to_pylist()
    assert got[0]["target_argmax_ids"] == [5, 6, 99, 100, 101]
    assert got[0]["target_entropy_frontier"] == 1.5
    toks = pq.read_table(str(tmp_path / "run0" / "tokens.parquet")).to_pylist()
    assert [t["accepted"] for t in toks][:4] == [True, True, False, False]


def test_writer_refuses_sealed_dir(tmp_path):
    d = str(tmp_path / "run0")
    w = TraceWriter(d, make_meta())
    summary, rounds = make_request()
    w.add_request(summary, rounds)
    w.finalize()
    with pytest.raises(TraceValidationError, match="immutable"):
        TraceWriter(d, make_meta())


def test_writer_validates_before_writing(tmp_path):
    d = str(tmp_path / "run0")
    w = TraceWriter(d, make_meta())
    summary, rounds = make_request()
    bad = dataclasses.replace(summary, n_rounds=99)
    w.add_request(bad, rounds)
    with pytest.raises(TraceValidationError):
        w.finalize()
    import os
    assert not os.path.exists(os.path.join(d, "MANIFEST.json"))


def test_schema_version_mismatch_fails(tmp_path):
    meta = make_meta(schema_version="0.0.9")
    with pytest.raises(TraceValidationError, match="schema_version"):
        TraceWriter(str(tmp_path / "x"), meta)
