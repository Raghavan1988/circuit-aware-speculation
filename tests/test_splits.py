"""I05 (partial): unit tests for pure prompt-grouped splitting.

Runs anywhere (stdlib only). Verifies determinism, per-domain coverage, the
no-token-leak property (splitting is by prompt hash), and exact fractions.
"""
import pytest

from cas.data.splits import PromptRecord, split_by_prompt, split_summary


def _mk(n, domain, start=0):
    return [
        PromptRecord(
            prompt_id=f"{domain}-{i}",
            dataset=domain,
            domain=domain,
            prompt_hash=f"hash-{domain}-{i}",
        )
        for i in range(start, start + n)
    ]


def test_deterministic_same_seed():
    recs = _mk(50, "code") + _mk(50, "math")
    a = split_by_prompt(recs, dev_fraction=0.5, seed=0)
    b = split_by_prompt(recs, dev_fraction=0.5, seed=0)
    assert a == b


def test_seed_changes_assignment():
    recs = _mk(100, "chat")
    a = split_by_prompt(recs, seed=0)
    b = split_by_prompt(recs, seed=1)
    assert a != b  # overwhelmingly likely to differ for 100 prompts


def test_both_splits_cover_every_domain():
    recs = _mk(20, "code") + _mk(20, "math") + _mk(20, "chat") + _mk(20, "summ")
    a = split_by_prompt(recs, dev_fraction=0.5, seed=7)
    summary = split_summary(a, recs)
    for domain, c in summary.items():
        assert c["dev"] > 0, f"{domain} has empty dev"
        assert c["test"] > 0, f"{domain} has empty test"


def test_fraction_is_exact_per_domain():
    recs = _mk(100, "code")
    a = split_by_prompt(recs, dev_fraction=0.3, seed=3)
    n_dev = sum(1 for v in a.values() if v == "dev")
    assert n_dev == 30


def test_duplicate_hashes_collapsed():
    # same prompt ingested twice must get one assignment, not two
    recs = _mk(10, "code") + _mk(10, "code")  # identical hashes repeated
    a = split_by_prompt(recs, seed=0)
    assert len(a) == 10


def test_invalid_fraction_raises():
    recs = _mk(10, "code")
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            split_by_prompt(recs, dev_fraction=bad)


def test_empty_raises():
    with pytest.raises(ValueError):
        split_by_prompt([], dev_fraction=0.5)


def test_single_prompt_domain_not_forced_empty():
    # a domain with 1 prompt cannot fill both splits; it should still assign
    recs = _mk(1, "rare") + _mk(10, "code")
    a = split_by_prompt(recs, seed=0)
    assert "hash-rare-0" in a
