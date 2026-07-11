"""Prompt-grouped dev/test splitting (pure, stdlib only).

The contract (docs/EXPERIMENT_CONTRACT.md, docs/RESEARCH_SPEC.md) forbids
token-level random splitting: adjacent tokens from one prompt are dependent, so
a token that leaks from dev into test inflates apparent generalization. We split
by *prompt*, per domain, with a fixed seed, and freeze the assignment as a
manifest keyed by prompt hash.

Determinism note: no wall-clock or unseeded RNG is used (the harness forbids
Date.now/Math.random-style nondeterminism). Assignment is a deterministic
function of (prompt_hash, seed), so re-running produces the identical manifest.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class PromptRecord:
    """One ingested prompt, before splitting. `prompt_hash` is the stable key."""

    prompt_id: str
    dataset: str
    domain: str
    prompt_hash: str


def _assignment_score(prompt_hash: str, seed: int) -> float:
    """Deterministic pseudo-uniform value in [0, 1) from (hash, seed).

    Uses SHA-256 of the salted hash so the split is reproducible from the
    manifest alone and independent of dict/set iteration order.
    """
    h = hashlib.sha256(f"{seed}:{prompt_hash}".encode()).hexdigest()
    return int(h[:16], 16) / float(1 << 64)


def split_by_prompt(
    records: list[PromptRecord],
    dev_fraction: float = 0.5,
    seed: int = 0,
) -> dict[str, str]:
    """Assign each prompt to 'dev' or 'test', grouped and balanced per domain.

    Every prompt with a distinct `prompt_hash` is assigned independently, but
    the fraction is enforced *within each domain* so both splits cover every
    domain. Assignment is by rank of the deterministic score within the domain,
    making the dev fraction exact up to integer rounding (not just expected).

    Args:
        records: ingested prompts (duplicates by prompt_hash are collapsed).
        dev_fraction: share of each domain's prompts assigned to dev.
        seed: split seed; recorded in the manifest.

    Returns:
        Mapping prompt_hash -> 'dev' | 'test'.

    Raises:
        ValueError: if dev_fraction is not in (0, 1) or records is empty.
    """
    if not 0.0 < dev_fraction < 1.0:
        raise ValueError(f"dev_fraction must be in (0, 1), got {dev_fraction}")
    if not records:
        raise ValueError("no records to split")

    # collapse duplicate prompt hashes, remember each one's domain
    domain_of: dict[str, str] = {}
    for r in records:
        domain_of.setdefault(r.prompt_hash, r.domain)

    by_domain: dict[str, list[str]] = {}
    for phash, domain in domain_of.items():
        by_domain.setdefault(domain, []).append(phash)

    assignment: dict[str, str] = {}
    for domain, hashes in by_domain.items():
        # deterministic order: sort by (score, hash) so ties break stably
        ranked = sorted(hashes, key=lambda h: (_assignment_score(h, seed), h))
        n_dev = round(len(ranked) * dev_fraction)
        # guarantee both splits are non-empty when the domain has >= 2 prompts
        if len(ranked) >= 2:
            n_dev = min(max(n_dev, 1), len(ranked) - 1)
        for i, phash in enumerate(ranked):
            assignment[phash] = "dev" if i < n_dev else "test"
    return assignment


def split_summary(assignment: dict[str, str], records: list[PromptRecord]) -> dict:
    """Human-auditable counts per domain and split, for the manifest header."""
    domain_of = {r.prompt_hash: r.domain for r in records}
    counts: dict[str, dict[str, int]] = {}
    for phash, split in assignment.items():
        d = domain_of.get(phash, "unknown")
        counts.setdefault(d, {"dev": 0, "test": 0})[split] += 1
    return counts
