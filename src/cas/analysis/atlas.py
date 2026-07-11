"""Acceptance-atlas aggregation (C04 / I18 groundwork).

Aggregates per-token acceptance by (token category x generation phase) with
prompt-grouped bootstrap confidence intervals. Pure stdlib over plain dict
rows (as produced by `tokens.parquet` -> `to_pylist()`), so the same code is
unit-testable locally and runs on a CPU container against sealed sweep
artifacts. Every number is derived from immutable traces (AGENTS.md); nothing
here fabricates or interpolates.

Categories are overlapping (D016): a token contributes to every category it
carries, so cells are NOT disjoint and their counts do not sum to the corpus
size — by design.
"""
from __future__ import annotations

import random


def atlas_cells(tokens, min_n: int = 1) -> dict:
    """Group token rows into (category, phase) cells.

    Args:
        tokens: iterable of dicts with at least `request_id`, `accepted`
            (bool), `categories` (list[str], may be empty), `phase` (str).
        min_n: drop cells with fewer total tokens than this.

    Returns:
        {(category, phase): {"n", "accepted", "rate", "prompts"}} where
        `prompts` maps prompt/request id -> [n_tokens, n_accepted] for the
        prompt-grouped bootstrap.
    """
    cells: dict = {}
    for t in tokens:
        cats = t.get("categories") or ["uncategorized"]
        phase = t.get("phase") or "unknown"
        acc = bool(t["accepted"])
        for cat in cats:
            c = cells.setdefault((cat, phase), {"n": 0, "k": 0, "prompts": {}})
            c["n"] += 1
            c["k"] += acc
            p = c["prompts"].setdefault(t["request_id"], [0, 0])
            p[0] += 1
            p[1] += acc
    out = {}
    for key, c in cells.items():
        if c["n"] >= min_n:
            out[key] = {"n": c["n"], "accepted": c["k"],
                        "rate": c["k"] / c["n"], "prompts": c["prompts"]}
    return out


def bootstrap_rate_ci(prompts: dict, n_boot: int = 1000, seed: int = 0,
                      alpha: float = 0.05) -> tuple[float, float]:
    """Percentile CI for an acceptance rate, resampling PROMPTS (not tokens).

    Token-level resampling is prohibited leakage-adjacent practice here
    (contract: prompt-grouped everything); prompts are the exchangeable unit.

    Args:
        prompts: {prompt_id: [n_tokens, n_accepted]}.
    Returns:
        (lo, hi) percentile bounds; (0.0, 0.0) for empty input.
    """
    ids = list(prompts)
    if not ids:
        return (0.0, 0.0)
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        n = k = 0
        for _ in ids:
            pn, pk = prompts[ids[rng.randrange(len(ids))]]
            n += pn
            k += pk
        stats.append(k / n if n else 0.0)
    stats.sort()
    lo_i = int((alpha / 2) * (n_boot - 1))
    hi_i = int((1 - alpha / 2) * (n_boot - 1))
    return (stats[lo_i], stats[hi_i])


def atlas_table(tokens, min_n: int = 50, n_boot: int = 1000,
                seed: int = 0) -> list[dict]:
    """Flat atlas rows sorted by acceptance rate, each with a prompt-grouped
    bootstrap CI — the direct input for the I18 atlas figure."""
    rows = []
    for (cat, phase), cell in atlas_cells(tokens, min_n=min_n).items():
        lo, hi = bootstrap_rate_ci(cell["prompts"], n_boot=n_boot, seed=seed)
        rows.append({"category": cat, "phase": phase, "n": cell["n"],
                     "rate": cell["rate"], "ci_lo": lo, "ci_hi": hi,
                     "n_prompts": len(cell["prompts"])})
    rows.sort(key=lambda r: r["rate"])
    return rows
