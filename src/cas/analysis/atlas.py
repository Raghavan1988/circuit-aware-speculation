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


def bootstrap_delta_ci(prompts_a: dict, prompts_b: dict, n_boot: int = 1000,
                       seed: int = 0, alpha: float = 0.05) -> dict:
    """Paired prompt-grouped bootstrap for a rate CONTRAST rate_a - rate_b.

    Resamples the UNION of prompt ids once per replicate and recomputes both
    rates from the same resampled prompt multiset (a prompt absent from a pool
    contributes nothing to that pool). Prompts are the exchangeable unit
    (contract: prompt-grouped everything); the pairing makes the contrast CI
    honest when the two pools share prompts, which category pools always do.

    Args:
        prompts_a / prompts_b: {prompt_id: [n_tokens, n_accepted]} per pool.
    Returns:
        {delta, lo, hi, p_delta_le_0, n_boot_effective}; delta is the full-pool
        point estimate. Replicates where either pool lands empty are skipped
        (counted out of n_boot_effective). Empty input pools -> zeros.
    """
    def _rate(prompts):
        n = sum(v[0] for v in prompts.values())
        k = sum(v[1] for v in prompts.values())
        return (k / n) if n else 0.0

    ids = sorted(set(prompts_a) | set(prompts_b))
    if not ids or not prompts_a or not prompts_b:
        return {"delta": 0.0, "lo": 0.0, "hi": 0.0, "p_delta_le_0": 1.0,
                "n_boot_effective": 0}
    point = _rate(prompts_a) - _rate(prompts_b)
    rng = random.Random(seed)
    deltas = []
    for _ in range(n_boot):
        na = ka = nb = kb = 0
        for _ in ids:
            pid = ids[rng.randrange(len(ids))]
            if pid in prompts_a:
                pn, pk = prompts_a[pid]
                na += pn
                ka += pk
            if pid in prompts_b:
                pn, pk = prompts_b[pid]
                nb += pn
                kb += pk
        if na and nb:
            deltas.append(ka / na - kb / nb)
    if not deltas:
        return {"delta": point, "lo": 0.0, "hi": 0.0, "p_delta_le_0": 1.0,
                "n_boot_effective": 0}
    deltas.sort()
    lo_i = int((alpha / 2) * (len(deltas) - 1))
    hi_i = int((1 - alpha / 2) * (len(deltas) - 1))
    return {"delta": point, "lo": deltas[lo_i], "hi": deltas[hi_i],
            "p_delta_le_0": sum(1 for d in deltas if d <= 0.0) / len(deltas),
            "n_boot_effective": len(deltas)}


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
