"""I15 intervention harness (G2 causal validation) — pure, tool-agnostic logic.

Turns the correlational first-token acceptance direction (I13/I23,
docs/autoresearch_outcomes.md) into a CAUSAL test per the G2 criteria (D020,
AGENTS.md): an intervention that is **layer-specific, dose-responsive, replicated,
and survives random + norm-matched controls** — PLUS the requirement unique to
this finding: the causal effect on acceptance must exceed the induced ENTROPY
change (a causal analog of "beyond entropy"). If the acceptance effect is fully
entropy-mediated, the direction is causally an entropy knob, not a beyond-entropy
mechanism.

This module holds only the pure numpy logic (direction derivation, controls,
dose-response, entropy-mediation, verdict) so it is unit-tested locally; the GPU
steering (forward-hook / nnsight) lives in modal_app.py::intervene. NOTHING here
upgrades language to "circuit"/"mechanism" — that is a human gate AFTER these
tests pass (D020). D015 specifies nnsight for interventions; the runner scaffolds
with transparent forward hooks (auditable, no image change) and nnsight may be
swapped in.
"""
from __future__ import annotations

import numpy as np


def acceptance_direction(states, accept, normalize=True) -> np.ndarray:
    """First-token acceptance direction at a layer: mean(accepted) − mean(rejected)
    frontier state. Points toward acceptance. Derive on DEV only.

    states: (n, d) float frontier reps. accept: (n,) in {0,1}. Returns (d,).
    """
    states = np.asarray(states, dtype=float)
    accept = np.asarray(accept)
    if len(np.unique(accept)) < 2:
        raise ValueError("need both accepted and rejected states to form a direction")
    d = states[accept == 1].mean(axis=0) - states[accept == 0].mean(axis=0)
    n = np.linalg.norm(d)
    return d / n if (normalize and n > 0) else d


def random_control(direction, seed=0) -> np.ndarray:
    """Norm-matched isotropic random control: iid Gaussian scaled to ||direction||.
    Satisfies 'random AND norm-matched' — the effect can't be a bigger push."""
    d = np.asarray(direction, dtype=float)
    rng = np.random.default_rng(seed)
    r = rng.standard_normal(d.shape)
    rn = np.linalg.norm(r)
    return r / rn * np.linalg.norm(d) if rn > 0 else r


def shuffled_control(direction, seed=0) -> np.ndarray:
    """Norm-PRESERVING structural control: permute the direction's coordinates.
    Same norm and same marginal distribution, but the learned coordinate structure
    is destroyed — isolates structure from magnitude/marginals."""
    rng = np.random.default_rng(seed)
    d = np.asarray(direction, dtype=float).copy()
    rng.shuffle(d)
    return d


def dose_response(alphas, accept_rates) -> dict:
    """Acceptance vs steering dose. Returns OLS slope, monotonicity, and the effect
    size = accept_rate(max α) − accept_rate(min α). `alphas` includes 0 (no-op)."""
    a = np.asarray(alphas, dtype=float)
    y = np.asarray(accept_rates, dtype=float)
    slope = float(np.linalg.lstsq(np.vstack([a, np.ones_like(a)]).T, y, rcond=None)[0][0])
    order = np.argsort(a)
    ys = y[order]
    diffs = np.diff(ys)
    monotone = bool(np.all(diffs >= -1e-9) or np.all(diffs <= 1e-9))
    return {"slope": slope, "monotone": monotone, "effect": float(ys[-1] - ys[0])}


def beats_controls(real_effect, control_effects, margin=1e-6) -> bool:
    """Real |dose effect| exceeds every control's |dose effect| (not just push size)."""
    ce = [abs(float(c)) for c in control_effects]
    return bool(abs(float(real_effect)) > (max(ce) if ce else 0.0) + margin)


def entropy_mediation(alpha, accept, entropy) -> dict:
    """The causal 'beyond entropy' test (per-sample). Linear-probability regressions:

        accept ~ alpha            -> total effect  c_total
        accept ~ alpha + entropy  -> direct effect c_direct (α coef | entropy)

    Returns c_total, c_direct, direct_fraction = c_direct/c_total, and
    ``beyond_entropy`` = the direct α effect is same-signed as the total and a
    material fraction of it. direct_fraction ≈ 0 means the acceptance effect is
    fully mediated by the induced entropy change — the direction is causally an
    entropy knob, NOT beyond entropy. (Linear-probability approximation; the GPU
    runner can also report a logistic version.)"""
    a = np.asarray(alpha, dtype=float)
    y = np.asarray(accept, dtype=float)
    e = np.asarray(entropy, dtype=float)
    c_total = float(np.linalg.lstsq(np.vstack([a, np.ones_like(a)]).T, y, rcond=None)[0][0])
    c_direct = float(np.linalg.lstsq(
        np.vstack([a, e, np.ones_like(a)]).T, y, rcond=None)[0][0])
    direct_fraction = c_direct / c_total if abs(c_total) > 1e-12 else 0.0
    beyond = bool(abs(c_direct) > 1e-6 and c_direct * c_total > 0
                  and abs(direct_fraction) > 0.1)
    return {"c_total": c_total, "c_direct": c_direct,
            "direct_fraction": float(direct_fraction), "beyond_entropy": beyond}


def entropy_stratified_effect(alpha, accept, entropy, n_bins=8) -> dict:
    """Robust, nonparametric complement to ``entropy_mediation``: within narrow
    entropy bins (entropy ~constant), does α still move acceptance? Averages the
    per-bin α→accept slope, weighted by bin count. A non-zero pooled within-bin
    slope is beyond-entropy evidence that does NOT depend on the linear-probability
    model being globally well-specified (it survives a step-shaped accept↔entropy
    relationship, which trips the LPM). Prefer this for the causal verdict."""
    a = np.asarray(alpha, dtype=float)
    y = np.asarray(accept, dtype=float)
    e = np.asarray(entropy, dtype=float)
    edges = np.quantile(e, np.linspace(0, 1, n_bins + 1))
    slopes, weights = [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (e >= lo) & ((e <= hi) if i == n_bins - 1 else (e < hi))
        if m.sum() >= 30 and np.std(a[m]) > 1e-9 and len(np.unique(y[m])) > 1:
            # within-bin, still regress OUT the residual entropy variation (coarse
            # bins leave some), then take alpha's partial coefficient.
            X = np.vstack([a[m], e[m], np.ones(int(m.sum()))]).T
            coef = float(np.linalg.lstsq(X, y[m], rcond=None)[0][0])
            slopes.append(coef)
            weights.append(int(m.sum()))
    if not slopes:
        return {"within_bin_slope": None, "n_bins_used": 0, "beyond_entropy": False}
    pooled = float(np.average(slopes, weights=weights))
    return {"within_bin_slope": pooled, "n_bins_used": len(slopes),
            "beyond_entropy": bool(abs(pooled) > 0.01)}     # magnitude threshold


def disruption(alphas, accept_rates) -> dict:
    """Causal metric for the OBSERVED inverted-U (peak-at-0) intervention shape:
    steering away from α=0 in EITHER direction degrades acceptance (perturbing a
    feature direction breaks the behavior it carries). Endpoint/monotone metrics
    are wrong for this shape; use disruption instead. Returns:

      disruption   = accept(α=0) − mean(accept at α≠0)   (>0 if steering breaks it)
      abs_monotone = acceptance decreases monotonically as |α| grows (dose in |α|)
      peak_at_zero = α=0 is the maximum

    The causal test is real disruption ≫ control (random/shuffled) disruption."""
    a = np.asarray(alphas, dtype=float)
    y = np.asarray(accept_rates, dtype=float)
    zi = int(np.argmin(np.abs(a)))
    y0 = float(y[zi])
    others = y[np.arange(len(a)) != zi]
    disrupt = float(y0 - np.mean(others)) if len(others) else 0.0
    absa = np.abs(a)
    levels = sorted(set(np.round(absa, 9).tolist()))
    means = [float(np.mean(y[np.round(absa, 9) == u])) for u in levels]   # |α|=0 first
    return {"disruption": disrupt,
            "abs_monotone": bool(np.all(np.diff(means) <= 1e-9)),
            "peak_at_zero": bool(y0 >= y.max() - 1e-9)}


def g2_verdict(disrupt_real, disrupt_controls, mediation, dose_real=None) -> dict:
    """Causal sub-verdict for the disruption-shaped intervention: steering along the
    direction disrupts acceptance MORE than norm-matched controls, dose-dependently
    (in |α|), beyond the induced entropy change. `disrupt_real`/`disrupt_controls`
    come from ``disruption()``; `dose_real` (optional, from ``dose_response``)
    carries the directional asymmetry as secondary info. Never emits "circuit"/
    "mechanism" language — layer-specificity + replication are a human gate across
    runs (D020)."""
    beats = beats_controls(disrupt_real["disruption"],
                           [d["disruption"] for d in disrupt_controls])
    causal = bool(disrupt_real["disruption"] > 0 and disrupt_real["abs_monotone"]
                  and beats and mediation.get("beyond_entropy", False))
    out = {"disruption": disrupt_real["disruption"],
           "dose_abs_monotone": disrupt_real["abs_monotone"],
           "peak_at_zero": disrupt_real["peak_at_zero"],
           "beats_controls": beats,
           "beyond_entropy": mediation.get("beyond_entropy", False),
           "causal_beyond_entropy": causal,
           "language": "diagnostic signal (G2 not tripped until a human confirms "
                       "layer-specificity + replication, D020)"}
    if dose_real is not None:
        out["directional_asymmetry"] = dose_real.get("effect")
    return out
