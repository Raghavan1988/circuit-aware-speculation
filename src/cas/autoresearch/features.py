"""Feature-spec registry for the generator-critic loop (I23/I13, D023).

Turns a `FeatureSpec` (docs/generator_critic.md §3.A) into a numeric feature
matrix over captured *target-frontier* representations. This is the parameterized
seed library that replaces hardcoded feature tuples with first-class,
reproducible candidate signals the generator can configure. There is NO
free-form code generation here (D023 -- reproducibility + safety): the generator
selects a family from `SEED_FAMILIES` and a small params dict; nothing more.

Design constraints (mirrors cas.analysis.baselines / scripts.fit_probes style):
  * pure + deterministic -- same spec + same acts -> byte-identical X. Any random
    projection is frozen from (d_model, k, seed), built ONCE, never learned and
    never re-fit per fold (leakage-safe; the critic's equal-capacity control in
    docs/generator_critic.md §4.4 needs the transform fixed across folds).
  * leakage-aware -- the input is the frontier rep already cached BEFORE round r
    drafts (types.py alignment note); no same-round outcome ever enters here.
  * numpy-only -- no torch / sklearn / pyarrow, so feature semantics are
    unit-tested in the local base env (sklearn fitting lives downstream).

Column layout is documented per family below because the eval / cost modules
(eval.py incremental_lift, cost.py bench_feature) and the CLI consume `X`
positionally alongside the equal-dimension norm-matched / random controls.
"""
from __future__ import annotations

import numpy as np

from cas.autoresearch.types import SEED_FAMILIES, FeatureSpec


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _as_f32(arr) -> np.ndarray:
    """Cast a captured (fp16) frontier block to float32.

    Frontier artifacts are stored fp16 (types.py); fp16 *accumulation* overflows
    on d_model-length reductions (norms, dot products), so we promote to float32
    once up front. The fp16 -> float32 cast is exact, so this does not perturb
    determinism.
    """
    return np.asarray(arr, dtype=np.float32)


def _gaussian_projection(d_model: int, k: int, seed: int) -> np.ndarray:
    """Frozen Johnson-Lindenstrauss-style Gaussian random projection.

    A pure function of (d_model, k, seed): `np.random.default_rng(seed)` draws an
    (d_model, k) matrix with entries ~ N(0, 1/k) (the 1/sqrt(k) scaling keeps
    projected norms ~ input norms). This matrix is a frozen constant of the spec
    -- it is NOT learned and NOT re-fit per fold -- so the same spec always yields
    the same projection and the cheap statistic is fully reproducible.
    """
    rng = np.random.default_rng(seed)
    proj = rng.standard_normal((d_model, k)).astype(np.float32)
    proj *= 1.0 / np.sqrt(float(k))
    return proj


def _require_layers(acts_by_layer: dict, layers, n: int) -> None:
    """Every layer must be captured and row-aligned to meta_rows, or ValueError."""
    for layer in layers:
        if layer not in acts_by_layer:
            raise ValueError(
                f"layer {layer} not in acts_by_layer "
                f"(captured layers={sorted(acts_by_layer)})")
        got = np.asarray(acts_by_layer[layer]).shape[0]
        if got != n:
            raise ValueError(
                f"layer {layer} has {got} rows but meta_rows has {n}; "
                "acts must be row-aligned to meta_rows")


# --------------------------------------------------------------------------- #
# per-family builders
# --------------------------------------------------------------------------- #
def _feat_raw(spec: FeatureSpec, acts_by_layer: dict, meta_rows: list, n: int
              ) -> np.ndarray:
    """raw: concat frontier vectors of spec.layers -> (n, sum_L d_model_L).

    For a single model all captured layers share d_model, so d_feat =
    d_model * len(spec.layers). The full residual stream; most expensive family.
    """
    return np.hstack([_as_f32(acts_by_layer[L]) for L in spec.layers])


def _feat_lowrank(spec: FeatureSpec, acts_by_layer: dict, meta_rows: list, n: int
                  ) -> np.ndarray:
    """lowrank: per layer, project to k dims via a frozen Gaussian projection.

    d_feat = k * len(spec.layers). params: {"k": int (required), "seed": int=0}.
    The projection is shared across layers of equal d_model (it is a pure
    function of (d_model, k, seed)); acts differ per layer so the blocks differ.
    A near-zero-cost statistic (one matmul per layer).
    """
    if "k" not in spec.params:
        raise ValueError("lowrank requires params['k']")
    k = int(spec.params["k"])
    if k < 1:
        raise ValueError(f"lowrank params['k'] must be >= 1, got {k}")
    seed = int(spec.params.get("seed", 0))
    proj_cache: dict = {}
    blocks = []
    for L in spec.layers:
        h = _as_f32(acts_by_layer[L])
        d = h.shape[1]
        key = (d, k, seed)
        if key not in proj_cache:
            proj_cache[key] = _gaussian_projection(d, k, seed)
        blocks.append(h @ proj_cache[key])
    return np.hstack(blocks)


def _feat_drift(spec: FeatureSpec, acts_by_layer: dict, meta_rows: list, n: int
                ) -> np.ndarray:
    """drift: frontier_rep(round) - frontier_rep(round - order) per request.

    The "divergence velocity" precursor (docs/generator_critic.md §3.A). Rows are
    grouped by request_id and ordered by round_id first (artifact order is NOT
    per-request contiguous), and the predecessor is the round `order` POSITIONS
    earlier in that per-request sequence (robust to non-consecutive round_ids).
    The first `order` rounds of each request have no predecessor: they emit a
    zero diff and the missing flag is set.

    Column layout, per layer block in spec.layers order:
        [ diff (d_model) , missing_flag (1) ]
    concatenated over layers -> d_feat = sum_L (d_model_L + 1). The missing flag
    is 1.0 when the row was imputed (zero diff) else 0.0; it takes the same value
    across every layer block for a given row (imputation is a property of the
    round position, not the layer), but is emitted per layer so each layer's
    diff carries its own availability channel.

    params: {"order": int=1}.
    """
    order = int(spec.params.get("order", 1))
    if order < 1:
        raise ValueError(f"drift params['order'] must be >= 1, got {order}")

    # group row indices by request_id, then order each group by round_id.
    groups: dict = {}
    for idx, m in enumerate(meta_rows):
        groups.setdefault(m["request_id"], []).append(idx)

    blocks = []
    for L in spec.layers:
        h = _as_f32(acts_by_layer[L])
        d = h.shape[1]
        block = np.zeros((n, d + 1), dtype=np.float32)  # [diff..., flag]
        for idxs in groups.values():
            ordered = sorted(idxs, key=lambda i: meta_rows[i]["round_id"])
            for pos, cur in enumerate(ordered):
                if pos - order >= 0:
                    pred = ordered[pos - order]
                    block[cur, :d] = h[cur] - h[pred]
                    # flag already 0.0
                else:
                    block[cur, d] = 1.0  # imputed: no predecessor
        blocks.append(block)
    return np.hstack(blocks)


def _feat_align(spec: FeatureSpec, acts_by_layer: dict, meta_rows: list, n: int
                ) -> np.ndarray:
    """align: agreement between two layers a, b -> (n, 2).

    Columns: [ cosine(h_a, h_b) , dot(h_a, h_b) / d_model ].
    params: {"layers": (a, b)}; defaults to spec.layers[:2]. Requires >= 2
    layers, both captured and of equal d_model. Cosine of a zero-norm row is 0.0
    (guarded), never NaN.

    NOTE (docs/generator_critic.md §3.A): the intended signal is draft<->target
    frontier alignment, which needs draft-frontier capture (not yet available).
    This layer<->layer variant is the runnable placeholder; the identical
    `build_features` interface extends to the draft<->target variant once the
    draft frontier is captured, via a future params key (e.g. {"other": "draft"}).
    """
    pair = tuple(int(x) for x in spec.params.get("layers", spec.layers[:2]))
    if len(pair) < 2:
        raise ValueError(
            "align requires >= 2 layers (params['layers'] or spec.layers[:2]); "
            f"got {pair}")
    a, b = pair[0], pair[1]
    _require_layers(acts_by_layer, (a, b), n)
    ha = _as_f32(acts_by_layer[a])
    hb = _as_f32(acts_by_layer[b])
    if ha.shape[1] != hb.shape[1]:
        raise ValueError(
            f"align layers {a},{b} have unequal d_model "
            f"({ha.shape[1]} vs {hb.shape[1]})")
    d = ha.shape[1]
    dot = np.sum(ha * hb, axis=1)
    na = np.linalg.norm(ha, axis=1)
    nb = np.linalg.norm(hb, axis=1)
    denom = na * nb
    cosine = np.where(denom > 0.0, dot / np.where(denom > 0.0, denom, 1.0), 0.0)
    dot_scaled = dot / d if d > 0 else dot
    return np.stack([cosine, dot_scaled], axis=1).astype(np.float32)


def _feat_norm(spec: FeatureSpec, acts_by_layer: dict, meta_rows: list, n: int
               ) -> np.ndarray:
    """norm: per layer [l2_norm, mean, std] of the frontier rep -> (n, 3*L).

    A "surprise-budget" / magnitude summary. Column layout, per layer block in
    spec.layers order: [ l2 , mean , std ].
    """
    blocks = []
    for L in spec.layers:
        h = _as_f32(acts_by_layer[L])
        l2 = np.linalg.norm(h, axis=1)
        mean = h.mean(axis=1)
        std = h.std(axis=1)
        blocks.append(np.stack([l2, mean, std], axis=1))
    return np.hstack(blocks).astype(np.float32)


_BUILDERS = {
    "raw": _feat_raw,
    "lowrank": _feat_lowrank,
    "drift": _feat_drift,
    "align": _feat_align,
    "norm": _feat_norm,
}


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def build_features(spec: FeatureSpec,
                   acts_by_layer: dict[int, "np.ndarray"],
                   meta_rows: list[dict]) -> "np.ndarray":
    """Return X of shape (n_rows, d_feat) in meta_rows order for `spec`.

    acts_by_layer: {layer -> ndarray(n_rows, d_model)}, rows aligned to meta_rows
                   (the loaded frontier_acts_L*.npy). Every layer in spec.layers
                   must be present or raise ValueError.
    meta_rows:     list of dicts carrying at least request_id, round_id (for
                   grouping/ordering the `drift` family). Rows are in artifact
                   order.

    Deterministic: same spec + same acts -> identical X. See each `_feat_*`
    docstring for the per-family column layout consumed downstream.
    """
    # FeatureSpec.__post_init__ already validates family + non-empty layers; we
    # re-assert defensively so a hand-built dict-turned-spec can't slip through.
    if spec.family not in SEED_FAMILIES:
        raise ValueError(
            f"unknown family {spec.family!r}; SEED_FAMILIES={SEED_FAMILIES}")
    n = len(meta_rows)
    _require_layers(acts_by_layer, spec.layers, n)
    return _BUILDERS[spec.family](spec, acts_by_layer, meta_rows, n)


def default_seed_specs(layers: tuple[int, ...]) -> list[FeatureSpec]:
    """Round-0 seed set spanning every SEED_FAMILY (docs/generator_critic.md §5).

    `align` is included only when >= 2 layers are supplied (it needs a layer
    pair); every other family is seeded on all `layers`. Names are stable keys
    for the loop's dedup-vs-seen (docs/generator_critic.md §5.3).
    """
    layers = tuple(int(L) for L in layers)
    if not layers:
        raise ValueError("layers must be non-empty")
    specs = [
        FeatureSpec("raw_frontier", "raw", layers),
        FeatureSpec("lowrank_k16_s0", "lowrank", layers, {"k": 16, "seed": 0}),
        FeatureSpec("drift_velocity_o1", "drift", layers, {"order": 1}),
        FeatureSpec("norm_summary", "norm", layers),
    ]
    if len(layers) >= 2:
        specs.append(FeatureSpec(
            "align_layerpair", "align", layers,
            {"layers": (layers[0], layers[1])}))
    return specs
