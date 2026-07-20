"""Shared contract for the generator-critic autoresearch loop (docs/generator_critic.md, D023).

Every autoresearch module imports ONLY this module (plus stdlib / numpy /
sklearn) so the loop's pieces stay independently testable with no cross-module
import cycles:

  * features.py  build_features(spec, acts_by_layer, meta_rows) -> np.ndarray
  * eval.py      incremental_lift(X_base, X_cand, y, groups, ...) -> dict
  * cost.py      bench_feature(transform, sample, ...) -> dict
  * capture_frontier_activations (modal_app.py) writes the frontier artifact

Scope: execution substrate for I23 (pre-round prediction) and I13 (incremental
information). No new scientific scope (D023); locked corpus (D022), tooling
(D015/D021), and measurement-first ordering (D018) are unchanged. Integrity: the
frozen bar is the pre-round surface baseline PREROUND_BASELINE; every candidate
must beat it AND equal-capacity (norm-matched + random) controls, under
prompt-grouped GroupKFold OOF, dev-only. Mechanistic/"circuit" language stays
G2-gated (D020): a predictive survivor is a "diagnostic signal", never a
"circuit", until interventions (I15) pass.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# The frozen bar a pre-round candidate must beat: the hardened pre-round surface
# baseline in cas.analysis.baselines.FEATURE_SETS (measured AUROC ~0.73 on
# Qwen-v1). Never selected on test data.
PREROUND_BASELINE = "preround_hardened"

# Seed families the generator may configure. The generator selects a family +
# params; it NEVER emits free-form code (D023 -- reproducibility + safety).
#   raw     : the frontier hidden vector at the chosen layer(s), concatenated.
#   lowrank : a fixed random/PCA projection to params["k"] dims (cheap statistic).
#   drift   : difference of consecutive frontier reps (params["order"], default 1)
#             -- a dynamical "divergence velocity" precursor.
#   align   : draft<->target frontier agreement (cosine / projection); requires
#             both sides captured (params["other"] = "draft").
#   norm    : scalar norm(s) / surprise-budget summary of the frontier rep.
SEED_FAMILIES = ("raw", "lowrank", "drift", "align", "norm")


@dataclass(frozen=True)
class FeatureSpec:
    """A candidate pre-round signal: a parameterized transform of cached
    verified-context (target frontier) representations at selected layers.

    name:   stable key (used for dedup in the loop + result rows).
    family: one of SEED_FAMILIES.
    layers: hidden-state indices to read (subset of the captured layers).
    params: family-specific config, e.g. {"k": 16}, {"order": 1},
            {"other": "draft"}.
    """

    name: str
    family: str
    layers: tuple[int, ...]
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.family not in SEED_FAMILIES:
            raise ValueError(
                f"unknown family {self.family!r}; SEED_FAMILIES={SEED_FAMILIES}")
        if not self.layers:
            raise ValueError("layers must be non-empty")


# ---- Target-frontier activation artifact contract (capture -> eval) ----------
# capture_frontier_activations writes, under FRONTIER_SUBDIR of a run's probe dir
# (/artifacts/probes/<run_id>/frontier/):
#   frontier_acts_L<L>.npy    : (n_rows, d_model) fp16, row-aligned to metadata
#   frontier_metadata.parquet : one row per (request_id, round r), columns
#                               FRONTIER_META_COLS
#
# Alignment / label semantics (the pre-round premise): the stored vector is the
# TARGET residual stream at the last-committed (frontier) position of the
# verified context that exists BEFORE round r drafts -- i.e. "already cached" at
# ~zero marginal cost. The label is round r's OWN realized acceptance:
#   accept       : bool  (round r accepted_prefix_len >= 1)
#   accepted_len : int   (round r accepted_prefix_len; survival/hazard target)
# So a probe on the frontier rep answers "can we predict this round's acceptance
# before spending any draft compute?" (I23 / C10).
FRONTIER_SUBDIR = "frontier"
FRONTIER_META_COLS = (
    "request_id", "round_id", "split", "domain", "phase",
    "accept", "accepted_len",
)


def frontier_acts_filename(layer: int) -> str:
    return f"frontier_acts_L{layer}.npy"
