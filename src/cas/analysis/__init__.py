"""Trace-corpus analysis (Task #4: atlas, oracle headroom, baselines).

Pure stdlib over plain dict rows so everything unit-tests locally and runs on
a CPU container against sealed Parquet sweeps. The surface-baseline fitting
(sklearn) lives in scripts run where sklearn is pinned.
"""
from .atlas import atlas_cells, atlas_table, bootstrap_rate_ci  # noqa: F401
from .oracle import (  # noqa: F401
    ACTIONS,
    accepted_for_action,
    action_utilities,
    measured_costs_from_rounds,
    oracle_policy_value,
)
