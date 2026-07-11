"""Trace records, invariant validation, and the durable writer (I02/I06).

Records are plain dataclasses so the engine can emit them without the storage
layer; `validate` encodes TRACE_SCHEMA.md invariants in pure stdlib;
`TraceWriter` persists validated, immutable Parquet runs.
"""
from .records import (  # noqa: F401
    SCHEMA_VERSION,
    RequestSummary,
    RoundTrace,
    RunMetadata,
    TokenTrace,
    derive_token_traces,
)
from .validate import TraceValidationError, validate_request, validate_round, validate_run  # noqa: F401
from .writer import TraceWriter  # noqa: F401
