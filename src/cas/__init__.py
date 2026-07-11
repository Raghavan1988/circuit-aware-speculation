"""Circuit-Aware Speculation (cas): research harness for exact speculative
decoding with per-round action control and mechanistic instrumentation.

See docs/RESEARCH_SPEC.md and docs/EXPERIMENT_CONTRACT.md for scope and locked
protocol. The pure verification logic lives in `cas.commit` (stdlib only); the
model-facing engine in `cas.spec_decode` (requires torch/transformers).
"""

__version__ = "0.0.0"  # no results yet; see docs/CLAIMS_LEDGER.md
