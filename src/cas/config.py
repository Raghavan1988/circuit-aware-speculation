"""Central configuration for models, decoding, and the action set.

Model revisions are pinned per docs/EXPERIMENT_CONTRACT.md ("Do not use floating
model aliases without a saved revision"). The revision fields start as None and
MUST be filled with the resolved commit SHA that `scripts/verify_env.py` prints
on first download, before any results run. Running the engine with an unpinned
revision emits a loud warning.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _default_dtype() -> str:
    """Model dtype, overridable via CAS_DTYPE for the fp32 equivalence check.

    Canonical runs use bfloat16 (D014). Setting CAS_DTYPE=float32 lets the
    equivalence gate isolate algorithmic correctness from bf16 argmax-tie noise:
    parallel-verify vs. sequential-decode forwards differ at ~1e-2 in bf16 but
    ~1e-7 in fp32, so fp32 should be exactly token-identical if the cache/commit
    logic is correct. Defaults to bfloat16 so nothing changes unless asked.
    """
    return os.environ.get("CAS_DTYPE", "bfloat16")

# Candidate actions per round; `skip` (0) means "do not draft, take one target
# step" (pure autoregressive). See docs/EXPERIMENT_CONTRACT.md "Decoding".
ACTION_LENGTHS: tuple[int, ...] = (0, 1, 2, 3, 4, 6, 8)
SKIP = 0


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    revision: str | None = None  # pin to a commit SHA before results runs
    dtype: str = "bfloat16"
    attn_implementation: str = "eager"  # D014: eager for hookability + determinism


@dataclass(frozen=True)
class EngineConfig:
    # Primary pair (ungated). Replication pair (Llama) is handled separately (I17).
    target: ModelSpec = field(
        default_factory=lambda: ModelSpec(
            "Qwen/Qwen2.5-7B-Instruct",
            revision="a09a35458c702b33eeacc393d103063234e8bc28",
            dtype=_default_dtype(),
        )
    )
    draft: ModelSpec = field(
        default_factory=lambda: ModelSpec(
            "Qwen/Qwen2.5-0.5B-Instruct",
            revision="7ae557604adf67be50417f59c2c2f167def9a775",
            dtype=_default_dtype(),
        )
    )
    max_new_tokens: int = 256  # D013 default; confirm/revise at first full sweep
    action_lengths: tuple[int, ...] = ACTION_LENGTHS
    seed: int = 0

    def revisions_pinned(self) -> bool:
        return self.target.revision is not None and self.draft.revision is not None
