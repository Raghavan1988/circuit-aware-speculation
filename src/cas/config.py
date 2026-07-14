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
    # D021: measurement-only fast path. None = eager execution (default; the
    # hookable capture path for T4/I10). A torch.compile mode string (e.g.
    # "reduce-overhead", "default") wraps the model to cut launch-bound overhead
    # for TIMING only; equivalence must be re-verified before any compiled path
    # produces a scientific result.
    compile_mode: str | None = None


def _pair() -> str:
    """Model pair selector, overridable via CAS_PAIR. Default 'qwen' (primary,
    ungated). 'llama' is the I17 replication pair (gated; needs an HF token)."""
    return os.environ.get("CAS_PAIR", "qwen").lower()


# (target_spec, draft_spec) per pair. Llama revisions are pinned after the first
# resolved download (verify_env prints them); None until then, with a warning.
def _pair_specs():
    dt = _default_dtype()
    if _pair() == "llama":
        # Revisions resolved via HfApi.model_info on 2026-07-13 (hf_check) and
        # pinned per the contract; env overrides retained for re-pinning.
        return (
            ModelSpec("meta-llama/Llama-3.1-8B-Instruct",
                      revision=os.environ.get("CAS_LLAMA_TARGET_REV")
                      or "0e9e39f249a16976918f6564b8830bc894c89659",
                      dtype=dt),
            ModelSpec("meta-llama/Llama-3.2-1B-Instruct",
                      revision=os.environ.get("CAS_LLAMA_DRAFT_REV")
                      or "9213176726f574b556790deb65791e0c5aa438b6",
                      dtype=dt),
        )
    return (
        ModelSpec("Qwen/Qwen2.5-7B-Instruct",
                  revision="a09a35458c702b33eeacc393d103063234e8bc28", dtype=dt),
        ModelSpec("Qwen/Qwen2.5-0.5B-Instruct",
                  revision="7ae557604adf67be50417f59c2c2f167def9a775", dtype=dt),
    )


@dataclass(frozen=True)
class EngineConfig:
    # Pair chosen by CAS_PAIR (D013 primary = Qwen; I17 replication = Llama).
    target: ModelSpec = field(default_factory=lambda: _pair_specs()[0])
    draft: ModelSpec = field(default_factory=lambda: _pair_specs()[1])
    max_new_tokens: int = 256  # D013 default; confirm/revise at first full sweep
    action_lengths: tuple[int, ...] = ACTION_LENGTHS
    seed: int = 0

    def revisions_pinned(self) -> bool:
        return self.target.revision is not None and self.draft.revision is not None
