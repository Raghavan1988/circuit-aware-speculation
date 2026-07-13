"""Model + tokenizer loading (issue I02).

Thin, explicit, and revision-aware. Loads the target and draft with the pinned
revision, requested dtype, and eager attention (D014). The draft and target
share the Qwen2.5 tokenizer family, which the loader asserts so that token ids
from the draft are directly comparable to the target's argmax.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import torch

from .config import EngineConfig, ModelSpec


@dataclass
class LoadedPair:
    target: Any
    draft: Any
    tokenizer: Any
    device: torch.device
    resolved: dict  # model_id -> resolved commit sha, recorded into run metadata


def _load_one(spec: ModelSpec, device: torch.device):
    from transformers import AutoModelForCausalLM

    if spec.revision is None:
        warnings.warn(
            f"{spec.model_id} loaded without a pinned revision; pin the commit "
            f"SHA in cas.config before any results run (contract requirement).",
            stacklevel=2,
        )
    dtype = getattr(torch, spec.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        spec.model_id,
        revision=spec.revision,
        torch_dtype=dtype,
        attn_implementation=spec.attn_implementation,
    ).to(device)
    model.eval()
    resolved = getattr(getattr(model, "config", None), "_commit_hash", None)
    # D021: optional measurement-only fast path. Capture provenance BEFORE
    # compiling (the wrapped module proxies attrs, but read the raw config to be
    # safe). torch.compile is applied last; the default (compile_mode=None) is
    # the unchanged eager, hookable model used for activation capture.
    if getattr(spec, "compile_mode", None):
        model = torch.compile(model, mode=spec.compile_mode)
    return model, (resolved or "unresolved")


def load_pair(cfg: EngineConfig, device: str | torch.device = "cuda") -> LoadedPair:
    """Load target + draft + shared tokenizer.

    Raises:
        RuntimeError: if the two models do not share a tokenizer/vocab, since the
            exact-match verification compares their token ids directly.
    """
    from transformers import AutoTokenizer

    dev = torch.device(device)
    tok = AutoTokenizer.from_pretrained(
        cfg.target.model_id, revision=cfg.target.revision
    )
    draft_tok = AutoTokenizer.from_pretrained(
        cfg.draft.model_id, revision=cfg.draft.revision
    )
    if tok.get_vocab() != draft_tok.get_vocab():
        raise RuntimeError(
            "target and draft tokenizers differ; exact-match verification "
            "requires a shared vocabulary (see docs/EXPERIMENT_CONTRACT.md)."
        )

    target, target_sha = _load_one(cfg.target, dev)
    draft, draft_sha = _load_one(cfg.draft, dev)
    return LoadedPair(
        target=target,
        draft=draft,
        tokenizer=tok,
        device=dev,
        resolved={cfg.target.model_id: target_sha, cfg.draft.model_id: draft_sha},
    )
