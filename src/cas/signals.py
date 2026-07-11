"""Cheap per-step signals from draft logits (entropy, top-1/top-2 margin).

Required by the trace schema (token trace) and used later by the length
controllers and probes. Kept tiny and torch-only so it composes with the engine.
All functions operate on a single-position logit vector or a batch of them.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Shannon entropy (nats) of the softmax over the last dim.

    Args:
        logits: (..., vocab) float tensor.
    Returns:
        (...) entropy per row. Computed in fp32 for numerical stability.
    """
    logp = torch.log_softmax(logits.float(), dim=-1)
    p = logp.exp()
    return -(p * logp).sum(dim=-1)


@torch.no_grad()
def top1_margin_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Probability gap between the top-1 and top-2 tokens (top-1 confidence).

    Args:
        logits: (..., vocab).
    Returns:
        (...) margin = p_top1 - p_top2 in [0, 1].
    """
    p = torch.softmax(logits.float(), dim=-1)
    top2 = p.topk(2, dim=-1).values
    return top2[..., 0] - top2[..., 1]


@torch.no_grad()
def greedy_token(logits: torch.Tensor) -> torch.Tensor:
    """Argmax token id(s) over the last dim."""
    return logits.argmax(dim=-1)
