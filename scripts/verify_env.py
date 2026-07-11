"""Environment verification + revision recorder (issue I01).

Prints the exact driver / CUDA / library versions and, crucially, the resolved
commit SHA of each model so the contract's revision-pinning requirement can be
satisfied. Run this first; copy the printed SHAs into cas.config before any
results run.

    python scripts/verify_env.py            # local (will warn if no CUDA)
    modal run modal_app.py::verify_env      # on the H100
"""
from __future__ import annotations

import json
import platform


def collect() -> dict:
    info: dict = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            info["device_name"] = torch.cuda.get_device_name(0)
            info["device_count"] = torch.cuda.device_count()
            info["bf16_supported"] = torch.cuda.is_bf16_supported()
    except Exception as e:  # pragma: no cover
        info["torch_error"] = repr(e)
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except Exception as e:  # pragma: no cover
        info["transformers_error"] = repr(e)
    return info


def resolve_revisions() -> dict:
    """Download config only and report the resolved commit SHA for each model."""
    from huggingface_hub import model_info

    from cas.config import EngineConfig

    cfg = EngineConfig()
    out = {}
    for spec in (cfg.target, cfg.draft):
        try:
            mi = model_info(spec.model_id)
            out[spec.model_id] = mi.sha
        except Exception as e:  # pragma: no cover
            out[spec.model_id] = f"error: {e!r}"
    return out


def tiny_forward_check() -> dict:
    """Load the draft (small) and run a 1-token forward to confirm the stack."""
    import torch

    from cas.config import EngineConfig
    from cas.models import _load_one

    cfg = EngineConfig()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, sha = _load_one(cfg.draft, dev)
    ids = torch.tensor([[1, 2, 3]], device=dev)
    with torch.no_grad():
        out = model(input_ids=ids)
    return {"draft_sha": sha, "logits_shape": list(out.logits.shape)}


if __name__ == "__main__":
    report = {"env": collect()}
    try:
        report["revisions"] = resolve_revisions()
    except Exception as e:
        report["revisions_error"] = repr(e)
    print(json.dumps(report, indent=2))
    print(
        "\nNext: paste the resolved SHAs into src/cas/config.py "
        "(ModelSpec.revision) before any results run."
    )
