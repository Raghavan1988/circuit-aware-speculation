"""Modal entrypoints for the circuit-aware-speculation harness (issue I01).

Phase 1 (custom-harness science) runs entirely here. Phase 2 (serving-engine
integration and load testing) is separate and out of scope for this file --
see PLAN.md and docs/DECISIONS.md (D009/D010).

Environment (D013): Modal + H100 80GB, weights cached on a persistent volume.
Pins are deliberate; re-lock after the first successful run and record in
DECISIONS.md.

Usage:
    modal run modal_app.py::verify_env
    modal run modal_app.py::ingest_data
    modal run modal_app.py::smoke_decode
    modal run modal_app.py::run_tests
"""
from __future__ import annotations

import os

import modal

# --- pinned image (re-lock after first successful run) ---------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "accelerate==1.1.1",
        "datasets==3.1.0",
        "huggingface_hub==0.26.2",
        "pyarrow==18.0.0",
        "scikit-learn==1.5.2",
        "numpy==1.26.4",
        "pytest==8.3.3",
    )
    # src-layout: cas lives under src/cas locally but is mounted flat at /root/cas
    # so that PYTHONPATH=/root makes `cas`, `scripts`, and `tests` all importable
    # (scripts/tests are implicit namespace packages; no __init__.py needed).
    .env({"PYTHONPATH": "/root"})
    .add_local_dir("src/cas", "/root/cas")
    .add_local_dir("scripts", "/root/scripts")
    .add_local_dir("tests", "/root/tests")
)

app = modal.App("circuit-aware-speculation")

# persistent caches: HF weights and generated artifacts survive between runs
hf_cache = modal.Volume.from_name("cas-hf-cache", create_if_missing=True)
artifacts = modal.Volume.from_name("cas-artifacts", create_if_missing=True)
VOLUMES = {"/root/.cache/huggingface": hf_cache, "/artifacts": artifacts}

# GPU tier for model-loading functions. H100 is the canonical runs (needs a
# payment method on file); override with e.g. `CAS_GPU=A10G modal run ...` to run
# the correctness gate on a cheaper 24GB card that credits cover without a card.
# The 7B+0.5B pair in bf16 is ~15GB and fits on A10G/L4 (24GB). T4 is 16GB and
# fp16-only -- avoid it for the bf16 bit-identity gate.
GPU = os.environ.get("CAS_GPU", "H100")


@app.function(image=image, gpu=GPU, volumes=VOLUMES, timeout=3600)
def verify_env() -> dict:
    import sys

    sys.path.insert(0, "/root")
    from scripts.verify_env import collect, resolve_revisions, tiny_forward_check

    report = {"env": collect(), "revisions": resolve_revisions()}
    report["forward_check"] = tiny_forward_check()
    hf_cache.commit()
    print(report)
    return report


@app.function(image=image, volumes=VOLUMES, timeout=7200)  # CPU-only: no GPU needed
def ingest_data() -> dict:
    """Ingest all domains, freeze the prompt-grouped split, write manifests.

    Per-domain size caps are read from cas.data.ingest defaults; not exposed as a
    CLI arg because Modal cannot parse `dict | None` function annotations.
    """
    import json
    import os

    caps = None
    from cas.data.ingest import ingest_all
    from cas.data.splits import PromptRecord, split_by_prompt, split_summary

    rows = ingest_all(caps)
    recs = [
        PromptRecord(r["prompt_id"], r["dataset"], r["domain"], r["prompt_hash"])
        for r in rows
    ]
    assignment = split_by_prompt(recs, dev_fraction=0.5, seed=0)
    summary = split_summary(assignment, recs)

    os.makedirs("/artifacts/data", exist_ok=True)
    with open("/artifacts/data/prompts.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open("/artifacts/data/split_manifest.json", "w") as f:
        json.dump({"seed": 0, "dev_fraction": 0.5,
                   "assignment": assignment, "summary": summary}, f, indent=2)
    artifacts.commit()
    print("ingested", len(rows), "prompts; split summary:", summary)
    return {"n_prompts": len(rows), "summary": summary}


@app.function(image=image, gpu=GPU, volumes=VOLUMES, timeout=3600)
def smoke_decode(prompt: str = "def fibonacci(n):", max_new_tokens: int = 64) -> dict:
    """End-to-end smoke: load the pair, run fixed-length speculative decode, and
    assert bit-identical output vs target-only greedy on this one prompt."""
    from cas.config import EngineConfig
    from cas.models import load_pair
    from cas.spec_decode import SpeculativeDecoder, fixed_length_policy

    cfg = EngineConfig(max_new_tokens=max_new_tokens)
    pair = load_pair(cfg)
    tok = pair.tokenizer
    ids = tok(prompt, return_tensors=None)["input_ids"]

    dec = SpeculativeDecoder(pair, cfg)
    ref = dec.greedy_reference(ids, max_new_tokens)
    res = dec.generate(ids, fixed_length_policy(4), request_id="smoke",
                       policy_name="fixed_4", max_new_tokens=max_new_tokens)
    identical = res.output_ids == ref
    print("identical:", identical, "| accepted/drafted:",
          res.summary.accepted_per_drafted())
    return {
        "identical_to_greedy": identical,
        "output_tokens": len(res.output_ids),
        "accepted_per_drafted": res.summary.accepted_per_drafted(),
        "mean_accepted_len": res.summary.total_accepted / max(res.summary.n_rounds, 1),
    }


@app.function(image=image, gpu=GPU, volumes=VOLUMES, timeout=3600)
def run_tests() -> int:
    """Run the full test suite (pure + GPU equivalence) on the H100."""
    import subprocess

    return subprocess.call(["python", "-m", "pytest", "-q", "/root/tests"])
