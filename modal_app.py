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
    .env({"PYTHONPATH": "/root",
          # long multi-prompt sweeps fragment the allocator; expandable
          # segments keeps reserved-but-unallocated memory reusable
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
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


SWEEP_POLICIES = ("target_only", "skip", "fixed_1", "fixed_2", "fixed_3",
                  "fixed_4", "fixed_6", "fixed_8")


@app.function(image=image, gpu=GPU, volumes=VOLUMES, timeout=6 * 3600)
def run_policy(
    policy: str,
    run_id: str,
    git_commit: str,
    created_at_utc: str,
    max_new_tokens: int = 256,
    cap: int = 0,
    dtype: str = "bfloat16",
) -> dict:
    """I07: run one policy over the frozen prompt set into a sealed trace run.

    `policy` is one of SWEEP_POLICIES, or "all" to loop every policy in one
    container (smoke/cap runs; the full sweep parallelizes via the `sweep`
    local entrypoint). Run target_only FIRST: it writes the reference output
    hashes the spec policies use for per-request equivalence_status. `cap`
    limits prompts for smoke runs (0 = all prompts; smoke artifacts land in
    the same immutable layout but under the caller-chosen run_id).
    """
    import hashlib
    import json
    import os

    os.environ["CAS_DTYPE"] = dtype
    import torch
    import transformers

    from cas.config import EngineConfig
    from cas.models import load_pair
    from cas.spec_decode import SpeculativeDecoder, _forward, fixed_length_policy
    from cas.timing import Stopwatch
    from cas.trace import RequestSummary, RunMetadata, TraceWriter
    from cas.trace.records import SCHEMA_VERSION  # noqa: F401 (stamped via meta)
    import dataclasses as dc

    cfg = EngineConfig(max_new_tokens=max_new_tokens)
    assert cfg.revisions_pinned(), "pin revisions before results runs (contract)"
    pair = load_pair(cfg)
    dec = SpeculativeDecoder(pair, cfg)
    tok = pair.tokenizer

    with open("/artifacts/data/prompts.jsonl") as f:
        rows = [json.loads(line) for line in f]
    with open("/artifacts/data/split_manifest.json") as f:
        manifest = json.load(f)
    assignment = manifest["assignment"]
    manifest_id = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:16]
    if cap:
        rows = rows[:cap]

    def out_hash(ids: list[int]) -> str:  # matches generate()'s format exactly
        return hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()

    @torch.no_grad()  # smoke 2026-07-11 OOMed without this: 256 retained
    def target_only_decode(prompt_ids: list[int]):  # autograd graphs ~38GB
        """Timed pure-autoregressive reference (no spec machinery at all)."""
        from transformers import DynamicCache

        sw = Stopwatch(dec.device)
        cache = DynamicCache()
        ids = torch.tensor([prompt_ids], device=dec.device)
        out: list[int] = []
        with sw.measure("prefill"):
            logits, cache, past = _forward(pair.target, ids, cache, 0)
        eos = tok.eos_token_id
        with sw.measure("decode"):
            for _ in range(max_new_tokens):
                nxt = int(logits[0, -1].argmax())
                out.append(nxt)
                if nxt == eos:
                    break
                step = torch.tensor([[nxt]], device=dec.device)
                logits, cache, past = _forward(pair.target, step, cache, past)
        return out, sw

    ref_path = f"/artifacts/traces/{run_id}/reference_hashes.json"
    policies = list(SWEEP_POLICIES) if policy == "all" else [policy]
    results = {}
    for pol in policies:
        refs = {}
        if pol != "target_only" and os.path.exists(ref_path):
            with open(ref_path) as f:
                refs = json.load(f)
        meta = RunMetadata(
            run_id=f"{run_id}/{pol}",
            created_at_utc=created_at_utc,
            git_commit=git_commit,
            config_hash=hashlib.sha256(repr(cfg).encode()).hexdigest()[:16],
            command=f"modal run modal_app.py::sweep ({pol}, cap={cap})",
            seed=cfg.seed,
            target_model_id=cfg.target.model_id,
            target_revision=cfg.target.revision or "",
            draft_model_id="" if pol == "target_only" else cfg.draft.model_id,
            draft_revision="" if pol == "target_only" else (cfg.draft.revision or ""),
            tokenizer_id=cfg.target.model_id,
            tokenizer_revision=cfg.target.revision or "",
            dtype=cfg.target.dtype,
            quantization=None,
            device_name=torch.cuda.get_device_name(0),
            device_count=torch.cuda.device_count(),
            driver_cuda_framework={
                "torch": torch.__version__,
                "cuda": torch.version.cuda or "",
                "transformers": transformers.__version__,
            },
            policy_name=pol,
            policy_version="1",
            split_manifest_id=manifest_id,
        )
        writer = TraceWriter(f"/artifacts/traces/{run_id}/{pol}", meta)
        hashes = {}
        n_ident = n_div = 0
        for row in rows:
            pid = row["prompt_id"]
            prompt_ids = tok(row["prompt_text"])["input_ids"]
            common = dict(dataset=row["dataset"], domain=row["domain"],
                          split=assignment.get(pid, "unknown"),
                          prompt_hash=row["prompt_hash"])
            if pol == "target_only":
                out, sw = target_only_decode(prompt_ids)
                h = out_hash(out)
                hashes[pid] = h
                summary = RequestSummary(
                    run_id=meta.run_id, request_id=pid, **common,
                    policy_name=pol, prompt_tokens=len(prompt_ids),
                    output_tokens=len(out),
                    termination_reason=("eos" if out and out[-1] == tok.eos_token_id
                                        else "max_new_tokens"),
                    output_token_hash=h, total_drafted=0, total_accepted=0,
                    total_rejected=0, n_rounds=0,
                    prefill_ns=sw.acc.components_ns.get("prefill", 0),
                    decode_ns=sw.acc.components_ns.get("decode", 0),
                    ttft_ns=sw.acc.components_ns.get("prefill", 0),
                    end_to_end_ns=sw.acc.total_ns(),
                    equivalence_status="reference",
                    reference_output_hash=h,
                )
                writer.add_request(summary, [])
            else:
                L = 0 if pol == "skip" else int(pol.split("_")[1])
                res = dec.generate(prompt_ids, fixed_length_policy(L),
                                   request_id=pid, run_id=meta.run_id,
                                   policy_name=pol,
                                   max_new_tokens=max_new_tokens)
                ref = refs.get(pid, "")
                status = "unchecked" if not ref else (
                    "identical" if res.summary.output_token_hash == ref
                    else "diverged")
                n_ident += status == "identical"
                n_div += status == "diverged"
                summary = dc.replace(res.summary, **common,
                                     equivalence_status=status,
                                     reference_output_hash=ref)
                writer.add_request(summary, res.rounds)
        m = writer.finalize()
        if pol == "target_only":
            os.makedirs(os.path.dirname(ref_path), exist_ok=True)
            with open(ref_path, "w") as f:
                json.dump(hashes, f)
        artifacts.commit()
        results[pol] = {"requests": m["n_requests"],
                        "identical": n_ident, "diverged": n_div}
        print(pol, "->", results[pol])
        torch.cuda.empty_cache()  # avoid fragmentation across policies
    return results


@app.local_entrypoint()
def sweep(run_id: str = "", cap: int = 0, max_new_tokens: int = 256,
          serial: bool = False):
    """I07 driver: target_only first (writes references), then the spec
    policies in parallel containers (or one container with --serial for
    smoke/cap runs, which loads the models once)."""
    import datetime
    import subprocess

    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True).stdout.strip() or "unknown"
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    run_id = run_id or f"sweep-{ts[:19].replace(':', '')}"
    print(f"run_id={run_id} git={git[:12]} cap={cap}")
    if serial:
        print(run_policy.remote("all", run_id, git, ts, max_new_tokens, cap))
        return
    print(run_policy.remote("target_only", run_id, git, ts, max_new_tokens, cap))
    handles = [run_policy.spawn(p, run_id, git, ts, max_new_tokens, cap)
               for p in SWEEP_POLICIES[1:]]
    for h in handles:
        print(h.get())


@app.function(image=image, gpu=GPU, volumes=VOLUMES, timeout=3600)
def run_tests(dtype: str = "bfloat16") -> int:
    """Run the full test suite (pure + GPU equivalence) on the GPU.

    `dtype` is passed as a Modal function arg (NOT a local env var, which does
    not cross into the remote container) and injected into the child pytest
    process's environment as CAS_DTYPE. Use `--dtype float32` to run the
    equivalence gate in fp32 and isolate algorithmic correctness from bf16
    argmax-tie noise. Prints the resolved model dtype so it can be verified.
    """
    import os
    import subprocess

    os.environ["CAS_DTYPE"] = dtype  # inherited by the child pytest process

    # Verify the override actually reaches the config the tests will load.
    from cas.config import EngineConfig

    resolved = EngineConfig().target.dtype
    print(f"run_tests: requested dtype={dtype!r} -> EngineConfig.target.dtype={resolved!r}")
    assert resolved == dtype, "CAS_DTYPE did not propagate to EngineConfig"

    return subprocess.call(["python", "-m", "pytest", "-q", "/root/tests"])
