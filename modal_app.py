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

# HF token for gated weights (the Llama I17 replication pair). Ungated Qwen does
# not need it; attaching is harmless when CAS_PAIR=qwen.
hf_secret = modal.Secret.from_name("huggingface-token")

# The secret exposes HUGGINGFACE_TOKEN, but huggingface_hub only reads HF_TOKEN —
# without this mapping the container is silently unauthenticated (root cause of
# the 2026-07-12/13 gated-repo 401s despite an approved gate). Runs at import
# time inside every container; harmless locally and when no secret is attached.
if os.environ.get("HUGGINGFACE_TOKEN") and not os.environ.get("HF_TOKEN"):
    os.environ["HF_TOKEN"] = os.environ["HUGGINGFACE_TOKEN"]

# GPU tier for model-loading functions. H100 is the canonical runs (needs a
# payment method on file); override with e.g. `CAS_GPU=A10G modal run ...` to run
# the correctness gate on a cheaper 24GB card that credits cover without a card.
# The 7B+0.5B pair in bf16 is ~15GB and fits on A10G/L4 (24GB). T4 is 16GB and
# fp16-only -- avoid it for the bf16 bit-identity gate.
GPU = os.environ.get("CAS_GPU", "H100")


@app.function(image=image, gpu=GPU, volumes=VOLUMES, timeout=3600,
              secrets=[hf_secret])
def verify_env(pair: str = "qwen") -> dict:
    import os
    import sys

    os.environ["CAS_PAIR"] = pair  # selects the pair before config is imported
    sys.path.insert(0, "/root")
    from scripts.verify_env import collect, resolve_revisions, tiny_forward_check

    report = {"pair": pair, "env": collect(), "revisions": resolve_revisions()}
    report["forward_check"] = tiny_forward_check()
    hf_cache.commit()
    print(report)
    return report


@app.local_entrypoint()
def verify(pair: str = "qwen"):
    verify_env.remote(pair)


@app.function(image=image, timeout=600, secrets=[hf_secret])  # CPU-only
def hf_check() -> dict:
    """Diagnose gated-repo auth: which HF env vars the secret exposes, whose
    token it is, and whether the two Llama repos are accessible with it."""
    import os

    keys = sorted(k for k in os.environ
                  if "HF" in k.upper() or "HUGGING" in k.upper())
    print("HF-ish env keys in container:", keys)
    from huggingface_hub import HfApi

    api = HfApi()
    try:
        who = api.whoami()
        print("whoami:", who.get("name"), "| type:", who.get("type"))
    except Exception as e:
        print("whoami FAILED (no usable token found by hub):", repr(e)[:200])
    out = {}
    for rid in ("meta-llama/Llama-3.1-8B-Instruct",
                "meta-llama/Llama-3.2-1B-Instruct"):
        try:
            mi = api.model_info(rid)
            out[rid] = mi.sha
            print(f"{rid} -> ACCESS OK, sha={mi.sha}")
        except Exception as e:
            out[rid] = f"FAIL {repr(e)[:120]}"
            print(f"{rid} -> FAIL: {repr(e)[:160]}")
    return out


@app.local_entrypoint()
def hfcheck():
    hf_check.remote()


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


@app.function(image=image, volumes=VOLUMES, timeout=2 * 3600, secrets=[hf_secret])
def ingest_v2() -> dict:
    """Build the exhaustive Core v2 corpus (CORPUS_PLAN / D022) into
    /artifacts/data_v2/ — the sealed v1 corpus at /artifacts/data/ is untouched.
    Per-loader isolation: a broken source is reported, not fatal."""
    import json
    import os
    from collections import Counter

    from cas.data.ingest import CORPUS_LICENSES, ingest_core_v2
    from cas.data.splits import PromptRecord, split_by_prompt, split_summary

    rows, status = ingest_core_v2()
    recs = [PromptRecord(r["prompt_id"], r["dataset"], r["domain"], r["prompt_hash"])
            for r in rows]
    assignment = split_by_prompt(recs, dev_fraction=0.5, seed=0)  # keyed by prompt_hash
    summary = split_summary(assignment, recs)

    os.makedirs("/artifacts/data_v2", exist_ok=True)
    with open("/artifacts/data_v2/prompts.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open("/artifacts/data_v2/split_manifest.json", "w") as f:
        json.dump({"seed": 0, "dev_fraction": 0.5,
                   "assignment": assignment, "summary": summary}, f, indent=2)
    lines = ["# Corpus v2 — dataset licenses / provenance (D022)", ""]
    for ds, meta in CORPUS_LICENSES.items():
        lines.append(f"- {ds}: SPDX={meta['spdx']}; "
                     f"redistributable={meta['redistributable']}; axis={meta['axis']}")
    lines += ["", "Reference completions are generated in-house with the study's "
              "Qwen2.5-7B/0.5B pair (Apache-2.0), sidestepping output-license terms.",
              "Copyright-text datasets (cnn_dailymail, xsum) must ship row_id only."]
    with open("/artifacts/data_v2/LICENSES.md", "w") as f:
        f.write("\n".join(lines))
    artifacts.commit()

    dom = dict(Counter(r["domain"] for r in rows))
    by_ds = dict(Counter(r["dataset"] for r in rows))
    print("=== per-loader status ==="); print(json.dumps(status, indent=2))
    print("=== domain counts ==="); print(dom)
    print("=== dataset counts ==="); print(by_ds)
    print("=== split summary ==="); print(json.dumps(summary, indent=2))
    return {"n_prompts": len(rows), "by_domain": dom, "by_dataset": by_ds,
            "status": status}


@app.local_entrypoint()
def ingestv2():
    ingest_v2.remote()


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


@app.function(image=image, gpu=GPU, volumes=VOLUMES, timeout=12 * 3600,
              secrets=[hf_secret])
def run_policy(
    policy: str,
    run_id: str,
    git_commit: str,
    created_at_utc: str,
    max_new_tokens: int = 256,
    cap: int = 0,
    dtype: str = "bfloat16",
    corpus: str = "data",  # D022: "data" = v1 (644), "data_v2" = Core v2 (1,494)
    pair: str = "qwen",    # I17: "qwen" (primary) or "llama" (replication)
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
    os.environ["CAS_PAIR"] = pair  # I17: selects Qwen (default) or the Llama pair
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

    with open(f"/artifacts/{corpus}/prompts.jsonl") as f:
        rows = [json.loads(line) for line in f]
    with open(f"/artifacts/{corpus}/split_manifest.json") as f:
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
        # Resume/skip: a sealed policy dir (write-once) is never recomputed, so
        # a killed sweep can be relaunched with the SAME run_id and continues
        # where it stopped (e.g. target_only already done -> not rerun).
        pol_manifest = f"/artifacts/traces/{run_id}/{pol}/MANIFEST.json"
        if os.path.exists(pol_manifest):
            print(f"{pol} -> already sealed, skipping")
            results[pol] = {"skipped": True}
            continue
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
        import time as _time
        t_pol = _time.time()
        for j, row in enumerate(rows):
            if j and j % 50 == 0:  # progress + live throughput (observability)
                rate = (_time.time() - t_pol) / j
                eta = rate * (len(rows) - j)
                print(f"{pol}: {j}/{len(rows)} prompts "
                      f"({rate:.2f}s/prompt, ETA {eta/60:.1f} min)", flush=True)
            pid = row["prompt_id"]
            prompt_ids = tok(row["prompt_text"])["input_ids"]
            common = dict(dataset=row["dataset"], domain=row["domain"],
                          # fix 2026-07-13: the split map is keyed by prompt_hash
                          # (ledger 2026-07-12 data-quality note); keying by
                          # prompt_id stamped every row "unknown" in the v1 sweep
                          split=assignment.get(row["prompt_hash"], "unknown"),
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
        secs = _time.time() - t_pol
        results[pol] = {"requests": m["n_requests"], "identical": n_ident,
                        "diverged": n_div, "secs": round(secs, 1),
                        "s_per_prompt": round(secs / max(len(rows), 1), 2)}
        print(pol, "->", results[pol], flush=True)
        torch.cuda.empty_cache()  # avoid fragmentation across policies
    return results


@app.function(image=image, volumes=VOLUMES, timeout=8 * 3600)  # CPU orchestrator
def run_sweep(run_id: str, git: str, created_at_utc: str,
              max_new_tokens: int = 256, cap: int = 0) -> dict:
    """Server-side I07 orchestrator: runs target_only first (writes the
    reference hashes), then fans the 7 spec policies out to parallel A100
    containers and gathers them — ALL server-side, so a launch with
    `modal run --detach` survives any local disconnect (this single function
    is the one kept alive; the `--detach` "last function only" caveat is met
    because the fan-out is spawned by THIS function, not the local process).
    """
    ref = run_policy.remote("target_only", run_id, git, created_at_utc,
                            max_new_tokens, cap)
    handles = [run_policy.spawn(p, run_id, git, created_at_utc,
                                max_new_tokens, cap)
               for p in SWEEP_POLICIES[1:]]
    results = {"target_only": ref}
    for p, h in zip(SWEEP_POLICIES[1:], handles):
        results[p] = h.get()
    print("sweep complete:", results)
    return results


@app.local_entrypoint()
def sweep(run_id: str = "", cap: int = 0, max_new_tokens: int = 256,
          serial: bool = False):
    """Thin launcher. For the real sweep use `modal run --detach
    modal_app.py::sweep`: it delegates to the server-side `run_sweep`
    orchestrator so the job is disconnect-proof. `--serial` runs all policies
    in one container (smoke/cap runs) via the local path."""
    import datetime
    import subprocess

    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True).stdout.strip() or "unknown"
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    run_id = run_id or f"sweep-{ts[:19].replace(':', '')}"
    print(f"run_id={run_id} git={git[:12]} cap={cap}")
    if serial:
        print(run_policy.remote("all", run_id, git, ts, max_new_tokens, cap))
    else:
        # Delegate orchestration server-side; this .remote() is the single
        # function --detach keeps alive.
        print(run_sweep.remote(run_id, git, ts, max_new_tokens, cap))


@app.function(image=image, gpu=GPU, volumes=VOLUMES, timeout=3600,
              secrets=[hf_secret])
def run_tests(dtype: str = "bfloat16", pair: str = "qwen") -> int:
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
    os.environ["CAS_PAIR"] = pair    # selects Qwen (default) or the Llama pair

    # Verify the overrides actually reach the config the tests will load.
    from cas.config import EngineConfig

    cfg = EngineConfig()
    print(f"run_tests: dtype={dtype!r} pair={pair!r} -> "
          f"target={cfg.target.model_id} @ {cfg.target.dtype}")
    assert cfg.target.dtype == dtype, "CAS_DTYPE did not propagate to EngineConfig"

    return subprocess.call(["python", "-m", "pytest", "-q", "/root/tests"])


@app.local_entrypoint()
def tests(dtype: str = "bfloat16", pair: str = "qwen"):
    print("exit", run_tests.remote(dtype, pair))


@app.function(image=image, volumes=VOLUMES, timeout=2 * 3600)  # CPU-only
def analyze(run_id: str, eval_split: str = "dev") -> dict:
    """T3 measurement analysis over a sealed sweep (oracle headroom, surface
    baselines, acceptance atlas). Pure CPU on the sealed corpus; writes a
    script-generated JSON report to the artifacts volume and returns it.
    """
    from scripts.run_t3_analysis import _print_summary, run

    run_dir = f"/artifacts/traces/{run_id}"
    out_path = f"/artifacts/analysis/{run_id}/t3_report.json"
    rep = run(run_dir, out_path, eval_split)
    _print_summary(rep)
    artifacts.commit()  # persist the written report to the volume
    print(f"\nwrote {out_path}")
    o = rep["t3_1_oracle_headroom"]
    return {"compute_headroom": o["compute_basis"]["headroom"],
            "full_headroom": o["full_basis"]["headroom"]}


@app.local_entrypoint()
def t3(run_id: str = "sweep-2026-07-11T203836", eval_split: str = "dev"):
    """Run T3 analysis remotely on a CPU container."""
    analyze.remote(run_id, eval_split)


@app.function(image=image, volumes=VOLUMES, timeout=1800)  # CPU-only
def sensitivity(run_id: str) -> dict:
    """(a) Oracle-headroom cost-sensitivity: what the headroom becomes under a
    range of hypothetical draft costs, on the sealed match vectors. Pure CPU."""
    import json as _json

    from scripts.run_t3_analysis import cost_sensitivity

    res = cost_sensitivity(f"/artifacts/traces/{run_id}")
    print("=== CONFOUND CHECKS ===")
    print(_json.dumps(res["confounds"], indent=2, sort_keys=True))
    print("\n=== HEADROOM vs HYPOTHETICAL DRAFT COST ===")
    print(f"{'draft ms/tok':>12} {'best fixed L':>13} {'headroom %':>11}")
    for r in res["sweep"]:
        print(f"{r['draft_ms_per_token']:>12} {r['best_fixed_L']:>13} "
              f"{r['headroom_pct']:>11}")
    return res


@app.local_entrypoint()
def sens(run_id: str = "sweep-2026-07-11T203836"):
    sensitivity.remote(run_id)


@app.function(image=image, volumes=VOLUMES, timeout=1800)  # CPU-only
def eval_policies(run_id: str = "sweep-2026-07-11T203836", eval_split: str = "dev",
                  ratio: float = 0.1) -> dict:
    """RQ2: adaptive length-policy evaluation over the sealed traces (fixed vs
    entropy-stop vs history vs oracle; latency-independent efficiency)."""
    from scripts.eval_length_policies import _print, run as run_eval

    res = run_eval(f"/artifacts/traces/{run_id}", eval_split, ratio)
    _print(res)
    os.makedirs(f"/artifacts/analysis/{run_id}", exist_ok=True)
    with open(f"/artifacts/analysis/{run_id}/rq2_length_policies_{eval_split}.json", "w") as f:
        import json as _json
        _json.dump(res, f, indent=2)
    artifacts.commit()
    return res


@app.local_entrypoint()
def evalp(run_id: str = "sweep-2026-07-11T203836", eval_split: str = "dev",
          ratio: float = 0.1):
    eval_policies.remote(run_id, eval_split, ratio)


@app.function(image=image, volumes=VOLUMES, timeout=1800)  # CPU-only
def taxonomy(run_id: str = "sweep-2026-07-11T203836", eval_split: str = "test",
             ratio: float = 0.1, tau: float = 2.0) -> dict:
    """T5.4: error taxonomy (over/under-drafting, calibration) + candidate-set
    ablation over the sealed traces."""
    import json as _json

    from scripts.error_taxonomy import _print_tax, run as run_tax

    res = run_tax(f"/artifacts/traces/{run_id}", eval_split, ratio, tau)
    _print_tax(res)
    os.makedirs(f"/artifacts/analysis/{run_id}", exist_ok=True)
    with open(f"/artifacts/analysis/{run_id}/t5_4_taxonomy_{eval_split}.json", "w") as f:
        _json.dump(res, f, indent=2)
    artifacts.commit()
    return {"candidate_set": res["candidate_set"],
            "monotone": res["calibration_monotone_decreasing"]}


@app.local_entrypoint()
def tax(run_id: str = "sweep-2026-07-11T203836", eval_split: str = "test"):
    taxonomy.remote(run_id, eval_split)


@app.function(image=image, volumes=VOLUMES, timeout=1800)  # CPU-only
def rq2_ci(run_id: str = "sweep-2026-07-11T203836", eval_split: str = "test") -> dict:
    """Prompt-grouped bootstrap CI for the RQ2 headline (entropy-stop vs best
    fixed): the error bars a reviewer asks for first."""
    import json as _json

    from scripts.eval_length_policies import rq2_confidence

    res = rq2_confidence(f"/artifacts/traces/{run_id}", eval_split)
    print(_json.dumps(res, indent=2))
    os.makedirs(f"/artifacts/analysis/{run_id}", exist_ok=True)
    with open(f"/artifacts/analysis/{run_id}/rq2_ci_{eval_split}.json", "w") as f:
        _json.dump(res, f, indent=2)
    artifacts.commit()
    return res


@app.local_entrypoint()
def rq2ci(run_id: str = "sweep-2026-07-11T203836", eval_split: str = "test"):
    rq2_ci.remote(run_id, eval_split)


@app.function(image=image, volumes=VOLUMES, timeout=2 * 3600, secrets=[hf_secret])
def lhf(run_id: str, tokenizer_id: str = "Qwen/Qwen2.5-7B-Instruct",
        tokenizer_rev: str = "a09a35458c702b33eeacc393d103063234e8bc28") -> dict:
    """Low-hanging-fruit analyses (#1,2,3,5,6) over a sealed fixed_8 run. Pure CPU.
    #4 transfer = compare `calibration_audit.eff_at_tuned_2.0` (Qwen's tau) to this
    run's own `eff_at_tau_star` (its optimum) — for the Llama/v2 runs that is the
    zero-shot cross-family / cross-corpus transfer of the controller."""
    import json as _json

    from scripts.lhf_analysis import run_all

    res = run_all(f"/artifacts/traces/{run_id}", tokenizer_id=tokenizer_id,
                  tokenizer_rev=tokenizer_rev or None)
    tag = run_id.replace("/", "_")
    with open(f"/artifacts/analysis/lhf_{tag}.json", "w") as f:
        _json.dump(res, f, indent=2)
    artifacts.commit()
    print(_json.dumps(res, indent=2))
    return {"pre_round_gate": res["pre_round_gate"],
            "calibration_audit": res["calibration_audit"],
            "skip_economics": res["skip_economics"],
            "category": res.get("category")}


@app.local_entrypoint()
def lhfrun(run_id: str = "sweep-2026-07-11T203836",
           tokenizer_id: str = "Qwen/Qwen2.5-7B-Instruct",
           tokenizer_rev: str = "a09a35458c702b33eeacc393d103063234e8bc28"):
    lhf.remote(run_id, tokenizer_id, tokenizer_rev)


@app.function(image=image, volumes=VOLUMES, timeout=1800)  # CPU-only
def routing_opp(run_id: str = "sweep-2026-07-11T203836", ratio: float = 0.1) -> dict:
    """RQ3 routing-opportunity scoping (per-domain acceptance + optimal length)."""
    from scripts.eval_length_policies import _print_routing, routing_opportunity

    res = routing_opportunity(f"/artifacts/traces/{run_id}", ratio)
    _print_routing(res)
    with open(f"/artifacts/analysis/{run_id}/rq3_routing_opportunity.json", "w") as f:
        import json as _json
        _json.dump(res, f, indent=2)
    artifacts.commit()
    return res


@app.local_entrypoint()
def routingopp(run_id: str = "sweep-2026-07-11T203836", ratio: float = 0.1):
    routing_opp.remote(run_id, ratio)


@app.function(image=image, gpu=GPU, volumes=VOLUMES, timeout=3600)
def probe_draft(domain: str = "code", cap_prompts: int = 40, max_new: int = 128,
                drafts: str = "Qwen/Qwen2.5-0.5B-Instruct,Qwen/Qwen2.5-Coder-0.5B-Instruct"
                ) -> dict:
    """RQ3 go/no-go: does a domain-specialized draft accept better than the
    general draft on its home domain? Apples-to-apples: generate the SAME target
    greedy continuation per prompt, then score each candidate draft's per-token
    argmax agreement (the acceptance condition) on it. Feasibility probe: drafts
    unpinned; tokenizer base-id alignment is verified (required for exact spec).
    """
    import json

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_grad_enabled(False)
    T_ID = "Qwen/Qwen2.5-7B-Instruct"
    T_REV = "a09a35458c702b33eeacc393d103063234e8bc28"
    ttok = AutoTokenizer.from_pretrained(T_ID, revision=T_REV)
    target = AutoModelForCausalLM.from_pretrained(
        T_ID, revision=T_REV, torch_dtype=torch.bfloat16,
        attn_implementation="eager").to("cuda").eval()
    tv = ttok.get_vocab()

    draft_ids = [d.strip() for d in drafts.split(",")]
    models, compat = {}, {}
    for did in draft_ids:
        try:
            dtok = AutoTokenizer.from_pretrained(did)
            dv = dtok.get_vocab()
            shared = set(tv) & set(dv)
            aligned = all(tv[s] == dv[s] for s in shared)
            compat[did] = {"target_vocab": len(tv), "draft_vocab": len(dv),
                           "shared": len(shared), "base_ids_aligned": bool(aligned),
                           "extra_in_draft": len(dv) - len(shared)}
            models[did] = AutoModelForCausalLM.from_pretrained(
                did, torch_dtype=torch.bfloat16,
                attn_implementation="eager").to("cuda").eval()
        except Exception as e:  # keep going; one bad id must not kill the probe
            compat[did] = {"error": repr(e)[:200]}
            print(f"skip {did}: {e!r}", flush=True)

    with open("/artifacts/data/prompts.jsonl") as f:
        rows = [json.loads(l) for l in f]
    prompts = [r for r in rows if r["domain"] == domain][:cap_prompts]

    import random

    stats = {d: {"match": 0, "total": 0} for d in models}
    per_prompt = {d: [] for d in models}   # per-prompt (match, total) for grouping
    for i, r in enumerate(prompts):
        ids = ttok(r["prompt_text"])["input_ids"]
        t = torch.tensor([ids], device="cuda")
        gen = target.generate(t, max_new_tokens=max_new, do_sample=False,
                              pad_token_id=ttok.eos_token_id)
        cont = gen[0, len(ids):].tolist()
        if not cont:
            continue
        full = torch.tensor([ids + cont], device="cuda")
        base = len(ids) - 1
        for did, m in models.items():
            logits = m(input_ids=full, use_cache=False).logits
            mp = sum(int(int(logits[0, base + j].argmax()) == tok_id)
                     for j, tok_id in enumerate(cont))
            stats[did]["match"] += mp
            stats[did]["total"] += len(cont)
            per_prompt[did].append((mp, len(cont)))
        if (i + 1) % 20 == 0:
            print(f"{i+1}/{len(prompts)} prompts", flush=True)

    acceptance = {d: (stats[d]["match"] / stats[d]["total"] if stats[d]["total"] else None)
                  for d in models}

    # Prompt-grouped paired comparison: the honest unit is the prompt, not the
    # token. Report per-prompt paired difference (specialist - general) with a
    # prompt-level bootstrap 95% CI. If the CI includes 0, 40-ish prompts cannot
    # distinguish the drafts (no clear winner).
    paired = None
    if len(models) == 2:
        a, b = list(models)  # a=general (first), b=specialist (second)
        diffs = [pb / tb - pa / ta
                 for (pa, ta), (pb, tb) in zip(per_prompt[a], per_prompt[b])
                 if ta and tb]
        rng = random.Random(0)
        boots = []
        for _ in range(2000):
            s = [diffs[rng.randrange(len(diffs))] for _ in diffs]
            boots.append(sum(s) / len(s))
        boots.sort()
        paired = {"specialist_minus_general": b + " - " + a,
                  "n_prompts_paired": len(diffs),
                  "mean_diff": sum(diffs) / len(diffs),
                  "ci95": [boots[49], boots[1949]],
                  "median_tokens_per_prompt": sorted(t for _, t in per_prompt[a])[len(per_prompt[a]) // 2]}

    res = {"domain": domain, "n_prompts": len(prompts), "max_new": max_new,
           "tokens_scored_per_draft": {d: stats[d]["total"] for d in models},
           "tokenizer_compat": compat, "acceptance": acceptance,
           "paired_prompt_grouped": paired,
           "sweep_general_code_block8_acceptance": 0.523}
    print(json.dumps(res, indent=2))
    with open(f"/artifacts/analysis/rq3_draft_probe_{domain}.json", "w") as f:
        json.dump(res, f, indent=2)
    artifacts.commit()
    return res


@app.local_entrypoint()
def draftprobe(domain: str = "code", cap_prompts: int = 40,
               drafts: str = "Qwen/Qwen2.5-0.5B-Instruct,Qwen/Qwen2.5-Coder-0.5B-Instruct"):
    probe_draft.remote(domain, cap_prompts, drafts=drafts)


# The RQ3 draft pool: general + specialists, with SIZE-MATCHED controls (Math has
# no 0.5B, so general-1.5B is included to isolate specialization from raw size).
DRAFT_POOL = (
    "Qwen/Qwen2.5-0.5B-Instruct,"        # general, 0.5B
    "Qwen/Qwen2.5-Coder-0.5B-Instruct,"  # code, 0.5B
    "Qwen/Qwen2.5-1.5B-Instruct,"        # general, 1.5B (size control)
    "Qwen/Qwen2.5-Math-1.5B-Instruct,"   # math, 1.5B
    "Qwen/Qwen2.5-Coder-1.5B-Instruct"   # code, 1.5B (size-matched code)
)


@app.function(image=image, gpu=GPU, volumes=VOLUMES, timeout=6 * 3600,
              secrets=[hf_secret])
def draft_matrix(max_new: int = 128, cap_per_domain: int = 0,
                 drafts: str = DRAFT_POOL, corpus: str = "data",
                 target: str = "Qwen/Qwen2.5-7B-Instruct",
                 target_rev: str = "a09a35458c702b33eeacc393d103063234e8bc28"
                 ) -> dict:
    """Exhaustive RQ3 study: draft x domain acceptance matrix vs one Qwen2.5-7B
    target. Generates the target greedy continuation ONCE per prompt, scores each
    candidate draft's per-token argmax agreement (drift-free acceptance) on the
    identical continuation, loading drafts one at a time (memory-safe). Reports
    the matrix plus size-matched specialist-vs-general paired diffs with
    prompt-grouped bootstrap CIs, and the oracle-router ceiling per domain.
    """
    import json
    import random

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_grad_enabled(False)
    T_ID, T_REV = target, (target_rev or None)
    ttok = AutoTokenizer.from_pretrained(T_ID, revision=T_REV)
    tv = ttok.get_vocab()

    with open(f"/artifacts/{corpus}/prompts.jsonl") as f:
        rows = [json.loads(l) for l in f]
    if cap_per_domain:
        seen: dict = {}
        capped = []
        for r in rows:
            d = r["domain"]
            if seen.get(d, 0) < cap_per_domain:
                capped.append(r); seen[d] = seen.get(d, 0) + 1
        rows = capped

    # 1) target greedy continuation once per prompt (sdpa for speed; argmax
    #    acceptance is robust to attn backend, and all drafts see the same conts)
    target = AutoModelForCausalLM.from_pretrained(
        T_ID, revision=T_REV, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa").to("cuda").eval()
    conts = []  # (domain, ids, cont)
    for i, r in enumerate(rows):
        ids = ttok(r["prompt_text"])["input_ids"]
        gen = target.generate(torch.tensor([ids], device="cuda"),
                              max_new_tokens=max_new, do_sample=False,
                              pad_token_id=ttok.eos_token_id)
        conts.append((r["domain"], ids, gen[0, len(ids):].tolist()))
        if (i + 1) % 100 == 0:
            print(f"target gen {i+1}/{len(rows)}", flush=True)
    del target
    torch.cuda.empty_cache()

    domains = sorted({d for d, _, _ in conts})
    draft_ids = [d.strip() for d in drafts.split(",")]
    # per[draft][domain] = list of (match, total) per prompt (prompt-grouped)
    per: dict = {}
    compat: dict = {}
    for did in draft_ids:
        try:
            dv = AutoTokenizer.from_pretrained(did).get_vocab()
            aligned = all(tv[s] == dv[s] for s in (set(tv) & set(dv)))
            compat[did] = {"vocab": len(dv), "aligned": bool(aligned)}
            if not aligned:
                print(f"{did}: tokenizer NOT aligned -> skip (exact spec impossible)")
                continue
            m = AutoModelForCausalLM.from_pretrained(
                did, torch_dtype=torch.bfloat16,
                attn_implementation="sdpa").to("cuda").eval()
        except Exception as e:
            compat[did] = {"error": repr(e)[:200]}
            print(f"skip {did}: {e!r}", flush=True)
            continue
        per[did] = {d: [] for d in domains}
        for dom, ids, cont in conts:
            if not cont:
                continue
            full = torch.tensor([ids + cont], device="cuda")
            base = len(ids) - 1
            logits = m(input_ids=full, use_cache=False).logits
            mp = sum(int(int(logits[0, base + j].argmax()) == t)
                     for j, t in enumerate(cont))
            per[did][dom].append((mp, len(cont)))
        del m
        torch.cuda.empty_cache()
        print(f"scored draft {did}", flush=True)

    def acc(did, dom):
        rows_ = per[did][dom]
        tot = sum(t for _, t in rows_)
        return (sum(mm for mm, _ in rows_) / tot) if tot else None

    def paired_ci(spec, gen, dom):
        a, b = per[spec][dom], per[gen][dom]
        diffs = [ps / ts - pg / tg for (ps, ts), (pg, tg) in zip(a, b) if ts and tg]
        if not diffs:
            return None
        rng = random.Random(0)
        boots = sorted(sum((diffs[rng.randrange(len(diffs))] for _ in diffs)) / len(diffs)
                       for _ in range(2000))
        return {"n_prompts": len(diffs), "mean_diff": sum(diffs) / len(diffs),
                "ci95": [boots[49], boots[1949]]}

    matrix = {did: {d: acc(did, d) for d in domains} for did in per}
    # size-matched specialist vs general comparisons (spec, general, domain)
    comps = {}
    for spec, gen, dom in [
        ("Qwen/Qwen2.5-Coder-0.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct", "code"),
        ("Qwen/Qwen2.5-Coder-1.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "code"),
        ("Qwen/Qwen2.5-Math-1.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "math"),
    ]:
        if spec in per and gen in per and dom in domains:
            comps[f"{spec.split('/')[-1]} vs {gen.split('/')[-1]} @ {dom}"] = \
                paired_ci(spec, gen, dom)
    # generic: each subsequent draft vs the FIRST draft, per domain (pair-agnostic
    # paired CI). Used for the Llama draft-SIZE supplementary comparison
    # (3B vs 1B) where the Qwen-specific specialist pairs above don't apply.
    order = list(per)
    vs_first = {}
    if len(order) >= 2:
        base = order[0]
        for did in order[1:]:
            for d in domains:
                vs_first[f"{did.split('/')[-1]} vs {base.split('/')[-1]} @ {d}"] = \
                    paired_ci(did, base, d)

    # oracle router: best draft per domain vs the general-0.5B incumbent
    oracle = {}
    for d in domains:
        best = max((did for did in per if matrix[did][d] is not None),
                   key=lambda did: matrix[did][d], default=None)
        oracle[d] = {"best_draft": best.split('/')[-1] if best else None,
                     "best_acc": matrix[best][d] if best else None}

    res = {"corpus": corpus, "target": target, "max_new": max_new,
           "n_prompts": len(conts), "domains": domains,
           "tokenizer_compat": compat, "matrix": matrix,
           "size_matched_comparisons": comps, "vs_first_draft": vs_first,
           "oracle_router": oracle}

    # ---- print ----
    short = lambda s: s.split('/')[-1].replace('Qwen2.5-', '')
    print(f"\n=== RQ3 DRAFT x DOMAIN ACCEPTANCE MATRIX (corpus={corpus}, "
          f"drift-free, vs 7B target) ===")
    hdr = f"{'draft':<26}" + "".join(f"{d:>9}" for d in domains)
    print(hdr)
    for did in per:
        print(f"{short(did):<26}" + "".join(
            f"{matrix[did][d]:>9.3f}" if matrix[did][d] is not None else f"{'-':>9}"
            for d in domains))
    print(f"{'ORACLE ROUTER':<26}" + "".join(
        f"{oracle[d]['best_acc']:>9.3f}" for d in domains))
    print("\n=== SIZE-MATCHED specialist vs general (paired, prompt-grouped 95% CI) ===")
    for k, v in comps.items():
        if v:
            sig = "SIGNIFICANT" if (v['ci95'][0] > 0 or v['ci95'][1] < 0) else "n.s. (tie)"
            print(f"  {k}: diff={v['mean_diff']:+.4f} "
                  f"CI[{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}] n={v['n_prompts']} -> {sig}")
    print("\n=== ORACLE ROUTER best draft per domain ===")
    for d in domains:
        print(f"  {d}: {oracle[d]['best_draft']} @ {oracle[d]['best_acc']:.3f}")
    if vs_first:
        print("\n=== vs FIRST draft (paired, prompt-grouped 95% CI) ===")
        for k, v in vs_first.items():
            if v:
                sig = "SIGNIF" if (v['ci95'][0] > 0 or v['ci95'][1] < 0) else "n.s."
                print(f"  {k}: diff={v['mean_diff']:+.4f} "
                      f"CI[{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}] n={v['n_prompts']} {sig}")

    # tag artifact by corpus AND target so Qwen-v1, v2, and Llama runs don't clobber
    tgt_tag = "" if target.startswith("Qwen/Qwen2.5-7B") else "_" + target.split("/")[-1]
    suffix = ("" if corpus == "data" else f"_{corpus}") + tgt_tag
    with open(f"/artifacts/analysis/rq3_draft_matrix{suffix}.json", "w") as f:
        json.dump(res, f, indent=2)
    artifacts.commit()
    return res


@app.local_entrypoint()
def draftmatrix(max_new: int = 128, cap_per_domain: int = 0, corpus: str = "data",
                drafts: str = DRAFT_POOL,
                target: str = "Qwen/Qwen2.5-7B-Instruct",
                target_rev: str = "a09a35458c702b33eeacc393d103063234e8bc28"):
    draft_matrix.remote(max_new, cap_per_domain, drafts=drafts, corpus=corpus,
                        target=target, target_rev=target_rev)


@app.function(image=image, gpu=GPU, volumes=VOLUMES, timeout=3600)
def bench_draft(run_id: str = "sweep-2026-07-11T203836", context_len: int = 128,
                n_warmup: int = 5, n_iter: int = 25, gap: int = 5,
                attn: str = "eager", compile_mode: str = "") -> dict:
    """T3.4: clean draft/verify latency, all actions in ONE container (removes
    the cross-container jitter confound C2), decomposed into fixed (gap catch-up)
    and per-token cost, under four configs:

      A as-run   : per-token entropy+margin+argmax with float()/int() syncs
      B sig-off  : per-token argmax sync only (int to feed next token)
      C batched  : NO per-token sync (argmax stays on-device); signals batched
                   once after the loop  <- deployment-realistic
      D floor    : C without any signals (pure incremental decode floor)

    Then rebuilds the oracle cost profile from the measured *config-C* costs and
    recomputes headroom on the sealed fixed_8 match vectors -> the real M3 number.
    """
    import json

    import torch
    from transformers import DynamicCache

    from cas.analysis.oracle import oracle_policy_value
    from cas.config import EngineConfig
    from cas.models import load_pair
    from cas.signals import (entropy_from_logits, greedy_token,
                             top1_margin_from_logits)
    from cas.spec_decode import _forward
    from scripts.run_t3_analysis import _match_vectors, _read_parquet

    import dataclasses

    torch.set_grad_enabled(False)
    ACTIONS = [1, 2, 3, 4, 6, 8]
    cfg = EngineConfig()
    # T3.4-b: override attention impl for the timing fork (does NOT touch the
    # eager capture path; this is a benchmark-only config). "sdpa"/"flash_
    # attention_2" fuse attention kernels -> fewer launches -> tests launch-bound.
    cmode = compile_mode or None  # "" -> None (eager, no compile)
    if attn != "eager" or cmode:
        cfg = dataclasses.replace(
            cfg,
            target=dataclasses.replace(
                cfg.target, attn_implementation=attn, compile_mode=cmode),
            draft=dataclasses.replace(
                cfg.draft, attn_implementation=attn, compile_mode=cmode),
        )
    pair = load_pair(cfg)
    dev = pair.device
    print(f"attn_implementation={attn}  compile_mode={cmode}  "
          f"draft.attn={cfg.draft.attn_implementation}")

    with open("/artifacts/data/prompts.jsonl") as f:
        row = json.loads(f.readline())
    ids = pair.tokenizer(row["prompt_text"])["input_ids"][:context_len]
    if len(ids) < 8:
        ids = (ids * 8)[:context_len]
    ctx = torch.tensor([ids], device=dev)

    def median_ms(fn):
        for _ in range(n_warmup):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(n_iter):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record(); fn(); e.record(); torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        ts.sort()
        return ts[len(ts) // 2]

    # warm caches
    d_cache = DynamicCache()
    d_logits, d_cache, d_ctx = _forward(pair.draft, ctx, d_cache, 0)
    cur0 = d_logits[0, -1].clone()
    t_cache = DynamicCache()
    _, t_cache, t_ctx = _forward(pair.target, ctx, t_cache, 0)

    def draft_loop(L, mode):
        cur = cur0
        stack = []
        for i in range(L):
            if mode == "A":
                _ = float(entropy_from_logits(cur))
                _ = float(top1_margin_from_logits(cur))
            if mode in ("A", "B"):
                tok = int(greedy_token(cur))               # host sync
                nxt = torch.tensor([[tok]], device=dev)
            else:                                          # C, D: no sync
                nxt = greedy_token(cur).view(1, 1)
            dl, _c, _ = _forward(pair.draft, nxt, d_cache, d_ctx + i)
            cur = dl[0, -1]
            if mode == "C":
                stack.append(cur)
        if mode == "C" and stack:
            st = torch.stack(stack)
            _ = entropy_from_logits(st)                    # batched, one shot
            _ = top1_margin_from_logits(st)
        d_cache.crop(d_ctx)

    def gap_forward():
        _dl, _c, _ = _forward(pair.draft, ctx[:, :gap], d_cache, d_ctx)
        d_cache.crop(d_ctx)

    def verify(n):
        def f():
            _l, _c, _ = _forward(pair.target, ctx[:, :n], t_cache, t_ctx)
            t_cache.crop(t_ctx)
        return f

    gap_ms = median_ms(gap_forward)
    verify_ms = {L: median_ms(verify(L + 1)) for L in ACTIONS}
    verify0_ms = median_ms(verify(1))
    draft = {m: {L: median_ms(lambda L=L, m=m: draft_loop(L, m))
                 for L in ACTIONS} for m in ("A", "B", "C", "D")}

    # per-token draft cost per config (for the report)
    per_tok = {m: {L: draft[m][L] / L for L in ACTIONS} for m in draft}

    # --- rebuild oracle cost profile from config C (deployment) -----------
    def costs_from(mode):
        c = {0: verify0_ms * 1e6}
        for L in ACTIONS:
            c[L] = (verify_ms[L] + gap_ms + draft[mode][L]) * 1e6
        return c

    matches = _match_vectors(
        _read_parquet(f"/artifacts/traces/{run_id}/fixed_8/rounds.parquet"))
    headroom = {}
    for mode in ("A", "C", "D"):
        c = costs_from(mode)
        r = oracle_policy_value(matches, c, actions=tuple(sorted(c)))
        headroom[mode] = {"best_fixed_L": r["best_fixed"][0],
                          "headroom_pct": round(r["headroom"] * 100, 2),
                          "fixed_tpc": {int(k): r["fixed"][k] for k in r["fixed"]}}

    print(f"\nctx_len={ctx.shape[1]}  gap_catchup={gap_ms:.2f}ms  "
          f"verify(skip,1pos)={verify0_ms:.2f}ms")
    print(f"{'L':>3} | {'verify':>7} | " + " | ".join(f"{m}ms/tok" for m in "ABCD"))
    for L in ACTIONS:
        print(f"{L:>3} | {verify_ms[L]:>7.2f} | " +
              " | ".join(f"{per_tok[m][L]:>6.2f}" for m in "ABCD"))
    print("\n=== REAL M3 HEADROOM (sealed fixed_8 labels, measured costs) ===")
    for mode in ("A", "C", "D"):
        h = headroom[mode]
        gate = "PROCEED" if h["headroom_pct"] >= 5 else "STOP"
        print(f"  [{mode}] best_fixed L={h['best_fixed_L']:>2}  "
              f"headroom={h['headroom_pct']:>6.2f}%  -> {gate}")

    out = {"attn": attn, "compile_mode": cmode, "gap_ms": gap_ms,
           "verify0_ms": verify0_ms, "verify_ms": verify_ms, "draft_ms": draft,
           "per_tok_ms": per_tok, "headroom": headroom,
           "context_len": ctx.shape[1], "n_iter": n_iter}
    tag = f"{attn}" + (f"_{cmode.replace('-', '')}" if cmode else "")
    path = f"/artifacts/analysis/{run_id}/t3_4_bench_{tag}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    artifacts.commit()
    print(f"\nwrote {path}")
    return headroom


@app.local_entrypoint()
def bench(run_id: str = "sweep-2026-07-11T203836", attn: str = "eager",
          compile_mode: str = ""):
    bench_draft.remote(run_id, attn=attn, compile_mode=compile_mode)


@app.function(image=image, gpu=GPU, volumes=VOLUMES, timeout=6 * 3600)
def capture_activations(run_id: str = "sweep-2026-07-11T203836",
                        cap_prompts: int = 120, split: str = "dev") -> dict:
    """I10: teacher-forced draft residual-stream capture at early/mid/late/final
    layers, aligned to the sealed fixed_8 proposals, reusing the frozen
    acceptance labels. Writes acts_L*.npy + metadata.parquet to the volume.

    Eager attention (D014/D015) keeps this the hookable path; we read the
    residual stream via output_hidden_states (numerically the real model).
    """
    import json
    import os

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch

    from cas.annotate.phases import annotate_phase
    from cas.capture import DEFAULT_LAYERS, generating_positions
    from cas.config import EngineConfig
    from cas.models import load_pair

    torch.set_grad_enabled(False)
    cfg = EngineConfig()
    pair = load_pair(cfg)
    draft, tok, dev = pair.draft, pair.tokenizer, pair.device

    base = f"/artifacts/traces/{run_id}"
    rounds = pq.read_table(f"{base}/fixed_8/rounds.parquet").to_pylist()
    summ = pq.read_table(f"{base}/fixed_8/request_summaries.parquet").to_pylist()
    with open("/artifacts/data/split_manifest.json") as f:
        assignment = json.load(f)["assignment"]          # prompt_hash -> split
    split_of = {s["request_id"]: assignment.get(s["prompt_hash"], "unknown")
                for s in summ}
    domain_of = {s["request_id"]: s["domain"] for s in summ}
    with open("/artifacts/data/prompts.jsonl") as f:
        prompt_text = {json.loads(l)["prompt_id"]: json.loads(l)["prompt_text"]
                       for l in (ln for ln in f)}

    by_req: dict = {}
    for r in rounds:
        by_req.setdefault(r["request_id"], []).append(r)
    reqs = [rid for rid in by_req
            if split_of.get(rid) == split and rid in prompt_text]
    reqs.sort()
    reqs = reqs[:cap_prompts]

    acts = {L: [] for L in DEFAULT_LAYERS}
    meta = []
    n_aligned = n_tok = 0
    for j, rid in enumerate(reqs):
        prompt_ids = tok(prompt_text[rid])["input_ids"]
        # The engine commits one prefill token (target greedy argmax after the
        # prompt) BEFORE round 0, so generated[0] is that token and round 0 has
        # start_output_pos == 1. Recompute it (deterministic prefill argmax, same
        # op as the sweep) so the committed prefix matches the sealed run exactly.
        t_ids = torch.tensor([prompt_ids], device=dev)
        first_tok = int(pair.target(input_ids=t_ids, use_cache=False).logits[0, -1].argmax())
        prefix_ids = list(prompt_ids) + [first_tok]  # grows by each round's emissions
        for r in sorted(by_req[rid], key=lambda x: x["round_id"]):
            if r["start_output_pos"] != len(prefix_ids) - len(prompt_ids):
                raise ValueError(f"{rid} r{r['round_id']} emit/pos mismatch "
                                 f"({r['start_output_pos']} vs {len(prefix_ids) - len(prompt_ids)})")
            proposals = list(r["proposed_token_ids"] or ())
            targ = list(r["target_argmax_ids"] or ())
            emitted = list(r["emitted_token_ids"] or ())
            if not proposals:
                prefix_ids += emitted
                continue
            ids = torch.tensor([prefix_ids + proposals], device=dev)
            out = draft(input_ids=ids, output_hidden_states=True, use_cache=False)
            hs, logits = out.hidden_states, out.logits
            positions = generating_positions(len(prefix_ids), len(proposals))
            for i, pos in enumerate(positions):
                pred = int(logits[0, pos].argmax())
                aligned = int(pred == proposals[i])
                n_aligned += aligned
                n_tok += 1
                label = int(i < len(targ) and proposals[i] == targ[i])
                for L in DEFAULT_LAYERS:
                    acts[L].append(hs[L][0, pos].to(torch.float16).cpu().numpy())
                meta.append({"request_id": rid, "round_id": r["round_id"],
                             "offset": i, "token_position": r["start_output_pos"] + i,
                             "label": label, "aligned": aligned,
                             "split": split_of.get(rid, "unknown"),
                             "domain": domain_of.get(rid, "unknown"),
                             "phase": annotate_phase(r["start_output_pos"] + i)})
            prefix_ids += emitted  # advance committed prefix for the next round
        if (j + 1) % 20 == 0:
            print(f"{j+1}/{len(reqs)} prompts, {n_tok} tokens, "
                  f"align={n_aligned/max(n_tok,1):.4f}", flush=True)

    outdir = f"/artifacts/probes/{run_id}"
    os.makedirs(outdir, exist_ok=True)
    for L in DEFAULT_LAYERS:
        np.save(f"{outdir}/acts_L{L}.npy", np.stack(acts[L]))
    pq.write_table(pa.Table.from_pylist(meta), f"{outdir}/metadata.parquet")
    with open(f"{outdir}/capture_meta.json", "w") as f:
        json.dump({"run_id": run_id, "split": split, "n_prompts": len(reqs),
                   "n_tokens": n_tok, "layers": list(DEFAULT_LAYERS),
                   "align_rate": n_aligned / max(n_tok, 1),
                   "d_model": int(acts[DEFAULT_LAYERS[0]][0].shape[0])}, f, indent=2)
    artifacts.commit()
    print(f"captured {n_tok} tokens x {len(DEFAULT_LAYERS)} layers; "
          f"align_rate={n_aligned/max(n_tok,1):.4f}; wrote {outdir}")
    return {"n_tokens": n_tok, "align_rate": n_aligned / max(n_tok, 1),
            "n_prompts": len(reqs)}


@app.local_entrypoint()
def capture(run_id: str = "sweep-2026-07-11T203836", cap_prompts: int = 120):
    capture_activations.remote(run_id, cap_prompts=cap_prompts)


@app.function(image=image, gpu=GPU, volumes=VOLUMES, timeout=6 * 3600)
def capture_frontier_activations(run_id: str = "sweep-2026-07-11T203836",
                                 cap_prompts: int = 120, split: str = "dev") -> dict:
    """I23/C10 (D023): teacher-forced TARGET residual-stream capture at the
    verified-context FRONTIER (last-committed) position that exists BEFORE each
    decode round drafts, labelled by that round's OWN realized acceptance. A probe
    on this "already cached" frontier rep answers: can we predict a round's
    acceptance before spending any draft compute?

    Mirrors capture_activations exactly for prefix reconstruction and volume IO,
    but (a) reads pair.target (not the draft), (b) reads the frontier position
    len(prefix)-1 (not the proposal-generating positions), and (c) labels from
    round r's accepted_prefix_len (not the sealed per-proposal target_match).

    Eager attention (D014/D015) keeps this the hookable path; the residual stream
    is read via output_hidden_states (numerically the real model).
    """
    import json
    import os

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch

    from cas.annotate.phases import annotate_phase
    from cas.autoresearch.types import (
        FRONTIER_META_COLS, FRONTIER_SUBDIR, frontier_acts_filename)
    from cas.capture import DEFAULT_LAYERS, frontier_position
    from cas.config import EngineConfig
    from cas.models import load_pair

    torch.set_grad_enabled(False)
    cfg = EngineConfig()
    pair = load_pair(cfg)
    target, tok, dev = pair.target, pair.tokenizer, pair.device

    base = f"/artifacts/traces/{run_id}"
    rounds = pq.read_table(f"{base}/fixed_8/rounds.parquet").to_pylist()
    summ = pq.read_table(f"{base}/fixed_8/request_summaries.parquet").to_pylist()
    with open("/artifacts/data/split_manifest.json") as f:
        assignment = json.load(f)["assignment"]          # prompt_hash -> split
    split_of = {s["request_id"]: assignment.get(s["prompt_hash"], "unknown")
                for s in summ}
    domain_of = {s["request_id"]: s["domain"] for s in summ}
    with open("/artifacts/data/prompts.jsonl") as f:
        prompt_text = {json.loads(l)["prompt_id"]: json.loads(l)["prompt_text"]
                       for l in (ln for ln in f)}

    if split not in ("dev", "test", "all"):
        raise ValueError(f"split must be 'dev', 'test', or 'all'; got {split!r}")

    by_req: dict = {}
    for r in rounds:
        by_req.setdefault(r["request_id"], []).append(r)
    # "all" captures dev AND test into one artifact (cap_prompts applied PER split
    # for balance); downstream fits still filter rows by their per-row `split`, so
    # dev-selection stays honest and test stays frozen until deliberately used.
    target_splits = ("dev", "test") if split == "all" else (split,)
    reqs = []
    for sp in target_splits:
        sp_reqs = sorted(rid for rid in by_req
                         if split_of.get(rid) == sp and rid in prompt_text)
        reqs += sp_reqs[:cap_prompts]

    acts = {L: [] for L in DEFAULT_LAYERS}
    meta = []
    n_rows = 0
    for j, rid in enumerate(reqs):
        prompt_ids = tok(prompt_text[rid])["input_ids"]
        # Same prefill-token reconstruction as capture_activations: the engine
        # commits one target greedy argmax token after the prompt BEFORE round 0,
        # so generated[0] is that token and round 0 has start_output_pos == 1.
        t_ids = torch.tensor([prompt_ids], device=dev)
        first_tok = int(pair.target(input_ids=t_ids, use_cache=False).logits[0, -1].argmax())
        prefix_ids = list(prompt_ids) + [first_tok]  # grows by each round's emissions
        for r in sorted(by_req[rid], key=lambda x: x["round_id"]):
            if r["start_output_pos"] != len(prefix_ids) - len(prompt_ids):
                raise ValueError(f"{rid} r{r['round_id']} emit/pos mismatch "
                                 f"({r['start_output_pos']} vs {len(prefix_ids) - len(prompt_ids)})")
            proposals = list(r["proposed_token_ids"] or ())
            emitted = list(r["emitted_token_ids"] or ())
            if not proposals:
                prefix_ids += emitted        # skip: no draft this round to predict
                continue
            # BEFORE appending this round's emissions, prefix_ids IS the verified
            # context that exists before round r drafts. Teacher-force the TARGET
            # over exactly this prefix and read the frontier (last-committed) row.
            # A single forward over the FULL committed sequence would yield the
            # identical vector at this position -- causal attention makes hidden
            # state[p] a function of tokens 0..p only -- but we forward per round
            # to mirror capture_activations and keep the alignment self-evident.
            fpos = frontier_position(len(prefix_ids))     # == len(prefix_ids) - 1
            ids = torch.tensor([prefix_ids], device=dev)
            hs = target(input_ids=ids, output_hidden_states=True,
                        use_cache=False).hidden_states
            # Label from round r itself (its own realized acceptance).
            accept = int(r["accepted_prefix_len"] >= 1)
            accepted_len = int(r["accepted_prefix_len"])
            for L in DEFAULT_LAYERS:
                acts[L].append(hs[L][0, fpos].to(torch.float16).cpu().numpy())
            # Dict keys in FRONTIER_META_COLS order (obeyed by from_pylist below).
            meta.append({"request_id": rid, "round_id": r["round_id"],
                         "split": split_of.get(rid, "unknown"),
                         "domain": domain_of.get(rid, "unknown"),
                         "phase": annotate_phase(r["start_output_pos"]),
                         "accept": accept, "accepted_len": accepted_len})
            n_rows += 1
            prefix_ids += emitted  # advance committed prefix for the next round
        if (j + 1) % 20 == 0:
            print(f"{j+1}/{len(reqs)} prompts, {n_rows} frontier rows", flush=True)

    if n_rows == 0:
        raise ValueError(
            f"no frontier rows captured for run={run_id} split={split}; check the "
            "run has fixed_8 traces and the split manifest matches the requested split")

    outdir = f"/artifacts/probes/{run_id}/{FRONTIER_SUBDIR}"
    os.makedirs(outdir, exist_ok=True)
    for L in DEFAULT_LAYERS:
        np.save(f"{outdir}/{frontier_acts_filename(L)}", np.stack(acts[L]))
    meta_table = pa.Table.from_pylist(meta).select(list(FRONTIER_META_COLS))
    pq.write_table(meta_table, f"{outdir}/frontier_metadata.parquet")
    with open(f"{outdir}/frontier_capture_meta.json", "w") as f:
        json.dump({"run_id": run_id, "split": split, "n_prompts": len(reqs),
                   "n_rows": n_rows, "layers": list(DEFAULT_LAYERS),
                   "d_model": int(acts[DEFAULT_LAYERS[0]][0].shape[0])}, f, indent=2)
    artifacts.commit()
    print(f"captured {n_rows} frontier rows x {len(DEFAULT_LAYERS)} layers; "
          f"wrote {outdir}")
    return {"n_rows": n_rows, "n_prompts": len(reqs),
            "layers": list(DEFAULT_LAYERS)}


@app.local_entrypoint()
def capture_frontier(run_id: str = "sweep-2026-07-11T203836",
                     cap_prompts: int = 120, split: str = "dev"):
    """Capture the target-frontier representation for a sealed run (D023).

    --split: 'dev' (default), 'test', or 'all' (dev+test in one artifact,
             cap_prompts applied per split). Capturing 'all' once avoids the
             re-capture-overwrites-dev footgun before the eventual test pass.

    LAPTOP-SAFE: launch with `modal run --detach` so the job keeps running on
    Modal's infrastructure after you close the laptop / terminal --
        modal run --detach modal_app.py::capture_frontier --split all
    then watch it later with `modal app list` and `modal app logs <app-id>`.
    """
    capture_frontier_activations.remote(run_id, cap_prompts=cap_prompts, split=split)


@app.function(image=image, volumes=VOLUMES, timeout=2 * 3600)  # CPU-only
def fit_probes(run_id: str = "sweep-2026-07-11T203836",
               layers: str = "6,12,18,24", eval_split: str = "dev") -> dict:
    """I12/C01: layerwise hidden probes + hidden⊕surface incremental-info test
    vs the ~0.84 surface baseline, on the captured activations."""
    from scripts.fit_probes import run as run_probes

    probe_dir = f"/artifacts/probes/{run_id}"
    run_dir = f"/artifacts/traces/{run_id}"
    ls = [int(x) for x in layers.split(",")]
    res = run_probes(probe_dir, run_dir, ls, eval_split)

    s = res["surface_only"]["auroc"]
    print(f"rows={res['n_rows']} pos_rate={res['pos_rate']:.3f} split={eval_split}")
    print(f"SURFACE baseline AUROC = {s:.4f}  (bar to beat)")
    print(f"{'layer':>6} {'hidden':>8} {'hid+surf':>9} {'Δ vs surf':>11}")
    for L, d in res["layers"].items():
        h, c = d["hidden_only"]["auroc"], d["hidden_plus_surface"]["auroc"]
        print(f"{L:>6} {h:>8.4f} {c:>9.4f} {c - s:>+11.4f}")
    import json as _json
    with open(f"{probe_dir}/probe_results.json", "w") as f:
        _json.dump(res, f, indent=2)
    artifacts.commit()
    best = max(res["layers"].items(), key=lambda kv: kv[1]["hidden_only"]["auroc"])
    verdict = best[1]["hidden_only"]["auroc"] > s
    print(f"best hidden layer {best[0]} @ {best[1]['hidden_only']['auroc']:.4f} -> "
          f"{'BEATS' if verdict else 'does NOT beat'} surface {s:.4f}")
    return {"surface": s, "best_hidden_layer": best[0],
            "best_hidden_auroc": best[1]["hidden_only"]["auroc"], "beats": verdict}


@app.local_entrypoint()
def probe(run_id: str = "sweep-2026-07-11T203836", layers: str = "6,12,18,24"):
    fit_probes.remote(run_id, layers=layers)


@app.function(image=image, volumes=VOLUMES, timeout=2 * 3600)  # CPU-only
def fit_autoresearch(run_id: str = "sweep-2026-07-11T203836",
                     eval_split: str = "dev", layers: str = "6,12,18,24",
                     spec_json: str = "", seed: int = 0) -> dict:
    """I13/I23 (D023): score PRE-ROUND candidate signals from the TARGET-frontier
    representation vs the frozen `preround_hardened` baseline (~0.73 AUROC), with
    equal-capacity norm-matched + random controls under prompt-grouped OOF.

    Reads the frontier artifact (capture_frontier) at
    /artifacts/probes/<run>/frontier/ and the sealed fixed_8 traces. Runs the
    default seed library (raw/lowrank/drift/norm/align), or one FeatureSpec via
    --spec-json. CPU-only; dev-only by default (test stays frozen). Every number is
    script-computed from immutable artifacts (AGENTS.md); results are CANDIDATES,
    not claims — G1/G2 gates are tripped by hand and "circuit" language stays
    G2-gated (D020)."""
    import json as _json

    from cas.autoresearch.features import default_seed_specs
    from cas.autoresearch.types import FeatureSpec
    from scripts.fit_autoresearch import (_baseline_by_round, _load_frontier,
                                          score_spec)

    layer_t = tuple(int(x) for x in layers.split(","))
    acts, meta = _load_frontier(f"/artifacts/probes/{run_id}", layer_t)
    base_by_key = _baseline_by_round(f"/artifacts/traces/{run_id}")

    if spec_json:
        d = _json.loads(spec_json)
        specs = [FeatureSpec(d["name"], d["family"], tuple(d["layers"]),
                             d.get("params", {}))]
    else:
        specs = default_seed_specs(layer_t)

    results = [score_spec(s, acts, meta, base_by_key, eval_split, seed=seed)
               for s in specs]
    os.makedirs(f"/artifacts/analysis/{run_id}", exist_ok=True)
    outp = f"/artifacts/analysis/{run_id}/autoresearch_{eval_split}.json"
    with open(outp, "w") as f:
        _json.dump({"run": run_id, "eval": eval_split, "results": results}, f,
                   indent=2)
    artifacts.commit()

    ranked = sorted([r for r in results if r.get("deltas")],
                    key=lambda r: (r["deltas"].get("auroc") or -1), reverse=True)
    print(f"run={run_id} split={eval_split}  frozen bar = preround_hardened "
          f"(base AUROC ~0.70 dev)")
    if ranked:
        _b = ranked[0].get("base_calibrated") or {}
        if _b.get("ece") is not None:
            print(f"  base (recalibrated): ece={_b['ece']:.4f} "
                  f"regret={_b['regret']:.4f}")
    print(f"{'candidate':>22} {'d_auroc':>8} {'win':>4} {'cal_ece':>8} "
          f"{'cal_reg':>8} {'d_reg_cal':>10} {'helps_dec':>9}")
    for r in ranked:
        d = r["deltas"].get("auroc")
        cc = r.get("combined_calibrated") or {}
        dc = r.get("deltas_calibrated") or {}
        ds = f"{d:+.4f}" if d is not None else "n/a"
        win = "Y" if (r["beats_baseline"] and r["beats_controls"]) else "-"
        ece, reg, dreg = cc.get("ece"), cc.get("regret"), dc.get("regret")
        es = f"{ece:.4f}" if ece is not None else "n/a"
        rs = f"{reg:.4f}" if reg is not None else "n/a"
        drs = f"{dreg:+.4f}" if dreg is not None else "n/a"
        print(f"{r['spec']['name']:>22} {ds:>8} {win:>4} {es:>8} {rs:>8} "
              f"{drs:>10} {str(r.get('helps_decision_calibrated')):>9}")
    winners = [r["spec"]["name"] for r in ranked
               if r["beats_baseline"] and r["beats_controls"]]
    print("beats baseline+controls: "
          + (", ".join(winners) if winners
             else "NONE (seed library did not clear the pre-round bar)"))
    return {"run": run_id, "eval": eval_split, "n_specs": len(results),
            "winners": winners,
            "leaderboard": [{"name": r["spec"]["name"],
                             "delta_auroc": r["deltas"].get("auroc"),
                             "beats_baseline": r["beats_baseline"],
                             "beats_controls": r["beats_controls"]} for r in ranked]}


@app.local_entrypoint()
def autoresearch(run_id: str = "sweep-2026-07-11T203836", eval_split: str = "dev",
                 layers: str = "6,12,18,24", spec_json: str = "", seed: int = 0):
    """Score pre-round candidate signals vs the frozen 0.73 bar (D023).

    eval_split stays 'dev' until the deliberate one-shot 'test' pass. LAPTOP-SAFE:
    prefix with `modal run --detach` to survive closing the laptop --
        modal run --detach modal_app.py::autoresearch --eval-split dev
    Read results afterwards with `autoresearch_show` (a detached run's return value
    never reaches your local shell).
    """
    fit_autoresearch.remote(run_id, eval_split, layers, spec_json, seed)


@app.function(image=image, volumes=VOLUMES, timeout=300)  # CPU-only, read-only
def show_autoresearch(run_id: str = "sweep-2026-07-11T203836",
                      eval_split: str = "dev") -> dict:
    """Print the saved autoresearch leaderboard from the volume (D023).

    Convenience reader for DETACHED `autoresearch` runs whose return value never
    reached the local shell: loads
    /artifacts/analysis/<run>/autoresearch_<split>.json and re-prints the ranked
    candidate table. Read-only (no fitting, no commit). Results are CANDIDATES, not
    claims; "circuit" language stays G2-gated (D020)."""
    import json as _json

    path = f"/artifacts/analysis/{run_id}/autoresearch_{eval_split}.json"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found -- run "
            f"`autoresearch --run-id {run_id} --eval-split {eval_split}` first")
    with open(path) as f:
        results = _json.load(f).get("results", [])

    ranked = sorted([r for r in results if r.get("deltas")],
                    key=lambda r: (r["deltas"].get("auroc") or -1), reverse=True)
    print(f"run={run_id} split={eval_split}  frozen bar = preround_hardened "
          f"(base AUROC ~0.70 dev)")
    if ranked:
        _b = ranked[0].get("base_calibrated") or {}
        if _b.get("ece") is not None:
            print(f"  base (recalibrated): ece={_b['ece']:.4f} "
                  f"regret={_b['regret']:.4f}")
    print(f"{'candidate':>22} {'d_auroc':>8} {'win':>4} {'cal_ece':>8} "
          f"{'cal_reg':>8} {'d_reg_cal':>10} {'helps_dec':>9}")
    for r in ranked:
        d = r["deltas"].get("auroc")
        cc = r.get("combined_calibrated") or {}
        dc = r.get("deltas_calibrated") or {}
        ds = f"{d:+.4f}" if d is not None else "n/a"
        win = "Y" if (r["beats_baseline"] and r["beats_controls"]) else "-"
        ece, reg, dreg = cc.get("ece"), cc.get("regret"), dc.get("regret")
        es = f"{ece:.4f}" if ece is not None else "n/a"
        rs = f"{reg:.4f}" if reg is not None else "n/a"
        drs = f"{dreg:+.4f}" if dreg is not None else "n/a"
        print(f"{r['spec']['name']:>22} {ds:>8} {win:>4} {es:>8} {rs:>8} "
              f"{drs:>10} {str(r.get('helps_decision_calibrated')):>9}")
    for r in [r for r in results if not r.get("deltas")]:
        nm = r.get("spec")
        nm = nm if isinstance(nm, str) else (nm or {}).get("name", "?")
        print(f"{nm:>24}  (skipped: {r.get('note', 'no result')})")
    winners = [r["spec"]["name"] for r in ranked
               if r["beats_baseline"] and r["beats_controls"]]
    print("beats baseline+controls: "
          + (", ".join(winners) if winners
             else "NONE (seed library did not clear the pre-round bar)"))
    return {"run": run_id, "eval": eval_split, "winners": winners,
            "n_results": len(results)}


@app.local_entrypoint()
def autoresearch_show(run_id: str = "sweep-2026-07-11T203836",
                      eval_split: str = "dev"):
    """Print the saved autoresearch leaderboard (read-only; for detached runs)."""
    show_autoresearch.remote(run_id, eval_split)
