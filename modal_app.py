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


@app.function(image=image, gpu=GPU, volumes=VOLUMES, timeout=3600)
def bench_draft(run_id: str = "sweep-2026-07-11T203836", context_len: int = 128,
                n_warmup: int = 5, n_iter: int = 25, gap: int = 5) -> dict:
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

    torch.set_grad_enabled(False)
    ACTIONS = [1, 2, 3, 4, 6, 8]
    cfg = EngineConfig()
    pair = load_pair(cfg)
    dev = pair.device

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

    out = {"gap_ms": gap_ms, "verify0_ms": verify0_ms, "verify_ms": verify_ms,
           "draft_ms": draft, "per_tok_ms": per_tok, "headroom": headroom,
           "context_len": ctx.shape[1], "n_iter": n_iter}
    path = f"/artifacts/analysis/{run_id}/t3_4_bench.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    artifacts.commit()
    print(f"\nwrote {path}")
    return headroom


@app.local_entrypoint()
def bench(run_id: str = "sweep-2026-07-11T203836"):
    bench_draft.remote(run_id)
