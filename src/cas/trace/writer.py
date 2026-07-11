"""Durable trace writer (issue I06): validated, columnar, write-once.

Layout per run (TRACE_SCHEMA.md artifact layers 1-4; activations, layer 5, are
separate artifacts referenced by id and never written here):

    <dir>/run_metadata.json          one record (layer 1)
    <dir>/request_summaries.parquet  one row per prompt x policy (layer 2)
    <dir>/rounds.parquet             one row per draft/verify round (layer 3)
    <dir>/tokens.parquet             one row per proposed token (layer 4)
    <dir>/MANIFEST.json              sha256 checksums + row counts

Rules enforced here, not by convention:
  * every invariant in cas.trace.validate passes before any byte is written;
  * a directory with a MANIFEST is immutable — the writer refuses to touch it
    (AGENTS.md: raw artifacts are immutable);
  * tuple fields are serialized as Parquet lists; open-ended dicts
    (latency_ns, predicted_payoffs) as JSON strings so the schema stays stable
    when component names change.

pyarrow is pinned in both the Modal image and requirements.txt; there is no
CSV/JSONL fallback for large arrays by design (schema: "Never place large
serialized arrays directly in CSV or JSONL fields").
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os

from .records import (
    SCHEMA_VERSION,
    RequestSummary,
    RoundTrace,
    RunMetadata,
    TokenTrace,
    derive_token_traces,
)
from .validate import TraceValidationError, validate_run

_JSON_FIELDS = {"latency_ns", "predicted_payoffs", "driver_cuda_framework"}


def _row(record) -> dict:
    """Dataclass -> flat dict; dict-valued fields become JSON strings."""
    out = {}
    for f in dataclasses.fields(record):
        v = getattr(record, f.name)
        if f.name in _JSON_FIELDS:
            v = json.dumps(v, sort_keys=True) if v is not None else None
        elif isinstance(v, tuple):
            v = list(v)
        out[f.name] = v
    return out


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class TraceWriter:
    """Accumulates one run's records in memory, validates, writes once.

    Usage:
        w = TraceWriter(out_dir, run_meta)
        w.add_request(summary, rounds)          # repeatedly
        manifest = w.finalize()                 # validate + write + seal
    """

    def __init__(self, out_dir: str, meta: RunMetadata):
        if meta.schema_version != SCHEMA_VERSION:
            raise TraceValidationError(
                f"run metadata schema_version {meta.schema_version!r} != "
                f"writer schema {SCHEMA_VERSION!r} (bump + migration note "
                f"required, TRACE_SCHEMA invariant 7)")
        manifest = os.path.join(out_dir, "MANIFEST.json")
        if os.path.exists(manifest):
            raise TraceValidationError(
                f"{out_dir} is a sealed trace run (MANIFEST present); raw "
                f"artifacts are immutable — write to a new directory")
        self.out_dir = out_dir
        self.meta = meta
        self._summaries: list[RequestSummary] = []
        self._rounds: dict[str, list[RoundTrace]] = {}
        self._tokens: list[TokenTrace] = []
        self._sealed = False

    def add_request(
        self,
        summary: RequestSummary,
        rounds: list[RoundTrace],
        tokens: list[TokenTrace] | None = None,
    ) -> None:
        """Queue one request. Tokens are derived from rounds when omitted."""
        if self._sealed:
            raise TraceValidationError("writer already finalized")
        self._summaries.append(summary)
        self._rounds[summary.request_id] = list(rounds)
        if tokens is None:
            tokens = [
                t for rt in rounds
                for t in derive_token_traces(rt, summary.run_id)
            ]
        self._tokens.extend(tokens)

    def finalize(self) -> dict:
        """Validate every invariant, write all layers, seal with MANIFEST."""
        if self._sealed:
            raise TraceValidationError("writer already finalized")
        validate_run(self.meta, self._summaries, self._rounds)

        import pyarrow as pa
        import pyarrow.parquet as pq

        os.makedirs(self.out_dir, exist_ok=True)
        files: dict[str, dict] = {}

        meta_path = os.path.join(self.out_dir, "run_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(_row(self.meta), f, indent=2, sort_keys=True)
        files["run_metadata.json"] = {"rows": 1}

        layers = [
            ("request_summaries.parquet", self._summaries),
            ("rounds.parquet",
             [rt for req in self._summaries
              for rt in self._rounds.get(req.request_id, [])]),
            ("tokens.parquet", self._tokens),
        ]
        for name, records in layers:
            path = os.path.join(self.out_dir, name)
            rows = [_row(r) for r in records]
            if rows:
                table = pa.Table.from_pylist(rows)
            else:  # empty layer still gets a typed file (target-only: tokens)
                table = pa.Table.from_pylist(
                    [_row(records_type_default(name))]).slice(0, 0)
            pq.write_table(table, path)
            files[name] = {"rows": len(rows)}

        for name in files:
            files[name]["sha256"] = _sha256(os.path.join(self.out_dir, name))

        manifest = {
            "schema_version": self.meta.schema_version,
            "run_id": self.meta.run_id,
            "files": files,
            "n_requests": len(self._summaries),
        }
        with open(os.path.join(self.out_dir, "MANIFEST.json"), "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        self._sealed = True
        return manifest


def records_type_default(layer_file: str):
    """A default-constructed record used only to type empty Parquet layers."""
    if layer_file.startswith("request_summaries"):
        return RequestSummary(
            run_id="", request_id="", dataset="", domain="", split="",
            prompt_hash="", policy_name="", prompt_tokens=0, output_tokens=0,
            termination_reason="", output_token_hash="", total_drafted=0,
            total_accepted=0, total_rejected=0, n_rounds=0)
    if layer_file.startswith("rounds"):
        return RoundTrace(
            request_id="", round_id=0, start_output_pos=0, requested_action=0,
            realized_draft_len=0, proposed_token_ids=(),
            accepted_prefix_len=0, first_rejection_pos=None,
            emitted_token_ids=(0,))
    return TokenTrace(
        run_id="", request_id="", round_id=0, token_position=0,
        proposal_offset=0, draft_token_id=0, target_token_id=0, accepted=False)
