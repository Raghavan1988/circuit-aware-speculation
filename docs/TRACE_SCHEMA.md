# Trace Schema

The trace format must be versioned before the first full experiment. Use a columnar format such as Parquet for tabular traces and a separate chunked tensor format for large activations.

## Artifact layers

1. **Run metadata:** one record per experiment invocation.
2. **Request summary:** one record per prompt and policy.
3. **Decode-round trace:** one record per draft/verify round.
4. **Token trace:** one record per proposed or target-generated token.
5. **Activation artifact:** tensors stored separately and referenced by identifier.

## Run metadata

| Field | Type | Requirement |
|---|---|---|
| `schema_version` | string | Required |
| `run_id` | string | Globally unique and immutable |
| `created_at_utc` | timestamp | Required |
| `git_commit` | string | Required |
| `config_hash` | string | Required |
| `command` | string | Required, with secrets removed |
| `seed` | integer | Required |
| `target_model_id/revision` | string | Required |
| `draft_model_id/revision` | string | Required unless target-only |
| `tokenizer_id/revision` | string | Required |
| `dtype` | string | Required |
| `quantization` | string/null | Required |
| `device_name/count` | string/integer | Required |
| `driver_cuda_framework_versions` | struct | Required |
| `policy_name/version` | string | Required |
| `split_manifest_id` | string | Required |

## Request summary

Required fields:

- `run_id`, `request_id`, `dataset`, `domain`, `split`, and `prompt_hash`;
- prompt and output token counts;
- termination reason and output-token hash;
- prefill, decode, and end-to-end latency;
- TPOT and throughput;
- total drafted, accepted, and rejected tokens;
- peak device memory;
- equivalence status and reference-output hash;
- failure or exclusion reason.

Raw prompt text should be optional and governed by dataset licensing. Stable dataset row identifiers are preferred.

## Decode-round trace

Required fields:

- `run_id`, `request_id`, `round_id`, and starting output position;
- requested action and realized draft length;
- proposed token identifiers;
- accepted prefix length and first rejection position;
- recent-acceptance state;
- signal values used by the controller;
- predicted payoff for every candidate action;
- drafting, verification, controller, tracing, and synchronized total latency;
- cache length before and after verification;
- intervention identifier when active.

Optional fields (required when the corresponding feature is active):

- pre-round prediction values and their feature source (hook point / cached layer), when a pre-round predictor is active (I23);
- draft early-exit layer, when early exit is active.

## Token trace

Required fields:

- `run_id`, `request_id`, `round_id`, `token_position`, and proposal offset;
- proposed draft token and verified target token identifiers;
- acceptance boolean;
- draft log probability, entropy, top-1/top-2 margin;
- post-verification draft–target divergence;
- overlapping token-category labels and generation-phase label;
- activation artifact identifier and tensor row offset, if captured.

Do not store full vocabulary logits by default. If needed for an analysis, save them as a separately declared artifact with retention and size limits.

## Activation metadata

Each activation artifact must declare:

- model and hook-point names;
- layer indices;
- tensor shapes and dtype;
- pooling or summary operation;
- token-to-row mapping;
- intervention name, direction source, strength, and control type;
- checksum and storage URI/path.

Never place large serialized arrays directly in CSV or JSONL fields.

## Validation invariants

- All foreign identifiers resolve to exactly one parent record.
- Accepted prefix length does not exceed realized draft length.
- Per-request token totals equal sums over rounds.
- Latency components are non-negative; synchronized total latency bounds the recorded GPU components within documented tolerance.
- Test records never contribute to fitted probe, calibration, threshold, or intervention directions.
- Target-only records contain no draft activations or fabricated draft statistics.
- Schema changes increment `schema_version` and include a migration note.
