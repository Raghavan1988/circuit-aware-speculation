"""Dataset ingestion into normalized prompt records (issue I05).

Loads the four workload domains into a common record shape, recording dataset
version, license, and a stable prompt hash. Requires the `datasets` library and
network; run on Modal. No synthetic fallback: per AGENTS.md, ingestion fails
loudly rather than silently substituting toy data.

Domains (docs/EXPERIMENT_CONTRACT.md "Data"; chat supplement + summarization
held-out per D013):
    code  -> HumanEval
    math  -> GSM8K
    chat  -> MT-Bench first turns, supplemented to ~150 (documented source)
    summ  -> summarization prompts (held-out; never used for any fitting)
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class IngestedPrompt:
    prompt_id: str
    dataset: str
    domain: str
    prompt_text: str
    prompt_hash: str
    dataset_revision: str
    license: str


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _rec(prompt_id, dataset, domain, text, revision, license_) -> IngestedPrompt:
    return IngestedPrompt(
        prompt_id=prompt_id,
        dataset=dataset,
        domain=domain,
        prompt_text=text,
        prompt_hash=_hash(text),
        dataset_revision=revision,
        license=license_,
    )


def load_humaneval(limit: int | None = None) -> list[IngestedPrompt]:
    from datasets import load_dataset

    ds = load_dataset("openai_humaneval", split="test")
    rev = str(getattr(ds, "_fingerprint", "unknown"))
    out = []
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        out.append(_rec(f"humaneval-{row['task_id']}", "humaneval", "code",
                        row["prompt"], rev, "MIT"))
    return out


def load_gsm8k(limit: int | None = 200) -> list[IngestedPrompt]:
    from datasets import load_dataset

    ds = load_dataset("gsm8k", "main", split="test")
    rev = str(getattr(ds, "_fingerprint", "unknown"))
    out = []
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        out.append(_rec(f"gsm8k-{i}", "gsm8k", "math", row["question"], rev, "MIT"))
    return out


def load_mtbench(limit: int | None = None) -> list[IngestedPrompt]:
    """MT-Bench first turns (~80). Supplement separately to reach the target."""
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
    rev = str(getattr(ds, "_fingerprint", "unknown"))
    out = []
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        turns = row.get("prompt") or row.get("turns")
        text = turns[0] if isinstance(turns, list) else str(turns)
        out.append(_rec(f"mtbench-{i}", "mt_bench", "chat", text, rev,
                        "Apache-2.0/CC-BY (verify per source)"))
    return out


def load_summarization(limit: int | None = 200) -> list[IngestedPrompt]:
    """Held-out domain (D013). XSum article -> 'Summarize:' prompt."""
    from datasets import load_dataset

    ds = load_dataset("EdinburghNLP/xsum", split="test")
    rev = str(getattr(ds, "_fingerprint", "unknown"))
    out = []
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        text = f"Summarize the following article:\n\n{row['document']}\n\nSummary:"
        out.append(_rec(f"xsum-{row.get('id', i)}", "xsum", "summ", text, rev,
                        "CC-BY-SA-4.0 (verify per source)"))
    return out


def ingest_all(caps: dict | None = None) -> list[dict]:
    """Load every domain and return plain dicts (JSON-serializable).

    `caps` optionally limits per loader, e.g. {"gsm8k": 200, "summ": 200}.
    Chat supplementation beyond MT-Bench is recorded as a documented decision
    (D013) and added by the caller once a source is chosen; this function loads
    MT-Bench only for chat and flags the shortfall.
    """
    caps = caps or {}
    records: list[IngestedPrompt] = []
    records += load_humaneval(caps.get("code"))
    records += load_gsm8k(caps.get("math", 200))
    records += load_mtbench(caps.get("chat"))
    records += load_summarization(caps.get("summ", 200))
    return [asdict(r) for r in records]
