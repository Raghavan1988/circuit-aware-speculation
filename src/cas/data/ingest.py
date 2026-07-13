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
    # v2 provenance (CORPUS_PLAN / D022); optional so v1 loaders are unchanged.
    spdx: str = ""
    row_id: str = ""


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _rec(prompt_id, dataset, domain, text, revision, license_,
         spdx: str = "", row_id: str = "") -> IngestedPrompt:
    return IngestedPrompt(
        prompt_id=prompt_id,
        dataset=dataset,
        domain=domain,
        prompt_text=text,
        prompt_hash=_hash(text),
        dataset_revision=revision,
        license=license_,
        spdx=spdx,
        row_id=str(row_id),
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


# ---------------------------------------------------------------------------
# v2 exhaustive corpus (CORPUS_PLAN.local.md; scope change authorized by D022).
# Core tier: ungated + redistributable, 8 axes. Copyright-text datasets store a
# row_id so the shipped artifact can carry ids only (TRACE_SCHEMA supports it);
# raw text stays in the local trace-generation cache. Every reference completion
# is generated in-house with the Qwen pair, sidestepping output-license issues.
# ---------------------------------------------------------------------------

# Per-dataset license metadata for the corpus NOTICE/LICENSES file.
CORPUS_LICENSES = {
    "humaneval": {"spdx": "MIT", "redistributable": "yes", "axis": "code"},
    "mbpp": {"spdx": "CC-BY-4.0", "redistributable": "yes", "axis": "code"},
    "gsm8k": {"spdx": "MIT", "redistributable": "yes", "axis": "math"},
    "mt_bench": {"spdx": "Apache-2.0", "redistributable": "yes", "axis": "chat"},
    "oasst1": {"spdx": "Apache-2.0", "redistributable": "yes", "axis": "chat"},
    "wmt14_de_en": {"spdx": "see-WMT-shared-task", "redistributable": "cite-source", "axis": "translation"},
    "nq_open": {"spdx": "CC-BY-SA-3.0", "redistributable": "copyleft-keep-license", "axis": "qa_rag"},
    "jsonschemabench": {"spdx": "MIT", "redistributable": "yes", "axis": "structured"},
    "cnn_dailymail": {"spdx": "Apache-2.0(code); article text publisher-copyright",
                      "redistributable": "ship-row-ids-only", "axis": "summarization"},
    "xsum": {"spdx": "unknown; BBC-copyright", "redistributable": "ship-row-ids-only", "axis": "summarization"},
}


def _fp(ds) -> str:
    return str(getattr(ds, "_fingerprint", "streaming"))


def _stream(dataset_id, config, split, n):
    """Stream the first n rows (avoids downloading full large datasets)."""
    import itertools

    from datasets import load_dataset

    kw = {"split": split, "streaming": True}
    ds = load_dataset(dataset_id, config, **kw) if config else load_dataset(dataset_id, **kw)
    return list(itertools.islice(ds, n))


def load_mbpp(limit: int | None = 200) -> list[IngestedPrompt]:
    """MBPP sanitized: short NL -> function synthesis (higher-alpha code)."""
    from datasets import load_dataset

    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    rev = _fp(ds)
    out = []
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        # sanitized config exposes 'prompt'; full config exposes 'text'
        desc = row.get("text") or row.get("prompt") or ""
        tests = row.get("test_list") or [""]
        text = f"{desc}\n\nWrite a Python function. Your code should satisfy: {tests[0]}"
        out.append(_rec(f"mbpp-{row['task_id']}", "mbpp", "code", text, rev,
                        "CC-BY-4.0", spdx="CC-BY-4.0", row_id=row["task_id"]))
    return out


def load_oasst(limit: int | None = 150) -> list[IngestedPrompt]:
    """OASST1 human-authored chat: root English prompter messages, one per tree."""
    rows = _stream("OpenAssistant/oasst1", None, "train", 40000)
    out, seen = [], set()
    for row in rows:
        if limit and len(out) >= limit:
            break
        if (row.get("parent_id") is None and row.get("role") == "prompter"
                and row.get("lang") == "en"):
            tid = row.get("message_tree_id")
            if tid in seen:
                continue
            seen.add(tid)
            out.append(_rec(f"oasst-{tid}", "oasst1", "chat", row["text"],
                            "streaming", "Apache-2.0", spdx="Apache-2.0", row_id=tid))
    return out


def load_wmt14(limit: int | None = 150) -> list[IngestedPrompt]:
    """WMT14 de-en (Spec-Bench translation slot); mid-alpha register."""
    rows = _stream("wmt/wmt14", "de-en", "test", limit or 150)
    out = []
    for i, row in enumerate(rows):
        de = row["translation"]["de"]
        text = f"Translate the following German text to English:\n\n{de}\n\nEnglish:"
        out.append(_rec(f"wmt14-{i}", "wmt14_de_en", "translation", text,
                        "streaming", "WMT-shared-task", spdx="see-source", row_id=i))
    return out


def load_nq_open(limit: int | None = 200) -> list[IngestedPrompt]:
    """Natural Questions Open (Spec-Bench QA/RAG slot); entity-heavy low-alpha."""
    from datasets import load_dataset

    ds = load_dataset("google-research-datasets/nq_open", split="validation")
    rev = _fp(ds)
    out = []
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        text = f"Answer the question concisely.\n\nQuestion: {row['question']}\nAnswer:"
        out.append(_rec(f"nqopen-{i}", "nq_open", "qa_rag", text, rev,
                        "CC-BY-SA-3.0", spdx="CC-BY-SA-3.0", row_id=i))
    return out


def load_jsonschemabench(limit: int | None = 150) -> list[IngestedPrompt]:
    """Schema-constrained JSON generation: the high-alpha structural extreme."""
    from datasets import load_dataset

    last_err = None
    for cfg in ("default", "all", "Github_easy", "Github_medium"):
        for split in ("test", "train", "val"):
            try:
                ds = _stream("epfl-dlab/JSONSchemaBench", cfg, split, limit or 150)
                if not ds:
                    continue
                schema_key = next((k for k in ("json_schema", "schema", "content")
                                   if k in ds[0]), None)
                if schema_key is None:
                    continue
                rev = f"{cfg}/{split}"
                out = []
                for i, row in enumerate(ds):
                    schema = row[schema_key]
                    text = ("Generate a JSON object that strictly conforms to this "
                            f"JSON Schema:\n\n{schema}\n\nJSON:")
                    out.append(_rec(f"jsonschema-{cfg}-{i}", "jsonschemabench",
                                    "structured", text, rev, "MIT",
                                    spdx="MIT", row_id=row.get("unique_id", i)))
                return out
            except Exception as e:  # try the next config/split
                last_err = e
    raise RuntimeError(f"JSONSchemaBench: no config/split loaded ({last_err!r})")


def load_cnndm(limit: int | None = 200) -> list[IngestedPrompt]:
    """CNN/DailyMail (Spec-Bench summarization slot); high prompt-output overlap.
    Article text is publisher-copyright -> ship row_id only in any release."""
    rows = _stream("abisee/cnn_dailymail", "3.0.0", "test", limit or 200)
    out = []
    for i, row in enumerate(rows):
        text = f"Summarize the following news article:\n\n{row['article']}\n\nSummary:"
        out.append(_rec(f"cnndm-{row.get('id', i)}", "cnn_dailymail", "summ",
                        text, "streaming", "publisher-copyright",
                        spdx="ship-row-ids-only", row_id=row.get("id", i)))
    return out


def ingest_core_v2(caps: dict | None = None) -> tuple[list[dict], dict]:
    """Assemble the Core v2 corpus. Each loader is isolated: a failure records a
    status entry and the build continues, so one broken source cannot kill the
    corpus. Returns (records as dicts, per-loader status)."""
    caps = caps or {}
    loaders = [
        ("humaneval", lambda: load_humaneval(caps.get("code", 164))),
        ("mbpp", lambda: load_mbpp(caps.get("mbpp", 200))),
        ("gsm8k", lambda: load_gsm8k(caps.get("math", 200))),
        ("mtbench", lambda: load_mtbench(caps.get("chat", 80))),
        ("oasst", lambda: load_oasst(caps.get("oasst", 150))),
        ("wmt14", lambda: load_wmt14(caps.get("translation", 150))),
        ("nq_open", lambda: load_nq_open(caps.get("qa_rag", 200))),
        ("jsonschemabench", lambda: load_jsonschemabench(caps.get("structured", 150))),
        ("cnndm", lambda: load_cnndm(caps.get("summ", 200))),
    ]
    records: list[IngestedPrompt] = []
    status: dict = {}
    for name, fn in loaders:
        try:
            recs = fn()
            records += recs
            status[name] = {"ok": True, "n": len(recs)}
        except Exception as e:  # isolate per-loader failures
            status[name] = {"ok": False, "n": 0, "error": repr(e)[:300]}
    return [asdict(r) for r in records], status
