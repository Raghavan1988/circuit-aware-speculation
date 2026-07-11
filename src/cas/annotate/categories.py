"""Overlapping token-category labels (issue I11).

Categories may co-fire; do not force mutual exclusion. Labels follow the
acceptance-atlas list in docs/RESEARCH_SPEC.md. Heuristic, tokenizer-light:
operates on decoded piece strings (and optional token_id for specials).

Bump CATEGORY_SET_VERSION on any label-set or rule change (TRACE_SCHEMA inv. 7).
"""
from __future__ import annotations

import re
import string
from typing import Iterable

CATEGORY_SET_VERSION = "v1.0.0"

KNOWN_CATEGORIES: frozenset[str] = frozenset(
    {
        "whitespace",
        "punctuation",
        "code_delimiter",
        "function_word",
        "content_word",
        "number",
        "operator",
        "named_entity",
        "sentence_boundary",
        "clause_boundary",
        "reasoning_transition",
        "repeated_span",
        "newline",
        "special",
    }
)

# Closed-class / function words (English-centric; overlapping with content is
# intentional only when a piece is multi-token-ish — single pieces match one).
_FUNCTION_WORDS: frozenset[str] = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "if",
        "then",
        "else",
        "of",
        "to",
        "in",
        "on",
        "at",
        "for",
        "from",
        "by",
        "with",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "can",
        "shall",
        "not",
        "no",
        "nor",
        "so",
        "than",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
        "they",
        "them",
        "their",
        "we",
        "our",
        "you",
        "your",
        "he",
        "she",
        "his",
        "her",
        "i",
        "my",
        "me",
        "who",
        "whom",
        "which",
        "what",
        "when",
        "where",
        "why",
        "how",
        "there",
        "here",
        "into",
        "onto",
        "upon",
        "about",
        "over",
        "under",
        "between",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "out",
        "up",
        "down",
        "off",
        "again",
        "further",
        "once",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "only",
        "own",
        "same",
        "too",
        "very",
        "just",
        "also",
        "than",
    }
)

_REASONING_TRANSITIONS: frozenset[str] = frozenset(
    {
        "therefore",
        "thus",
        "hence",
        "however",
        "nevertheless",
        "nonetheless",
        "moreover",
        "furthermore",
        "meanwhile",
        "consequently",
        "accordingly",
        "because",
        "since",
        "although",
        "though",
        "whereas",
        "instead",
        "otherwise",
        "finally",
        "first",
        "second",
        "third",
        "next",
        "then",
        "lastly",
        "overall",
        "summary",
        "conclude",
        "conclusion",
        "step",
        "proof",
        "lemma",
        "assume",
        "suppose",
        "claim",
        "show",
        "prove",
        "wait",
        "hmm",
        "let",
        "lets",
        "let's",
        "okay",
        "ok",
        "note",
        "recall",
        "observe",
        "consider",
        "given",
        "henceforth",
    }
)

_CODE_DELIMS: frozenset[str] = frozenset(
    {
        "(",
        ")",
        "[",
        "]",
        "{",
        "}",
        "<",
        ">",
        "</",
        "/>",
        "->",
        "=>",
        "::",
        "...",
        "`",
        "```",
        "'",
        '"',
        "'''",
        '"""',
        "\\",
        "#",
        "@",
        "$",
        ";",
    }
)

_OPERATORS: frozenset[str] = frozenset(
    {
        "+",
        "-",
        "*",
        "/",
        "//",
        "%",
        "=",
        "==",
        "!=",
        "<=",
        ">=",
        "+=",
        "-=",
        "*=",
        "/=",
        "&&",
        "||",
        "&",
        "|",
        "^",
        "~",
        "<<",
        ">>",
        "**",
        ":=",
        "<-",
    }
)

_SENTENCE_END: frozenset[str] = frozenset({".", "!", "?", "。", "！", "？"})
_CLAUSE_PUNCT: frozenset[str] = frozenset({",", ";", ":", "—", "–", "…"})

_SPECIAL_PIECES: frozenset[str] = frozenset(
    {
        "",
        "<s>",
        "</s>",
        "<pad>",
        "<unk>",
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|end|>",
        "<|begin_of_text|>",
        "<|eot_id|>",
    }
)

# SentencePiece / BPE often attaches a leading space or "▁"/"Ġ" marker.
_LEAD_SPACE_RE = re.compile(r"^[\s▁Ġ]+")
_TRAIL_SPACE_RE = re.compile(r"[\s]+$")
_ALPHA_RE = re.compile(r"[A-Za-z]+")
_NUM_RE = re.compile(r"\d")
_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def _strip_bpe_markers(piece: str) -> str:
    """Normalize common BPE/SentencePiece space markers for lexical tests."""
    s = piece.replace("▁", " ").replace("Ġ", " ").replace("Ċ", "\n")
    return s


def _core_text(piece: str) -> str:
    s = _strip_bpe_markers(piece).strip()
    return s


def _is_whitespace_only(piece: str) -> bool:
    s = _strip_bpe_markers(piece)
    return len(s) > 0 and s.isspace()


def _has_newline(piece: str) -> bool:
    s = piece.replace("Ċ", "\n")
    return "\n" in s or "\r" in s


def annotate_categories(
    piece: str,
    *,
    token_id: int | None = None,
    context_pieces: Iterable[str] | None = None,
) -> frozenset[str]:
    """Return overlapping category labels for one decoded piece.

    Ambiguity is preserved: e.g. a piece can be both ``operator`` and
    ``code_delimiter``, or both ``reasoning_transition`` and ``function_word``.
    """
    labels: set[str] = set()
    ctx = list(context_pieces) if context_pieces is not None else []

    if piece in _SPECIAL_PIECES or (token_id is not None and token_id < 0):
        labels.add("special")
        # Special tokens may still participate in repetition checks below.
        if piece and piece in ctx:
            labels.add("repeated_span")
        return frozenset(labels)

    if _has_newline(piece):
        labels.add("newline")
        labels.add("whitespace")

    if _is_whitespace_only(piece):
        labels.add("whitespace")
        # Pure whitespace: still check repetition.
        if piece in ctx:
            labels.add("repeated_span")
        return frozenset(labels)

    # Leading-space BPE pieces: mark whitespace *and* continue on the core.
    stripped = _strip_bpe_markers(piece)
    if stripped[:1].isspace() and stripped.strip():
        labels.add("whitespace")

    core = _core_text(piece)
    if not core:
        if not labels:
            labels.add("whitespace")
        if piece in ctx:
            labels.add("repeated_span")
        return frozenset(labels)

    # Exact multi-char symbols first.
    if core in _CODE_DELIMS:
        labels.add("code_delimiter")
        if core in _SENTENCE_END:
            labels.add("sentence_boundary")
            labels.add("punctuation")
        if core in _CLAUSE_PUNCT:
            labels.add("clause_boundary")
            labels.add("punctuation")
    if core in _OPERATORS:
        labels.add("operator")
        # Many operators also appear in code.
        if core in {"=", "==", "!=", "<=", ">=", "->", "=>", "::", "+", "-", "*", "/"}:
            labels.add("code_delimiter")

    # Single-character punctuation / delimiters not caught above.
    if len(core) == 1:
        ch = core
        if ch in _SENTENCE_END:
            labels.add("sentence_boundary")
            labels.add("punctuation")
        elif ch in _CLAUSE_PUNCT:
            labels.add("clause_boundary")
            labels.add("punctuation")
        elif ch in string.punctuation:
            labels.add("punctuation")
            if ch in "()[]{}<>":
                labels.add("code_delimiter")
            if ch in "+-*/%=<>!&|^~":
                labels.add("operator")

    # Numbers (digit presence in core).
    if _NUM_RE.search(core):
        labels.add("number")
        # Pure numeric / numeric-with-punct (e.g. 3.14, 1_000).
        if re.fullmatch(r"[\d_.,]+", core):
            pass  # number only among lexical classes
        elif re.fullmatch(r"0[xX][0-9a-fA-F]+", core):
            labels.add("code_delimiter")  # hex-ish code literal cue

    # Lexical word classes.
    words = _WORD_RE.findall(core)
    if words:
        lower_words = [w.lower() for w in words]
        if any(w in _REASONING_TRANSITIONS for w in lower_words):
            labels.add("reasoning_transition")
        if any(w in _FUNCTION_WORDS for w in lower_words):
            labels.add("function_word")
        # Content word: alphabetic token that is not purely function/transition.
        for w, lw in zip(words, lower_words):
            if lw not in _FUNCTION_WORDS and lw not in _REASONING_TRANSITIONS:
                labels.add("content_word")
                break
        # If only function/transition words, still not content_word.
        if (
            "function_word" not in labels
            and "reasoning_transition" not in labels
            and any(c.isalpha() for c in core)
        ):
            labels.add("content_word")

        # Named entity heuristic: Capitalized word not at pure sentence start
        # markers; keep overlapping with content_word.
        for w in words:
            if len(w) >= 2 and w[0].isupper() and any(c.islower() for c in w[1:]):
                labels.add("named_entity")
                labels.add("content_word")
                break
            if w.isupper() and len(w) >= 2 and w.lower() not in _FUNCTION_WORDS:
                # Acronyms (HTTP, JSON) — treat as named_entity + content.
                labels.add("named_entity")
                labels.add("content_word")
                break

    # Fallback: if nothing lexical/structural matched, tag residual punct.
    if not labels:
        if all(c in string.punctuation for c in core):
            labels.add("punctuation")
        elif _ALPHA_RE.search(core):
            labels.add("content_word")
        else:
            labels.add("punctuation")

    # Repeated / copied span: exact piece match earlier in the stream.
    if piece in ctx or (core and any(_core_text(p) == core for p in ctx)):
        labels.add("repeated_span")

    # Guard: never emit unknown labels.
    unknown = labels - KNOWN_CATEGORIES
    if unknown:
        raise RuntimeError(f"internal: unknown categories {unknown}")

    return frozenset(labels)
