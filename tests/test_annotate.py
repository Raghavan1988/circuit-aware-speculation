"""I11: token-category + generation-phase annotation tests.

CPU / stdlib only (no torch). Includes a seeded stratified manual-sample
check: golden labels are fixed in this file; agreement is computed by the
test script (never hand-typed into a results table).
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict

import pytest

from cas.annotate import (
    CATEGORY_SET_VERSION,
    KNOWN_CATEGORIES,
    KNOWN_PHASES,
    PHASE_SET_VERSION,
    AnnotatedToken,
    annotate_sequence,
    annotate_token,
)
from cas.annotate.categories import annotate_categories
from cas.annotate.phases import (
    MID_END,
    PREFIX_END,
    annotate_phase,
    annotate_phase_relative,
)


# ---------------------------------------------------------------------------
# Unit: versions and known sets
# ---------------------------------------------------------------------------


def test_versions_are_nonempty_strings():
    assert isinstance(CATEGORY_SET_VERSION, str) and CATEGORY_SET_VERSION
    assert isinstance(PHASE_SET_VERSION, str) and PHASE_SET_VERSION
    assert CATEGORY_SET_VERSION.startswith("v")
    assert PHASE_SET_VERSION.startswith("v")


def test_known_categories_cover_research_spec_atlas():
    required = {
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
    }
    assert required <= KNOWN_CATEGORIES


# ---------------------------------------------------------------------------
# Unit: categories (overlapping)
# ---------------------------------------------------------------------------


def test_whitespace_and_newline():
    assert "whitespace" in annotate_categories("   ")
    assert "newline" in annotate_categories("\n")
    assert "whitespace" in annotate_categories("\n")


def test_punctuation_sentence_and_clause():
    assert {"punctuation", "sentence_boundary"} <= annotate_categories(".")
    assert {"punctuation", "clause_boundary"} <= annotate_categories(",")
    assert {"punctuation", "clause_boundary"} <= annotate_categories(";")


def test_code_delimiters_and_operators_overlap():
    cats = annotate_categories("->")
    assert "code_delimiter" in cats
    cats_eq = annotate_categories("==")
    assert "operator" in cats_eq
    assert "code_delimiter" in cats_eq


def test_function_word_not_forced_exclusive():
    cats = annotate_categories(" the")  # BPE-ish leading space
    assert "function_word" in cats
    # leading space marker may also yield whitespace
    # (preserve ambiguity: both allowed)


def test_content_word_and_named_entity_overlap():
    cats = annotate_categories(" Paris")
    assert "content_word" in cats
    assert "named_entity" in cats


def test_number():
    assert "number" in annotate_categories("42")
    assert "number" in annotate_categories("3.14")


def test_reasoning_transition():
    assert "reasoning_transition" in annotate_categories(" therefore")
    assert "reasoning_transition" in annotate_categories("However")


def test_repeated_span_uses_context():
    ctx = ["foo", " bar", "baz"]
    cats = annotate_categories(" bar", context_pieces=ctx)
    assert "repeated_span" in cats
    cats_new = annotate_categories(" qux", context_pieces=ctx)
    assert "repeated_span" not in cats_new


def test_special_token():
    assert "special" in annotate_categories("<|endoftext|>")
    assert "special" in annotate_categories("", token_id=-1)


def test_empty_category_set_allowed_only_if_rules_say_so():
    # Pure implementation always emits at least one label for non-empty input
    # or special; empty piece is special or whitespace.
    cats = annotate_categories("")
    assert "special" in cats or "whitespace" in cats


def test_categories_are_subset_of_known():
    samples = [
        "hello",
        " the",
        "123",
        "(",
        ")",
        "==",
        ".",
        ",",
        "\n",
        " therefore",
        "HTTP",
        "->",
        "```",
        "  ",
    ]
    for p in samples:
        cats = annotate_categories(p)
        assert cats <= KNOWN_CATEGORIES, (p, cats)


# ---------------------------------------------------------------------------
# Unit: phases
# ---------------------------------------------------------------------------


def test_absolute_phase_bins():
    assert annotate_phase(0) == "prefix"
    assert annotate_phase(PREFIX_END - 1) == "prefix"
    assert annotate_phase(PREFIX_END) == "mid"
    assert annotate_phase(MID_END - 1) == "mid"
    assert annotate_phase(MID_END) == "late"
    assert annotate_phase(10_000) == "late"


def test_phase_negative_raises():
    with pytest.raises(ValueError):
        annotate_phase(-1)


def test_relative_phase_tertiles():
    n = 9
    phases = [annotate_phase_relative(i, n) for i in range(n)]
    assert phases[0] == "prefix"
    assert "mid" in phases
    assert phases[-1] == "late"
    assert set(phases) <= KNOWN_PHASES


# ---------------------------------------------------------------------------
# Unit: public API
# ---------------------------------------------------------------------------


def test_annotate_token_signature_and_versions():
    ann = annotate_token(42, " hello", 0, [])
    assert isinstance(ann, AnnotatedToken)
    assert isinstance(ann.categories, frozenset)
    assert ann.phase == "prefix"
    assert ann.category_set_version == CATEGORY_SET_VERSION
    assert ann.phase_set_version == PHASE_SET_VERSION
    assert ann.categories_sorted() == sorted(ann.categories)


def test_annotate_sequence_left_to_right_repetition():
    pieces = ["foo", " bar", "foo"]
    anns = annotate_sequence(pieces)
    assert "repeated_span" not in anns[0].categories
    assert "repeated_span" in anns[2].categories  # second "foo"


def test_annotate_sequence_length_mismatch():
    with pytest.raises(ValueError):
        annotate_sequence(["a", "b"], token_ids=[1])


# ---------------------------------------------------------------------------
# Stratified manual-sample validation (scripted agreement)
# ---------------------------------------------------------------------------
# Golden labels are the human-checked expectations for a fixed sample.
# Agreement is computed below — do not paste a number into docs by hand.

# (domain, piece, position, context_pieces, expected_categories_superset,
#  expected_phase_absolute)
# expected_categories_superset: every listed label MUST appear (extras ok,
# preserving overlap). Use exact-equality cases via EXPECT_EXACT below.

_STRATIFIED_GOLDEN: list[tuple] = [
    # --- code domain ---
    ("code", "def", 0, [], {"content_word"}, "prefix"),
    ("code", "(", 1, ["def"], {"code_delimiter"}, "prefix"),
    ("code", "x", 2, ["def", "("], {"content_word"}, "prefix"),
    ("code", ")", 3, ["def", "(", "x"], {"code_delimiter"}, "prefix"),
    ("code", ":", 4, ["def", "(", "x", ")"], {"clause_boundary", "punctuation"}, "prefix"),
    ("code", "\n", 5, ["def", "(", "x", ")", ":"], {"newline", "whitespace"}, "prefix"),
    ("code", "    ", 6, ["def", "(", "x", ")", ":", "\n"], {"whitespace"}, "prefix"),
    ("code", "return", 7, ["def", "(", "x", ")", ":", "\n", "    "], {"content_word"}, "prefix"),
    ("code", "42", 8, ["def", "(", "x", ")", ":", "\n", "    ", "return"], {"number"}, "prefix"),
    ("code", "==", 9, ["a"], {"operator", "code_delimiter"}, "prefix"),
    # --- math domain ---
    ("math", "Let", 0, [], {"content_word"}, "prefix"),
    ("math", " x", 1, ["Let"], {"content_word"}, "prefix"),
    ("math", " =", 2, ["Let", " x"], {"operator"}, "prefix"),
    ("math", "3", 3, ["Let", " x", " ="], {"number"}, "prefix"),
    ("math", "+", 4, ["Let", " x", " =", "3"], {"operator"}, "prefix"),
    ("math", "4", 5, ["Let", " x", " =", "3", "+"], {"number"}, "prefix"),
    ("math", ".", 6, ["Let", " x", " =", "3", "+", "4"], {"sentence_boundary", "punctuation"}, "prefix"),
    ("math", " therefore", 7, [], {"reasoning_transition"}, "prefix"),
    ("math", " we", 8, [" therefore"], {"function_word"}, "prefix"),
    ("math", " have", 9, [" therefore", " we"], {"function_word"}, "prefix"),
    # --- chat domain ---
    ("chat", "Hello", 0, [], {"content_word", "named_entity"}, "prefix"),
    ("chat", "!", 1, ["Hello"], {"sentence_boundary", "punctuation"}, "prefix"),
    ("chat", " How", 2, ["Hello", "!"], {"function_word"}, "prefix"),
    ("chat", " are", 3, ["Hello", "!", " How"], {"function_word"}, "prefix"),
    ("chat", " you", 4, ["Hello", "!", " How", " are"], {"function_word"}, "prefix"),
    ("chat", "?", 5, [], {"sentence_boundary", "punctuation"}, "prefix"),
    ("chat", " the", 6, [], {"function_word"}, "prefix"),
    ("chat", " weather", 7, [" the"], {"content_word"}, "prefix"),
    ("chat", " the", 8, [" the", " weather"], {"function_word", "repeated_span"}, "prefix"),
    ("chat", "Paris", 9, [], {"named_entity", "content_word"}, "prefix"),
    # --- summ domain ---
    ("summ", "In", 0, [], {"function_word"}, "prefix"),
    ("summ", " summary", 1, ["In"], {"reasoning_transition"}, "prefix"),
    ("summ", ",", 2, ["In", " summary"], {"clause_boundary", "punctuation"}, "prefix"),
    ("summ", " the", 3, [], {"function_word"}, "prefix"),
    ("summ", " article", 4, [" the"], {"content_word"}, "prefix"),
    ("summ", " discusses", 5, [" the", " article"], {"content_word"}, "prefix"),
    ("summ", " climate", 6, [], {"content_word"}, "prefix"),
    ("summ", ".", 7, [], {"sentence_boundary", "punctuation"}, "prefix"),
    ("summ", "\n", 8, [], {"newline", "whitespace"}, "prefix"),
    ("summ", "Overall", 9, [], {"reasoning_transition"}, "prefix"),
    # --- phase bins (position-focused; piece arbitrary) ---
    ("phase", "x", PREFIX_END - 1, [], set(), "prefix"),
    ("phase", "x", PREFIX_END, [], set(), "mid"),
    ("phase", "x", MID_END, [], set(), "late"),
]


def _agreement(predicted: frozenset[str], required: set[str]) -> bool:
    """A sample agrees if every human-required label is present (extras ok)."""
    return required <= set(predicted)


def test_stratified_manual_sample_agreement():
    """Seeded stratified sample; agreement computed here (scripted).

    Draw is stratified by domain over the fixed golden pool (reproducible).
    Manual check = golden expected labels authored with the sample.
    """
    seed = 20260710
    rng = random.Random(seed)

    by_domain: dict[str, list] = defaultdict(list)
    for row in _STRATIFIED_GOLDEN:
        by_domain[row[0]].append(row)

    # Stratified: take all phase rows + up to 8 per content domain.
    sample: list = []
    for domain in ("code", "math", "chat", "summ"):
        pool = list(by_domain[domain])
        rng.shuffle(pool)
        sample.extend(pool[:8])
    sample.extend(by_domain["phase"])

    n = len(sample)
    n_agree = 0
    failures: list[str] = []
    domain_counts: Counter[str] = Counter()
    domain_agree: Counter[str] = Counter()

    for domain, piece, position, context, required, exp_phase in sample:
        domain_counts[domain] += 1
        ann = annotate_token(0, piece, position, list(context))
        cat_ok = _agreement(ann.categories, required) if required else True
        phase_ok = ann.phase == exp_phase
        ok = cat_ok and phase_ok
        if ok:
            n_agree += 1
            domain_agree[domain] += 1
        else:
            failures.append(
                f"{domain!r} piece={piece!r} pos={position}: "
                f"cats={sorted(ann.categories)} need>={sorted(required)}; "
                f"phase={ann.phase!r} need={exp_phase!r}"
            )

    rate = n_agree / n if n else 0.0
    # Print scripted metrics (captured by pytest -s); never hand-copy to tables.
    print(
        f"I11 stratified agreement: {n_agree}/{n} = {rate:.4f} "
        f"(seed={seed}, category_set={CATEGORY_SET_VERSION})"
    )
    for d in sorted(domain_counts):
        print(f"  domain {d}: {domain_agree[d]}/{domain_counts[d]}")

    assert n >= 20, "sample too small"
    assert set(domain_counts) >= {"code", "math", "chat", "summ", "phase"}
    # High bar for rule-based golden set: all must agree, else fix rules/goldens.
    assert not failures, "disagreements:\n" + "\n".join(failures)
    assert n_agree == n
    assert rate == 1.0


def test_stratified_sample_is_deterministic():
    """Same seed → same sample composition (reproducibility)."""
    seed = 20260710

    def draw():
        rng = random.Random(seed)
        by_domain: dict[str, list] = defaultdict(list)
        for row in _STRATIFIED_GOLDEN:
            by_domain[row[0]].append(row)
        sample = []
        for domain in ("code", "math", "chat", "summ"):
            pool = list(by_domain[domain])
            rng.shuffle(pool)
            sample.extend((domain, p[1], p[2]) for p in pool[:8])
        return sample

    assert draw() == draw()
