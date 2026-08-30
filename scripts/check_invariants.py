"""Verify that a manuscript rewrite preserves the underlying science.

The rewrite may improve wording and presentation, but it must not:

- remove scientific numbers,
- introduce unexplained numbers,
- systematically remove caveats/hedges,
- lose LaTeX structural anchors, or
- make affirmative mechanistic claims.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

REPO = Path(__file__).resolve().parent.parent
TEX = "paper/main.tex"

# Words that tend to keep claims within the scope supported by the evidence.
HEDGES = [
    "only",
    "not",
    "no",
    "never",
    "narrow",
    "narrowed",
    "scope",
    "scoped",
    "sensitivity",
    "descriptive",
    "descriptively",
    "caveat",
    "fails",
    "fail",
    "cannot",
    "untested",
    "open",
    "null",
    "unmeasured",
    "limitation",
    "limited",
    "expected-null",
    "does not",
    "did not",
    "we make no",
    "rather than",
]

BANNED = [
    "mechanism",
    "mechanistic",
    "circuit",
    "circuits",
]

NUM_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")


# -----------------------------------------------------------------------------
# Manuscript parsing
# -----------------------------------------------------------------------------

def body_of(source: str) -> str:
    """Return only the contents of the LaTeX document body."""
    if r"\begin{document}" in source:
        source = source.split(r"\begin{document}", 1)[1]

    return source.split(r"\end{document}", 1)[0]


def numbers(source: str) -> Counter:
    """Count numeric tokens in the document body.

    Numbers are normalized so equivalent forms compare equally:

        1,000     -> 1000
        14{,}336  -> 14336

    LaTeX sometimes writes commas as ``{,}``. We first turn those into normal
    commas, then remove commas from all numeric tokens before counting them.
    """
    body = body_of(source).replace("{,}", ",")

    return Counter(
        number.replace(",", "")
        for number in NUM_RE.findall(body)
    )


def hedges(source: str) -> Counter:
    """Count occurrences of scope/caveat vocabulary."""
    text = body_of(source).lower()

    return Counter({
        hedge: len(re.findall(rf"\b{re.escape(hedge)}\b", text))
        for hedge in HEDGES
    })


def anchors(source: str) -> dict[str, set]:
    """Collect structural LaTeX anchors that should survive a rewrite."""
    body = body_of(source)

    return {
        "label": set(re.findall(r"\\label\{([^}]*)\}", body)),
        "ref": set(re.findall(r"\\ref\{([^}]*)\}", body)),
        "graphic": set(
            re.findall(r"\\includegraphics\[[^]]*\]\{([^}]*)\}", body)
        ),
        "bibitem": set(re.findall(r"\\bibitem\{([^}]*)\}", body)),
    }


# -----------------------------------------------------------------------------
# Input loading
# -----------------------------------------------------------------------------

def read_ref(ref: str, path: str) -> str:
    """Read a file from a Git revision."""
    result = subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )

    if result.returncode:
        sys.exit(f"cannot read {ref}:{path}\n{result.stderr}")

    return result.stdout


# -----------------------------------------------------------------------------
# Main validation
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ref",
        default="HEAD",
        help="git revision to compare against",
    )
    parser.add_argument(
        "--old",
        help="explicit old file (overrides --ref)",
    )
    parser.add_argument(
        "--new",
        default=str(REPO / TEX),
    )
    parser.add_argument(
        "--new-content",
        nargs="*",
        default=[],
        help=(
            "files of legitimately-new prose (e.g. a glossary); "
            "numbers they introduce are allowed to be added"
        ),
    )

    args = parser.parse_args()

    # Load reference and candidate manuscripts.
    if args.old:
        old = Path(args.old).read_text()
    else:
        old = read_ref(args.ref, TEX)

    new = Path(args.new).read_text()

    # Numbers appearing in explicitly declared new content are allowed additions.
    allowed_additions = Counter()
    for path in args.new_content:
        allowed_additions += numbers(Path(path).read_text())

    failures: list[str] = []

    # -------------------------------------------------------------------------
    # 1. Numbers
    # -------------------------------------------------------------------------
    #
    # Removing a number completely is treated as loss of scientific content.
    #
    # Adding a number is allowed only if it comes from explicitly declared
    # new content. This prevents a rewrite from silently inventing statistics.
    #
    # Removing duplicate mentions is allowed. For example, a number may disappear
    # from the abstract while remaining in the results section.
    # -------------------------------------------------------------------------

    old_numbers = numbers(old)
    new_numbers = numbers(new)

    reduced = old_numbers - new_numbers

    eliminated = {
        value: count
        for value, count in reduced.items()
        if new_numbers[value] == 0
    }

    deduped = {
        value: count
        for value, count in reduced.items()
        if new_numbers[value] > 0
    }

    added = (new_numbers - old_numbers) - allowed_additions
    explained = (new_numbers - old_numbers) - added

    print(
        f"numbers: {sum(old_numbers.values())} reference / "
        f"{sum(new_numbers.values())} candidate"
        f"  ({sum(explained.values())} additions explained by new content)"
    )

    for value, count in sorted(eliminated.items()):
        failures.append(
            f"NUMBER ELIMINATED (gone from paper)  {value!r} x{count}"
        )

    for value, count in sorted(added.items()):
        failures.append(
            f"NUMBER ADDED (unexplained)  {value!r} x{count}"
        )

    for value, count in sorted(deduped.items()):
        print(
            f"  note: {value!r} mentioned {count} fewer time(s) but still in "
            f"the paper ({new_numbers[value]}x) -- de-duplication, not a loss"
        )

    if not eliminated and not added:
        print("  ok - no number eliminated; all additions explained")

    # -------------------------------------------------------------------------
    # 2. Hedges and caveats
    # -------------------------------------------------------------------------
    #
    # Individual words are allowed to move or be replaced during rewriting.
    # For example:
    #
    #   "scoped to"       -> "is narrow"
    #   "unmeasured"      -> "did not measure"
    #   "rather than..."  -> "we do not..."
    #
    # Therefore individual reductions are only notes.
    #
    # The actual gate looks for systematic hedge shedding: total hedge vocabulary
    # may not fall by more than 10%.
    #
    # This is only a coarse automated check. Reviewers must still compare the old
    # and new manuscripts for semantic claim drift.
    # -------------------------------------------------------------------------

    old_hedges = hedges(old)
    new_hedges = hedges(new)

    old_hedge_total = sum(old_hedges.values())
    new_hedge_total = sum(new_hedges.values())

    regressions = {
        hedge: (old_hedges[hedge], new_hedges[hedge])
        for hedge in HEDGES
        if new_hedges[hedge] < old_hedges[hedge]
    }

    print(
        f"hedges: {old_hedge_total} reference / "
        f"{new_hedge_total} candidate"
    )

    if (
        old_hedge_total
        and new_hedge_total < old_hedge_total * 0.90
    ):
        failures.append(
            f"HEDGE SHEDDING: total scope vocabulary fell "
            f"{old_hedge_total} -> {new_hedge_total} "
            f"({100 * (1 - new_hedge_total / old_hedge_total):.0f}% drop, "
            f"limit 10%)"
        )

    for hedge, (before, after) in sorted(regressions.items()):
        print(
            f"  note: hedge word {hedge!r} reduced {before} -> {after} "
            f"(verify the caveat survived under other words)"
        )

    if new_hedge_total >= old_hedge_total:
        print(
            f"  ok - scope vocabulary did not shrink overall "
            f"({old_hedge_total} -> {new_hedge_total})"
        )

    # -------------------------------------------------------------------------
    # 3. Structural anchors
    # -------------------------------------------------------------------------

    old_anchors = anchors(old)
    new_anchors = anchors(new)

    for kind in old_anchors:
        lost = old_anchors[kind] - new_anchors[kind]

        if lost:
            failures.append(
                f"{kind.upper()} LOST: {sorted(lost)}"
            )

    if not any(
        old_anchors[kind] - new_anchors[kind]
        for kind in old_anchors
    ):
        print("  ok - all labels, refs, graphics and bibitems present")

    # -------------------------------------------------------------------------
    # 4. Banned mechanistic vocabulary
    # -------------------------------------------------------------------------
    #
    # G2 / D020 prohibits *claiming* a mechanism.
    #
    # The words themselves are still allowed in explicit disclaimers such as:
    #
    #     "This is not a mechanistic account."
    #
    # Affirmative uses fail. Negated uses are reported but allowed.
    # -------------------------------------------------------------------------

    candidate_body = body_of(new).lower()

    negations = [
        "not ",
        "no ",
        "never ",
        "without ",
        "rather than ",
        "makes no ",
        "make no ",
        "avoid ",
        "nor ",
    ]

    for banned_word in BANNED:
        matches = re.finditer(
            rf"\b{banned_word}\b",
            candidate_body,
        )

        for match in matches:
            before = candidate_body[
                max(0, match.start() - 70):match.start()
            ]

            context = candidate_body[
                max(0, match.start() - 40):match.end() + 10
            ].strip()

            if any(negation in before for negation in negations):
                print(
                    f"  note: {banned_word!r} used in a disclaimer "
                    f"(allowed): ...{context}..."
                )
            else:
                failures.append(
                    f"BANNED WORD (G2/D020) used affirmatively: "
                    f"{banned_word!r} at "
                    f"...{candidate_body[max(0, match.start() - 40):match.end() + 20].strip()}..."
                )

    # -------------------------------------------------------------------------
    # Result
    # -------------------------------------------------------------------------

    print()

    if failures:
        print(f"FAILED - {len(failures)} violation(s):")

        for failure in failures:
            print(f"  {failure}")

        sys.exit(1)

    print(
        "PASSED - no numeric, hedge, structural or vocabulary violations"
    )


if __name__ == "__main__":
    main()
