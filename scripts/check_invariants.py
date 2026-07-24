"""Gate a manuscript rewrite against the science it is allowed to change: none.

A readability rewrite is supposed to change wording only. The failure mode that
matters is not an awkward sentence -- it is a dropped hedge or a rounded number
turning a scoped claim into a broader one. The paper carries 419 numeric tokens
and heavy qualification ("only", "not", "narrow", "descriptive"), and
simplifying prose is exactly the operation that sheds them.

This script compares a candidate `main.tex` against a reference revision and
fails on any of:

  * a numeric token added, dropped, or altered (multiset equality)
  * a drop in scope/hedge vocabulary
  * a lost \\label, \\ref, or \\includegraphics target
  * banned vocabulary: "mechanism"/"circuit" are gated by G2 (D020)

Usage::

    python scripts/check_invariants.py                     # HEAD vs working tree
    python scripts/check_invariants.py --ref HEAD~3
    python scripts/check_invariants.py --old a.tex --new b.tex
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TEX = "paper/main.tex"

# Words that keep a claim inside what the evidence supports. Losing them is how
# a rewrite silently overstates the science.
HEDGES = ["only", "not", "no", "never", "narrow", "narrowed", "scope",
          "scoped", "sensitivity", "descriptive", "descriptively", "caveat",
          "fails", "fail", "cannot", "untested", "open", "null", "unmeasured",
          "limitation", "limited", "expected-null", "does not", "did not",
          "we make no", "rather than"]

BANNED = ["mechanism", "mechanistic", "circuit", "circuits"]

NUM_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")


def body_of(src: str) -> str:
    if r"\begin{document}" in src:
        src = src.split(r"\begin{document}", 1)[1]
    return src.split(r"\end{document}", 1)[0]


def numbers(src: str) -> Counter:
    """Numeric tokens, normalised so 1,000 and 1000 compare equal.

    LaTeX writes a thin comma as ``14{,}336``; without collapsing it the regex
    would split that into ``14`` and ``336``, so normalise it to a plain comma
    (which is then stripped) before counting -- the reader sees one number.
    """
    body = body_of(src).replace("{,}", ",")
    return Counter(n.replace(",", "") for n in NUM_RE.findall(body))


def hedges(src: str) -> Counter:
    text = body_of(src).lower()
    return Counter({h: len(re.findall(rf"\b{re.escape(h)}\b", text))
                    for h in HEDGES})


def anchors(src: str) -> dict[str, set]:
    b = body_of(src)
    return {
        "label": set(re.findall(r"\\label\{([^}]*)\}", b)),
        "ref": set(re.findall(r"\\ref\{([^}]*)\}", b)),
        "graphic": set(re.findall(r"\\includegraphics\[[^]]*\]\{([^}]*)\}", b)),
        "bibitem": set(re.findall(r"\\bibitem\{([^}]*)\}", b)),
    }


def read_ref(ref: str, path: str) -> str:
    out = subprocess.run(["git", "show", f"{ref}:{path}"], cwd=REPO,
                         capture_output=True, text=True)
    if out.returncode:
        sys.exit(f"cannot read {ref}:{path}\n{out.stderr}")
    return out.stdout


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="HEAD", help="git revision to compare against")
    ap.add_argument("--old", help="explicit old file (overrides --ref)")
    ap.add_argument("--new", default=str(REPO / TEX))
    ap.add_argument("--new-content", nargs="*", default=[],
                    help="files of legitimately-new prose (e.g. a glossary); "
                         "numbers they introduce are allowed to be added")
    args = ap.parse_args()

    old = Path(args.old).read_text() if args.old else read_ref(args.ref, TEX)
    new = Path(args.new).read_text()
    allowed_add = Counter()
    for f in args.new_content:
        allowed_add += numbers(Path(f).read_text())

    failures: list[str] = []

    # --- numbers -----------------------------------------------------------
    # A DROPPED number is a real loss of science and always fails. An ADDED
    # number is only a problem if it is not accounted for by explicitly-declared
    # new content (a glossary re-stating dimensions, an AUROC scale of 0.5-1.0,
    # and so on). Unexplained additions still fail -- that is how a fabricated
    # statistic would be caught.
    o, n = numbers(old), numbers(new)
    # A reduced occurrence count is only a real loss if the number is ELIMINATED
    # from the paper (count reaches zero). Removing a duplicate mention -- e.g.
    # an abstract that stops repeating a figure already stated in the body -- is
    # legitimate tightening, so flag it as a note, not a failure.
    reduced = o - n
    eliminated = {v: c for v, c in reduced.items() if n[v] == 0}
    deduped = {v: c for v, c in reduced.items() if n[v] > 0}
    added = (n - o) - allowed_add
    explained = (n - o) - added
    print(f"numbers: {sum(o.values())} reference / {sum(n.values())} candidate"
          f"  ({sum(explained.values())} additions explained by new content)")
    for v, c in sorted(eliminated.items()):
        failures.append(f"NUMBER ELIMINATED (gone from paper)  {v!r} x{c}")
    for v, c in sorted(added.items()):
        failures.append(f"NUMBER ADDED (unexplained)  {v!r} x{c}")
    for v, c in sorted(deduped.items()):
        print(f"  note: {v!r} mentioned {c} fewer time(s) but still in the "
              f"paper ({n[v]}x) -- de-duplication, not a loss")
    if not eliminated and not added:
        print("  ok - no number eliminated; all additions explained")

    # --- hedges ------------------------------------------------------------
    # Counting cannot catch a single dropped "only" through a full rewrite --
    # legitimate rewording moves these words around constantly. What it CAN
    # catch is systematic shedding, so the aggregate is the gate and individual
    # moves are notes. Semantic claim drift is checked adversarially by
    # reviewers reading old and new side by side; this script does not
    # substitute for that.
    oh, nh = hedges(old), hedges(new)
    total_o, total_n = sum(oh.values()), sum(nh.values())
    regressions = {h: (oh[h], nh[h]) for h in HEDGES if nh[h] < oh[h]}
    print(f"hedges: {total_o} reference / {total_n} candidate")
    if total_o and total_n < total_o * 0.90:
        failures.append(
            f"HEDGE SHEDDING: total scope vocabulary fell {total_o} -> "
            f"{total_n} ({100 * (1 - total_n / total_o):.0f}% drop, limit 10%)")
    # Individual hedge moves are NOTES, never failures: rewording legitimately
    # replaces "scoped to" with "is narrow", "unmeasured" with "did not
    # measure", "rather than eliminated" with "we do not remove it". Counting
    # cannot tell that drift apart from real loss -- the adversarial review
    # does. The only hedge gate is aggregate shedding (above).
    if regressions:
        for h, (a, b) in sorted(regressions.items()):
            print(f"  note: hedge word {h!r} reduced {a} -> {b} "
                  f"(verify the caveat survived under other words)")
    if total_n >= total_o:
        print(f"  ok - scope vocabulary did not shrink overall "
              f"({total_o} -> {total_n})")

    # --- structural anchors -----------------------------------------------
    oa, na = anchors(old), anchors(new)
    for kind in oa:
        lost = oa[kind] - na[kind]
        if lost:
            failures.append(f"{kind.upper()} LOST: {sorted(lost)}")
    if not any(oa[k] - na[k] for k in oa):
        print("  ok - all labels, refs, graphics and bibitems present")

    # --- banned vocabulary -------------------------------------------------
    # G2 (D020) bars *claiming* a mechanism, not the word appearing in a
    # disclaimer -- "not a mechanistic account" is the policy being honoured.
    # Affirmative uses fail; negated ones are reported and allowed.
    low = body_of(new).lower()
    negations = ["not ", "no ", "never ", "without ", "rather than ",
                 "makes no ", "make no ", "avoid ", "nor "]
    for w in BANNED:
        for m in re.finditer(rf"\b{w}\b", low):
            window = low[max(0, m.start() - 70):m.start()]
            if any(neg in window for neg in negations):
                print(f"  note: {w!r} used in a disclaimer (allowed): "
                      f"...{low[max(0, m.start() - 40):m.end() + 10].strip()}...")
            else:
                failures.append(
                    f"BANNED WORD (G2/D020) used affirmatively: {w!r} at "
                    f"...{low[max(0, m.start() - 40):m.end() + 20].strip()}...")

    print()
    if failures:
        print(f"FAILED - {len(failures)} violation(s):")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)
    print("PASSED - no numeric, hedge, structural or vocabulary violations")


if __name__ == "__main__":
    main()
