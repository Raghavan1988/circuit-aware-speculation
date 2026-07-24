"""Measure the manuscript's readability from LaTeX source.

Reports Flesch Reading Ease (higher = easier; the target for this manuscript is
>= 70) and Flesch-Kincaid Grade Level, overall and per section.

What is measured, and why
-------------------------
Readability formulas are defined over prose. LaTeX source is not prose, so the
text is cleaned first:

  * dropped entirely: comments, math, tables, the bibliography, figure
    environments other than their captions, and citation brackets
  * kept: the abstract, body paragraphs, and figure/table captions
  * unwrapped: \\textbf{x}, \\emph{x}, \\texttt{x} -> x

Section headings are excluded by default. They are fragments, not sentences,
and counting them inflates the score by shortening mean sentence length. Pass
--with-headings to see their effect.

Numerals are a known sensitivity: "+0.0555" is one short token to a syllable
counter but is read aloud as many. The script therefore reports the score both
with numerals kept and with them removed, so the headline number can be
checked against the stricter reading.

The primary figure comes from `textstat`, a widely used third-party
implementation, so the score does not depend on a counter written here. A small
independent implementation is reported alongside it as a cross-check.

Usage::

    python scripts/readability.py                    # paper/main.tex
    python scripts/readability.py --file other.tex
    python scripts/readability.py --sections         # per-section breakdown
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import textstat

REPO = Path(__file__).resolve().parent.parent
DEFAULT_TEX = REPO / "paper" / "main.tex"

# Environments whose entire content is not prose.
DROP_ENVS = ["tabular", "table", "thebibliography", "equation", "align",
             "displaymath", "enumerate*", "verbatim"]


def strip_latex(src: str, *, keep_headings: bool = False) -> str:
    """Reduce LaTeX source to the prose a reader actually reads."""
    text = src

    # Comments (but not escaped \%).
    text = re.sub(r"(?<!\\)%.*?$", "", text, flags=re.MULTILINE)

    # Preamble: everything before \begin{document}.
    if r"\begin{document}" in text:
        text = text.split(r"\begin{document}", 1)[1]
    text = text.split(r"\end{document}", 1)[0]

    # Captions are prose and must survive the figure/table drop, so lift them
    # out before the environments around them are removed.
    captions = re.findall(r"\\caption\{", text)
    caption_texts = []
    for m in re.finditer(r"\\caption\{", text):
        start = m.end()
        depth, i = 1, start
        while i < len(text) and depth:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        caption_texts.append(text[start:i - 1])

    for env in DROP_ENVS:
        text = re.sub(rf"\\begin\{{{env}\}}.*?\\end\{{{env}\}}", " ", text,
                      flags=re.DOTALL)
    text = re.sub(r"\\begin\{figure\}.*?\\end\{figure\}", " ", text,
                  flags=re.DOTALL)

    text = text + "\n\n" + "\n\n".join(caption_texts)

    # Math.
    text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.DOTALL)
    text = re.sub(r"\$[^$]*\$", " ", text)
    text = re.sub(r"\\\[.*?\\\]", " ", text, flags=re.DOTALL)

    # Citations and cross-references are not read as words.
    text = re.sub(r"\[arXiv:[^\]]*\]", " ", text)
    text = re.sub(r"\\(ref|label|cite\w*)\{[^}]*\}", " ", text)
    text = re.sub(r"\\S\b", " Section ", text)

    # Headings.
    if keep_headings:
        text = re.sub(r"\\(sub)*section\*?\{([^}]*)\}", r"\2. ", text)
        text = re.sub(r"\\paragraph\{([^}]*)\}", r"\1. ", text)
    else:
        text = re.sub(r"\\(sub)*section\*?\{[^}]*\}", " ", text)
        text = re.sub(r"\\paragraph\{[^}]*\}", " ", text)

    # Inline formatting: keep the words, drop the wrapper.
    for _ in range(6):
        text = re.sub(r"\\(textbf|emph|textit|texttt|underline|text)\{([^{}]*)\}",
                      r"\2", text)

    text = re.sub(r"\\(begin|end)\{[^}]*\}", " ", text)
    text = re.sub(r"\\item\b", " ", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^{}]*\})?", " ", text)

    # Punctuation and spacing normalisation.
    text = text.replace("---", " - ").replace("--", " - ")
    text = text.replace("~", " ").replace("\\%", "%")
    text = re.sub(r"[{}]", " ", text)
    text = re.sub(r"``|''", '"', text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


VOWELS = "aeiouy"


def _syllables(word: str) -> int:
    """Independent syllable heuristic, used only as a cross-check."""
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    if len(w) <= 3:
        return 1
    w = re.sub(r"(?:[^laeiouy]es|[^laeiouy]e)$", "", w)
    w = re.sub(r"^y", "", w)
    n = len(re.findall(r"[aeiouy]{1,2}", w))
    return max(1, n)


SENTINEL = "\x00"


def _split_sentences(text: str) -> list[str]:
    """Split on sentence enders, shielding abbreviations and decimals first."""
    protected = text
    for abbr in ["e.g.", "i.e.", "cf.", "vs.", "Fig.", "Eq.", "et al.", "Dr.",
                 "approx.", "resp."]:
        protected = protected.replace(abbr, abbr.replace(".", SENTINEL))
    # A decimal point is not a sentence end: "0.055" must stay one token.
    protected = re.sub(r"(\d)\.(\d)", lambda m: m.group(1) + SENTINEL + m.group(2),
                       protected)
    parts = re.split(r"(?<=[.!?])\s+", protected)
    return [p.replace(SENTINEL, ".").strip() for p in parts if p.strip()]


def own_flesch(text: str) -> tuple[float, float, float, int]:
    sentences = _split_sentences(text)
    words = re.findall(r"[A-Za-z][A-Za-z'\-]*", text)
    if not sentences or not words:
        return 0.0, 0.0, 0.0, 0
    syl = sum(_syllables(w) for w in words)
    wps = len(words) / len(sentences)
    spw = syl / len(words)
    score = 206.835 - 1.015 * wps - 84.6 * spw
    return score, wps, spw, len(words)


def report(text: str, label: str, *, verbose: bool = True) -> float:
    no_numbers = re.sub(r"[+\-]?\d[\d,.]*", " ", text)
    ts = textstat.flesch_reading_ease(text)
    ts_nonum = textstat.flesch_reading_ease(no_numbers)
    grade = textstat.flesch_kincaid_grade(text)
    own, wps, spw, nwords = own_flesch(text)

    if verbose:
        print(f"{label}")
        print(f"  Flesch Reading Ease (textstat)      : {ts:6.1f}"
              f"   {'PASS' if ts >= 70 else 'below target'}")
        print(f"  ... numerals removed                : {ts_nonum:6.1f}"
              f"   {'PASS' if ts_nonum >= 70 else 'below target'}")
        print(f"  Flesch Reading Ease (own check)     : {own:6.1f}")
        print(f"  Flesch-Kincaid Grade Level          : {grade:6.1f}")
        print(f"  words/sentence {wps:5.1f}   syllables/word {spw:4.2f}"
              f"   words {nwords}")
    return ts


def split_sections(src: str) -> list[tuple[str, str]]:
    """Split raw LaTeX into (heading, body) pairs for a per-section report."""
    out = []
    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", src,
                         flags=re.DOTALL)
    if abstract:
        out.append(("Abstract", abstract.group(1)))
    pieces = re.split(r"\\section\*?\{([^}]*)\}", src)
    for i in range(1, len(pieces), 2):
        out.append((pieces[i], pieces[i + 1]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(DEFAULT_TEX))
    ap.add_argument("--sections", action="store_true")
    ap.add_argument("--with-headings", action="store_true")
    ap.add_argument("--target", type=float, default=70.0)
    args = ap.parse_args()

    src = Path(args.file).read_text()
    whole = strip_latex(src, keep_headings=args.with_headings)

    print(f"=== {args.file} ===\n")
    score = report(whole, "WHOLE DOCUMENT")

    if args.sections:
        print("\n--- per section (worst first) ---")
        rows = []
        for name, body in split_sections(src):
            clean = strip_latex(r"\begin{document}" + body + r"\end{document}",
                                keep_headings=args.with_headings)
            if len(clean.split()) < 40:
                continue
            s = textstat.flesch_reading_ease(clean)
            _, wps, spw, n = own_flesch(clean)
            rows.append((s, name, wps, spw, n))
        for s, name, wps, spw, n in sorted(rows):
            flag = "ok  " if s >= args.target else "LOW "
            print(f"  {flag} {s:6.1f}  {name[:38]:40s} "
                  f"w/s {wps:5.1f}  syl/w {spw:4.2f}  ({n} words)")

    raise SystemExit(0 if score >= args.target else 1)


if __name__ == "__main__":
    main()
