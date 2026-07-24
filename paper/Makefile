# Manuscript build (I19). Run as `make -C paper` from the repo root, or `make`
# from inside paper/.
#
#   make            build main.pdf from main.tex + the tracked figures/*.pdf
#   make figures    regenerate figures/*.pdf from the sealed analysis artifacts
#   make artifacts  pull those artifacts from the Modal volume into ../artifacts
#   make clean      remove build products (including main.pdf)
#
# The figure PDFs are tracked in git precisely so that a plain `make` works on a
# fresh clone with no Modal access and no artifact mirror. You only need
# `make artifacts && make figures` when the underlying analysis JSONs change.
#
# Provenance (AGENTS.md): every number in the manuscript, in a table or in a
# figure, comes from a script-generated immutable artifact. `make figures` only
# reshapes recorded numbers into geometry; it computes no statistics.

TEX     := main.tex
PDF     := main.pdf
PYTHON  ?= python3
LATEX   := pdflatex -interaction=nonstopmode -halt-on-error

FIGDIR  := figures
FIGS    := $(wildcard $(FIGDIR)/*.pdf)

# Sealed analysis artifacts the figure script reads. intervene.json exists for
# the two settings that were intervened on only; its absence on the third is
# expected, hence the `|| true` in the pull loop.
VOLUME  := cas-artifacts
RUNS    := sweep-2026-07-11T203836 sweep-v2-f8-2026-07-13 sweep-llama-f8-2026-07-13
JSONS   := c04_atlas_dev.json c04_atlas_test.json \
           autoresearch_test_domctl.json autoresearch_frozen_transfer_domctl.json \
           autoresearch_length_dev_domctl.json intervene.json
MIRROR  := ../artifacts/analysis

.PHONY: all figures artifacts readability invariants check clean

# pdflatex writes main.pdf before the reference check below can reject it.
# Without this, a failed build leaves a stale PDF that is newer than main.tex,
# so the next `make` reports "nothing to be done" and the failure looks fixed.
.DELETE_ON_ERROR:

all: $(PDF)

# Two passes: the first writes main.aux, the second resolves \ref to the
# tables, figures and sections. The grep is the build gate -- LaTeX exits 0 on
# unresolved references, so an unchecked build silently ships "??" in the PDF.
$(PDF): $(TEX) $(FIGS)
	$(LATEX) $(TEX)
	$(LATEX) $(TEX)
	@if grep -qi 'undefined' main.log; then \
	  echo "BUILD FAILED: undefined references or citations (see main.log)"; \
	  exit 1; \
	fi
	@echo "built $(PDF): $$(grep -o 'Output written.*' main.log)"

figures:
	$(PYTHON) ../scripts/make_figures.py

# Prose gates. `readability` reports Flesch Reading Ease per section (target
# >= 70). `invariants` fails if a rewrite changed any number, shed scope
# vocabulary, lost a label or figure, or claimed a mechanism (G2/D020).
# `check` runs both, and is what to run after editing prose.
readability:
	$(PYTHON) ../scripts/readability.py --sections

invariants:
	$(PYTHON) ../scripts/check_invariants.py

check: readability invariants

artifacts:
	@for r in $(RUNS); do \
	  mkdir -p $(MIRROR)/$$r; \
	  for f in $(JSONS); do \
	    modal volume get --force $(VOLUME) analysis/$$r/$$f $(MIRROR)/$$r/$$f \
	      >/dev/null 2>&1 && echo "ok   $$r/$$f" || echo "skip $$r/$$f (absent)"; \
	  done; \
	done

clean:
	rm -f main.aux main.log main.out main.toc $(PDF)
