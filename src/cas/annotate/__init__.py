"""Token-category and generation-phase annotation (issue I11).

Pure functions of the token stream (decoded pieces + positions). No torch.
The trace writer (I06) should call :func:`annotate_token` per proposed/target
token once categories land on the schema.

Design notes: ``docs/DECISIONS.md`` D016; category list follows the acceptance
atlas list in ``docs/RESEARCH_SPEC.md``.
"""

from cas.annotate.categories import (
    CATEGORY_SET_VERSION,
    KNOWN_CATEGORIES,
    annotate_categories,
)
from cas.annotate.phases import (
    PHASE_SET_VERSION,
    KNOWN_PHASES,
    annotate_phase,
)
from cas.annotate.api import AnnotatedToken, annotate_sequence, annotate_token

__all__ = [
    "AnnotatedToken",
    "CATEGORY_SET_VERSION",
    "KNOWN_CATEGORIES",
    "KNOWN_PHASES",
    "PHASE_SET_VERSION",
    "annotate_categories",
    "annotate_phase",
    "annotate_sequence",
    "annotate_token",
]
