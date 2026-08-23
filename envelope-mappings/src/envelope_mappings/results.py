"""Result types returned by EnvelopeClassifier.classify().

Three distinct outcomes, deliberately not folded into one flagged result:
each means a different thing and routes to a different place.

- PatternRecord     -- confident match, fields already extracted.
- AmbiguousMatch     -- matched one *company* (logo), but no single
                        year-variant template was confident enough.
                        Needs a human to pick the right template.
- NewTemplateNeeded  -- nothing matched with any confidence, at either
                        the company (logo) stage or the template stage.
                        Needs a new EnvelopeTemplate subclass written.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from envelope_mappings.template import EnvelopeTemplate


@dataclass
class PatternRecord:
    """A confident match: the winning template's extracted fields."""

    envelope: Any
    template_name: str
    company: str
    year_code: int | None
    confidence: float
    fields: dict[str, Any]


@dataclass
class AmbiguousMatch:
    """Company identified (logo matched), but no year-variant template
    was confident enough to extract from automatically.

    `candidates` is the top few (template, score) pairs, best first --
    the plausible contenders a human would choose between.
    """

    envelope: Any
    company: str
    candidates: list[tuple[EnvelopeTemplate, float]] = field(default_factory=list)


@dataclass
class NewTemplateNeeded:
    """Nothing matched with confidence -- either no logo matched at all,
    or a logo matched but no template for that company fit.

    `closest_scores` is optional context (top few near-misses, whatever
    stage failed) -- useful when present, not guaranteed informative for
    a true one-off, so callers should not rely on it being populated.
    """

    envelope: Any
    stage: str  # "company" or "template" -- which stage failed to match
    closest_scores: list[tuple[str, float]] | None = None
