"""EnvelopeTemplate -- the agnostic parent, extended once per company/
year-variant.

The parent owns IDENTIFICATION mechanics only (thresholds, company/year
parsing from the class name). It has zero company-specific knowledge.
Subclasses own EXTRACTION only (field_regions, field_validators,
extract_fields()) -- the parent never touches those.

Extending coverage for a new company or era is purely additive: define a
subclass, append an instance to the classifier's `templates` list (and a
CompanyLogo to `logos` if it's a new company). Nothing here needs to
change.

Class naming convention (drives auto-derivation of `company`/`year_code`):
    <Company><Year>   e.g. Vogue1970, McCalls1965

    - `company` auto-derives from the leading letters of the class name.
      Two-word companies need an explicit override (auto-derivation would
      only catch the first word) -- e.g.:
          class KwikSew1985(EnvelopeTemplate):
              company = "KwikSew"
    - `year_code` auto-derives from the trailing digits. Override if a
      template needs something other than a single anchor year (e.g. a
      range) -- the attribute is a plain int by default but nothing here
      enforces that; override the type too if needed.
"""

from __future__ import annotations

import re
from typing import Callable

import numpy as np

from envelope_mappings.fingerprint import EnvelopeFingerprint

_CLASSNAME_PATTERN = re.compile(r"^([A-Za-z]+?)(\d+)$")


class EnvelopeTemplate:
    HIGH_THRESHOLD = 0.75
    LOW_THRESHOLD = 0.45

    company: str | None = None
    year_code: int | None = None

    # Fractional bounding boxes (0.0-1.0 of envelope width/height) per
    # field, and a validator per field -- both intentionally empty here.
    # Subclasses fill these in; the parent/classifier never reads them
    # directly except via extract_fields(), which subclasses implement.
    field_regions: dict[str, tuple[float, float, float, float]] = {}
    field_validators: dict[str, Callable[[str], bool]] = {}

    # The reference fingerprint this template matches against. Not set at
    # class-definition time -- a subclass instance needs a real reference
    # envelope image to compute this from, so it's set explicitly via
    # set_reference() once you have one (e.g. in __init__, or after
    # construction). classify() will treat a template with no reference
    # set as unable to match (score 0) rather than raising, so a
    # half-finished template doesn't crash the whole classifier.
    reference_fingerprint: EnvelopeFingerprint | None = None

    def set_reference(self, reference_image: np.ndarray) -> None:
        """Computes and stores this template's reference fingerprint from
        a representative envelope image. Call this once per template
        instance before using it in a classifier.
        """
        self.reference_fingerprint = self.fingerprint(reference_image)

    def __init__(self):
        if self.company is None or self.year_code is None:
            derived_company, derived_year = self._parse_classname()
            if self.company is None:
                self.company = derived_company
            if self.year_code is None:
                self.year_code = derived_year

    def _parse_classname(self) -> tuple[str, int | None]:
        match = _CLASSNAME_PATTERN.match(type(self).__name__)
        if not match:
            return type(self).__name__, None
        return match.group(1), int(match.group(2))

    def fingerprint(self, envelope: np.ndarray) -> EnvelopeFingerprint:
        """Default: the generic color/edge/text-layout fingerprint.
        Override if a particular template needs different logic (e.g.
        cropping to a specific region before fingerprinting).
        """
        return EnvelopeFingerprint.compute(envelope)

    def extract_fields(self, envelope: np.ndarray) -> dict:
        """Must be implemented per subclass -- uses this template's own
        field_regions/field_validators. The parent/classifier never
        calls this until a match has already cleared HIGH_THRESHOLD.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement extract_fields()"
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(company={self.company!r}, "
            f"year_code={self.year_code!r})"
        )
