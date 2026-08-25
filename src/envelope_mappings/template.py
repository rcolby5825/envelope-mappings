"""EnvelopeTemplate -- the agnostic parent, extended once per company/
year-variant.

The parent owns IDENTIFICATION mechanics only (thresholds, company/year
parsing from the class name). It has zero company-specific knowledge.
Subclasses own EXTRACTION CONFIGURATION only (field_regions,
field_validators) -- the actual extraction MECHANISM lives in its own
module, extraction.py, as a standalone FieldExtractor -- see that
module's docstring for why it's kept separate.

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
    - `year_code` auto-derives from the trailing digits and is used as a
      single anchor/sort year -- kept as strictly-digits-only by design,
      so classname parsing stays simple and unambiguous. For real
      collection data, an exact year is often NOT known -- use
      `year_range` (a separate, optional (start, end) tuple you set
      explicitly) to express "somewhere in this span" without
      complicating the classname convention at all.

REFERENCE IMAGES ARE NOT PART OF THIS REPO. Real envelope photos/logo
crops live wherever you keep them locally -- set a PATH via
set_reference_path(), not the image data itself. Loading only happens
the first time the reference is actually needed (lazily), so you can
define a template and point it at a path before the file exists there
yet, and add the real image whenever. set_reference() (with an in-memory
array) still works too, e.g. for tests.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from envelope_mappings.extraction import extractor
from envelope_mappings.fingerprint import EnvelopeFingerprint
from envelope_mappings.results import FieldResult

_CLASSNAME_PATTERN = re.compile(r"^([A-Za-z]+?)(\d+)$")


class EnvelopeTemplate:
    HIGH_THRESHOLD = 0.75
    LOW_THRESHOLD = 0.45

    company: str | None = None
    year_code: int | None = None

    # Optional, explicit -- set this when only a span is known, e.g.
    #   class Pictorial1934(EnvelopeTemplate):
    #       year_range = (1932, 1936)
    # Purely informational: nothing here reads it for matching logic,
    # it's just a place to record what you actually know.
    year_range: tuple[int, int] | None = None

    # Fractional bounding boxes (0.0-1.0 of envelope width/height) per
    # field, and a validator per field -- both intentionally empty here.
    # Subclasses fill these in as DATA; the actual extraction mechanism
    # that reads them lives in extraction.py, not here.
    field_regions: dict[str, tuple[float, float, float, float]] = {}
    field_validators: dict[str, Callable[[str], bool]] = {}

    def __init__(self):
        if self.company is None or self.year_code is None:
            derived_company, derived_year = self._parse_classname()
            if self.company is None:
                self.company = derived_company
            if self.year_code is None:
                self.year_code = derived_year

        self._reference_fingerprint: EnvelopeFingerprint | None = None
        self._reference_path: Path | None = None

    def _parse_classname(self) -> tuple[str, int | None]:
        match = _CLASSNAME_PATTERN.match(type(self).__name__)
        if not match:
            return type(self).__name__, None
        return match.group(1), int(match.group(2))

    def set_reference(self, reference_image: np.ndarray) -> None:
        """Computes and stores this template's reference fingerprint
        immediately from an in-memory image. Use this for tests/synthetic
        data; for real files use set_reference_path() instead, which
        defers loading until actually needed.
        """
        self._reference_fingerprint = self.fingerprint(reference_image)
        self._reference_path = None

    def set_reference_path(self, path: str | Path) -> None:
        """Points this template at a reference image file WITHOUT loading
        it yet. The file doesn't need to exist at the time you call this
        -- it's only read (and the fingerprint computed + cached) the
        first time `reference_fingerprint` is actually accessed, e.g.
        during classification. This lets you define and commit a template
        class before you've placed the real image on disk.
        """
        self._reference_path = Path(path)
        self._reference_fingerprint = None  # invalidate any previous cache

    @property
    def reference_fingerprint(self) -> EnvelopeFingerprint | None:
        """None if no reference has been set at all yet (via either
        method above). Raises FileNotFoundError -- not silently returns
        None -- if a path WAS set but the file isn't there when actually
        needed, so a missing image is loud and specific rather than
        quietly scoring 0 like a template with no reference configured
        at all would.
        """
        if self._reference_fingerprint is not None:
            return self._reference_fingerprint

        if self._reference_path is not None:
            if not self._reference_path.exists():
                raise FileNotFoundError(
                    f"{type(self).__name__} reference image not found at "
                    f"{self._reference_path} -- place the file there, or "
                    f"call set_reference_path() with the correct location."
                )
            image = cv2.imread(str(self._reference_path))
            if image is None:
                raise ValueError(
                    f"{type(self).__name__} reference image at "
                    f"{self._reference_path} could not be read (corrupt "
                    f"file or unsupported format?)."
                )
            self._reference_fingerprint = self.fingerprint(image)
            return self._reference_fingerprint

        return None

    def fingerprint(self, envelope: np.ndarray) -> EnvelopeFingerprint:
        """Default: the generic color/edge/text-layout fingerprint.
        Override if a particular template needs different logic (e.g.
        cropping to a specific region before fingerprinting).
        """
        return EnvelopeFingerprint.compute(envelope)

    def extract_fields(self, envelope: np.ndarray) -> dict[str, FieldResult]:
        """Delegates to the shared FieldExtractor singleton (see
        extraction.py) -- this template only supplies field_regions/
        field_validators as data. Override this method instead if a
        specific template needs a genuinely different extraction
        mechanism (e.g. a dedicated FieldExtractor with different OCR
        settings) rather than just different field data.
        """
        return extractor.extract(self, envelope)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(company={self.company!r}, "
            f"year_code={self.year_code!r})"
        )
