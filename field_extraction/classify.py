"""Auto-classification: figure out which template (company + side)
best matches an uploaded photo, instead of asking the person to pick
one from a dropdown.

Two stages, run in order:

1. Company detection via keyword match. For any template with a
   "company_name" field, OCR just that field and check (fuzzily, to
   tolerate OCR noise) whether the expected company keyword appears in
   it. If exactly one company's keyword is found with reasonable
   confidence, candidates narrow to that company's templates only.

2. Side (front/back) via mean field confidence, among whatever
   candidates are left after stage 1 (either one company's two
   templates, or all of them if stage 1 found nothing conclusive).

Why two stages, not confidence alone: tried confidence-only first and
it mis-classified an Excella front photo as Pictorial/back -- these
envelopes are covered in dense printed text almost everywhere, so even
a WRONG template's bboxes usually land on SOME real, legible text and
score plausible confidence. A wrong template scored within 1 point of
the right one in testing. Company-name keyword matching is a much
stronger signal where it's available (checks WHAT the text says, not
just how legible it looks); confidence-based side detection is fine
once mis-classification-across-companies is ruled out first, since
front-vs-back within the SAME company is a much less ambiguous
comparison (confirmed reliable in testing once narrowed).

This is NOT the production package's logo/fingerprint classifier
(src/envelope_mappings/classifier.py) -- that does real image-based
company-logo keypoint matching (ORB) plus a weighted color/edge/layout
fingerprint, a fundamentally more robust approach than text-keyword
matching. It currently can't even be imported: extraction.py there
imports FieldResult from results.py, which doesn't define it -- a
pre-existing bug, unrelated to anything in this folder (see
field_extraction/README.md). This module is a lighter stand-in that
works with what's already built here; worth revisiting with the real
classifier once that import bug is fixed, or sooner if this stops
being reliable as more companies are added.
"""

from __future__ import annotations

import difflib
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from extractor import (  # noqa: E402
    crop_fractional_region,
    extract_field_map,
    ocr_region,
)
from field_regions import EnvelopeFieldMap, list_templates  # noqa: E402

# Keyword expected in each company's "company_name" field, if that
# template has one. Add an entry here for any new company that has a
# company_name (or similarly-named) field defined in field_regions.py.
COMPANY_KEYWORDS = {
    "Excella": "EXCELLA",
    "Pictorial": "PICTORIAL",
}

# Minimum OCR confidence AND fuzzy-match ratio for stage 1 to trust a
# company-name match. Both need to clear the bar -- a confident read of
# clearly-wrong text, or a plausible-looking match at low confidence,
# are each individually common enough to be worth guarding against.
_MIN_KEYWORD_OCR_CONFIDENCE = 50
_MIN_KEYWORD_MATCH_RATIO = 0.6


def _fuzzy_contains(text: str, keyword: str, threshold: float) -> bool:
    """True if any substring of text (the same length as keyword) is at
    least `threshold` similar to keyword, case-insensitive. Tolerates
    the kind of single-character OCR noise seen in testing (e.g.
    "FXCELLA" for "EXCELLA") without requiring an exact substring.
    """
    text = text.upper()
    keyword = keyword.upper()
    n = len(keyword)
    if n == 0 or len(text) < 1:
        return False
    best_ratio = 0.0
    for i in range(max(1, len(text) - n + 1)):
        window = text[i : i + n]
        ratio = difflib.SequenceMatcher(None, window, keyword).ratio()
        best_ratio = max(best_ratio, ratio)
    return best_ratio >= threshold


def _detect_company_by_keyword(
    image: np.ndarray, candidates: list[tuple[str, str, str, EnvelopeFieldMap]]
) -> str | None:
    """Stage 1. Returns a company name if exactly one is confidently
    matched via its company_name field, else None (ambiguous, no
    company_name field among the candidates, or no confident match --
    all treated the same: fall through to stage 2 with every
    candidate).
    """
    matched_companies: set[str] = set()
    for company, _side, _pattern_number, field_map in candidates:
        region = field_map.get("company_name")
        if region is None:
            continue
        keyword = COMPANY_KEYWORDS.get(company)
        if keyword is None:
            continue
        crop = crop_fractional_region(image, region.bbox)
        value, confidence = ocr_region(crop, "company_name")
        if (
            confidence is not None
            and confidence >= _MIN_KEYWORD_OCR_CONFIDENCE
            and _fuzzy_contains(value, keyword, _MIN_KEYWORD_MATCH_RATIO)
        ):
            matched_companies.add(company)

    if len(matched_companies) == 1:
        return next(iter(matched_companies))
    return None  # zero or multiple matches -- not conclusive either way


def auto_classify(
    image: np.ndarray,
) -> tuple[str, str, EnvelopeFieldMap, dict[str, Any], float]:
    """Tries known templates against image (which should already be
    rotated upright -- this does not detect rotation itself, see
    extractor.apply_detected_rotation). Returns the best match as
    (company, side, field_map, results, confidence), where confidence
    is that template's mean per-field OCR confidence (0-100).

    Raises ValueError if ENVELOPE_FIELD_MAPS is empty -- nothing to
    classify against.
    """
    candidates = list_templates()
    if not candidates:
        raise ValueError("No templates defined in ENVELOPE_FIELD_MAPS.")

    detected_company = _detect_company_by_keyword(image, candidates)
    if detected_company is not None:
        candidates = [c for c in candidates if c[0] == detected_company]

    best: tuple[str, str, EnvelopeFieldMap, dict[str, Any], float] | None = None
    for company, side, _pattern_number, field_map in candidates:
        results = extract_field_map(image, field_map)
        confidences = [
            r["confidence"] for r in results.values() if r["confidence"] is not None
        ]
        score = sum(confidences) / len(confidences) if confidences else 0.0
        if best is None or score > best[4]:
            best = (company, side, field_map, results, score)

    assert best is not None  # candidates was non-empty going in, so this always runs
    return best
