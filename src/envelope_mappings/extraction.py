"""FieldExtractor -- the extraction mechanism, pulled out of
EnvelopeTemplate into its own dedicated place.

Deliberately separate from EnvelopeTemplate for the same reason
CompanyLogo and EnvelopeFingerprint are their own modules: extraction is
a distinct concern (crop a region, OCR it, validate it) with no
per-template state of its own -- it only needs field_regions/
field_validators as INPUT, it doesn't own them. Keeping it standalone
means the extraction data structure (FieldResult) and mechanism (OCR
settings, confidence filtering, crop math) can be amended in ONE place
without touching template.py at all.

USAGE:
    A single shared instance (`extractor`, at the bottom of this module)
    is the one every template uses by default -- there's no per-template
    state to justify separate instances, so this follows the standard
    Python singleton pattern of one module-level instance rather than
    enforcing it via __new__ (which is usually more machinery than
    needed for something with no state to protect).

    EnvelopeTemplate.extract_fields() delegates to this singleton:

        from envelope_mappings.extraction import extractor

        class SomeTemplate(EnvelopeTemplate):
            field_regions = {...}
            field_validators = {...}
            # extract_fields() inherited from EnvelopeTemplate, which
            # just calls extractor.extract(self, envelope)

    You can also call it directly, or swap in your own FieldExtractor
    instance/subclass for a specific template if it ever needs different
    OCR settings than everything else:

        from envelope_mappings.extraction import FieldExtractor

        class CustomExtractor(FieldExtractor):
            MIN_FIELD_OCR_CONFIDENCE = 50  # stricter, for a noisy template

        class SomeTemplate(EnvelopeTemplate):
            def extract_fields(self, envelope):
                return CustomExtractor().extract(self, envelope)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from envelope_mappings.results import FieldResult

if TYPE_CHECKING:
    from envelope_mappings.template import EnvelopeTemplate


class FieldExtractor:
    # Below this OCR confidence, treat as noise (e.g. stray marks) rather
    # than a real detection -- found necessary via testing: without this,
    # a genuinely blank region leaked a spurious low-confidence token
    # instead of a clean empty result.
    MIN_FIELD_OCR_CONFIDENCE = 30

    # PSM 7 = "treat the image as a single text line" -- the right mode
    # for a tight crop around one short field (pattern number, size),
    # as distinct from whole-PAGE OCR (see fingerprint.py's
    # _compute_text_layout, which needs PSM 6/11 instead because it's
    # working with a whole illustrated page, not a pre-cropped field).
    # Confirmed via testing that PSM 7 correctly reads a multi-word
    # single-line field cleanly.
    OCR_PSM_MODE = 7

    def extract(
        self, template: "EnvelopeTemplate", envelope: np.ndarray
    ) -> dict[str, FieldResult]:
        """Crops each region in template.field_regions, OCRs it,
        validates against template.field_validators if a validator is
        registered for that field, and returns one FieldResult per
        field.

        Returns an empty dict if field_regions is empty (a template
        that's matching correctly but hasn't had its fields defined
        yet) -- NOT an error, since classification and extraction are
        deliberately separate concerns; a template with no fields
        defined can still be a valid, confident MATCH.
        """
        results: dict[str, FieldResult] = {}
        for field_name, bbox in template.field_regions.items():
            crop = self._crop_fractional_region(envelope, bbox)
            value, confidence = self._ocr_field(crop)

            validator = template.field_validators.get(field_name)
            valid = validator(value) if validator else None

            results[field_name] = FieldResult(
                value=value, valid=valid, confidence=confidence
            )
        return results

    @staticmethod
    def _crop_fractional_region(
        envelope: np.ndarray, bbox: tuple[float, float, float, float]
    ) -> np.ndarray:
        """bbox is (x1, y1, x2, y2) as fractions of width/height (0.0-1.0),
        e.g. (0.1, 0.05, 0.6, 0.15) for a region starting 10% in from the
        left, 5% down, ending 60% across and 15% down.
        """
        h, w = envelope.shape[:2]
        x1, y1, x2, y2 = bbox
        px1, py1 = int(x1 * w), int(y1 * h)
        px2, py2 = int(x2 * w), int(y2 * h)
        return envelope[py1:py2, px1:px2]

    def _ocr_field(self, crop: np.ndarray) -> tuple[str, float | None]:
        """Returns (cleaned_text, average_confidence) -- confidence is
        None if nothing above MIN_FIELD_OCR_CONFIDENCE was detected in
        the crop (a genuinely blank region, or only stray-mark noise).
        """
        import pytesseract

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        data = pytesseract.image_to_data(
            gray, config=f"--psm {self.OCR_PSM_MODE}",
            output_type=pytesseract.Output.DICT,
        )

        words = []
        confs = []
        for i in range(len(data["text"])):
            token = data["text"][i].strip()
            if not token:
                continue
            conf = int(data["conf"][i]) if data["conf"][i] != "-1" else -1
            if conf < self.MIN_FIELD_OCR_CONFIDENCE:
                continue
            words.append(token)
            confs.append(conf)

        value = " ".join(words)
        confidence = float(np.mean(confs)) if confs else None
        return value, confidence


# The shared singleton -- see module docstring. Everything that just
# needs "the standard extraction behavior" should use this rather than
# constructing its own FieldExtractor.
extractor = FieldExtractor()
