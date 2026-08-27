"""Extraction mechanism for the field_extraction dev/test harness --
crop + OCR only. No field data lives here; see field_regions.py.

Deliberately separate from src/envelope_mappings/extraction.py's
FieldExtractor. That one is the production singleton, driven by
EnvelopeTemplate.field_regions (dict[str, tuple]). This one is a
standalone harness driven by field_regions.py's EnvelopeFieldMap
objects, so bboxes and OCR settings can be iterated on against real
photos before anything gets copied into the real templates.

Same OCR approach as the production extractor for consistency:
- PSM 7 ("single text line") for short single-line fields.
- PSM 6 ("uniform block of text") for multi-line fields -- PSM 7 would
  garble a paragraph or table by assuming one line, which is why this
  harness branches on MULTILINE_FIELDS below rather than using one PSM
  for everything.
- MIN_FIELD_OCR_CONFIDENCE = 30, matching the production extractor's
  fix for a blank region leaking a spurious low-confidence token.
"""

from __future__ import annotations

import re

import cv2
import numpy as np

MIN_FIELD_OCR_CONFIDENCE = 30

# Fields expected to span multiple lines -- these need PSM 6, not the
# PSM 7 single-line default. Extend this set as new multi-line fields
# get added to field_regions.py.
MULTILINE_FIELDS = {
    "garment_description",
    "to_make_instructions",
    "material_required_table",
    "corresponding_measurements_table",
    "general_instructions",
    "piece_list",
    "size_bust_hip_table",
    "materials_suitable",
    "notion_guide",
    "company_name",  # two lines on Excella: "EXCELLA PATTERNS" / "EXCELLA CORPORATION · NEW YORK"
}


def crop_fractional_region(
    image: np.ndarray, bbox: tuple[float, float, float, float]
) -> np.ndarray:
    """bbox is (x1, y1, x2, y2) as fractions of width/height (0.0-1.0)."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    px1, py1 = int(x1 * w), int(y1 * h)
    px2, py2 = int(x2 * w), int(y2 * h)
    return image[py1:py2, px1:px2]


_ROTATION_TO_CV2 = {
    0: None,
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def _bbox_by_saturation(image: np.ndarray) -> tuple[int, int, int, int]:
    """Isolates the envelope via HSV saturation -- works when the paper
    is visibly more colorful (tan/brown) than the gray photo background.
    Found to fail on paler/more-aged paper (see _bbox_by_edges).
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    _, mask = cv2.threshold(
        saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    largest = max(contours, key=cv2.contourArea)
    return cv2.boundingRect(largest)


def _bbox_by_edges(image: np.ndarray) -> tuple[int, int, int, int]:
    """Isolates the envelope via edge/text density instead of color --
    doesn't depend on the paper being saturated relative to the
    background, which _bbox_by_saturation needs. Found necessary for
    a pattern whose paper had aged to a pale cream close enough to the
    gray background that saturation-based Otsu split found only a tiny
    stray sliver instead of the envelope.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 30, 90)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 35))
    dilated = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    largest = max(contours, key=cv2.contourArea)
    return cv2.boundingRect(largest)


def detect_rotation(image: np.ndarray) -> int:
    """Returns the clockwise rotation in degrees (0/90/180/270) needed
    to make the photo upright. Auto-detected per-photo rather than
    trusting a stored per-company/per-side assumption -- found via
    testing that this genuinely varies photo to photo (Excella's back
    photo needed no rotation, Pictorial's back photo needed 90, despite
    both being "the back photo" of their respective envelopes).

    Two stages:

    1. Get an initial answer from tesseract's OSD (image_to_osd), tried
       against a sequence of candidate crops until one succeeds: the
       envelope isolated via saturation, the envelope isolated via edge
       density (needed when the paper is too pale for saturation to
       separate it from the background -- see _bbox_by_edges), each of
       those retried rotated 90 degrees if OSD raised on the first try,
       and finally the whole uncropped image as a last resort.

    2. Sanity-check that answer against its 180-degree opposite using
       _word_likeness_score, and flip to the opposite if it scores
       higher. Needed because OSD can return a confident-looking answer
       that's exactly 180 degrees wrong (picks the wrong text-baseline
       direction) rather than failing outright -- found in testing on
       a photo where OSD said 270 but the correct answer was 90. None
       of stage 1's retries catch this, since they only fire on
       exceptions, not on a wrong-but-confident result. Comparing
       actual recognizable-word counts between a
       rotation and its opposite catches it: the correct orientation
       reliably produces more alphabetic 3+ letter tokens than its
       upside-down twin, even when neither produces fully clean text.

    Falls back to 0 with a printed warning only if stage 1 fails
    completely (no candidate crop produced any OSD answer at all).
    """
    candidates: list[tuple[int, int, int, int]] = []
    for bbox_fn in (_bbox_by_saturation, _bbox_by_edges):
        try:
            x, y, w, h = bbox_fn(image)
            if w > 200 and h > 200:  # reject degenerate slivers
                candidates.append((x, y, w, h))
        except Exception:
            continue
    candidates.append((0, 0, image.shape[1], image.shape[0]))  # whole image

    for x, y, w, h in candidates:
        crop = image[y : y + h, x : x + w]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop

        degrees = _try_osd(gray)
        if degrees is None:
            rotated_gray = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
            retry = _try_osd(rotated_gray)
            degrees = None if retry is None else (retry + 90) % 360

        if degrees is not None:
            opposite = (degrees + 180) % 360
            if _word_likeness_score(gray, opposite) > _word_likeness_score(
                gray, degrees
            ):
                return opposite
            return degrees

    print(
        "  [rotation detection failed on every candidate crop, assuming "
        "0 -- check this photo manually, likely needs --force-rotation]"
    )
    return 0


_WORDLIKE_RE = re.compile(r"^[A-Za-z]{3,}$")


def _word_likeness_score(gray: np.ndarray, degrees: int) -> int:
    """Rotates gray by degrees, OCRs it (sparse-text mode), and counts
    tokens that look like real words (alphabetic, 3+ letters, decent
    confidence) rather than symbol/noise garbage. Used only to break a
    tie between a rotation and its 180-degree opposite -- see
    detect_rotation's docstring. Not meant as a general-purpose OCR
    quality metric; word-likeness alone is a coarse but cheap signal
    that was sufficient to fix the one case tested where it mattered.
    """
    import pytesseract

    cv2_constant = _ROTATION_TO_CV2[degrees % 360]
    rotated = gray if cv2_constant is None else cv2.rotate(gray, cv2_constant)
    _, binary = cv2.threshold(rotated, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    data = pytesseract.image_to_data(
        binary, config="--psm 11", output_type=pytesseract.Output.DICT
    )
    count = 0
    for text, conf in zip(data["text"], data["conf"]):
        if text.strip() and int(conf) >= 40 and _WORDLIKE_RE.match(text):
            count += 1
    return count


def _try_osd(gray: np.ndarray) -> int | None:
    """Runs tesseract OSD on an already-prepared grayscale crop. Returns
    the detected clockwise rotation in degrees, or None if OSD raised
    (couldn't find enough text to determine orientation).
    """
    import pytesseract

    try:
        osd = pytesseract.image_to_osd(gray, config="--psm 0")
    except Exception:
        return None

    for line in osd.splitlines():
        if line.startswith("Rotate:"):
            return int(line.split(":")[1].strip())
    return None


def apply_detected_rotation(image: np.ndarray) -> tuple[np.ndarray, int]:
    """Detects and applies rotation in one step. Returns (rotated_image,
    degrees_applied) -- the degrees are returned so callers can log/print
    what was detected rather than it happening silently.
    """
    degrees = detect_rotation(image)
    cv2_constant = _ROTATION_TO_CV2[degrees]
    if cv2_constant is None:
        return image, degrees
    return cv2.rotate(image, cv2_constant), degrees


def _prep_variants(gray: np.ndarray) -> dict[str, np.ndarray]:
    """Several binarization strategies, keyed by name, since no single
    one handles every field on these photos -- found via testing:

    - 'otsu': global Otsu threshold. Works for most normal dark-ink-on-
      paper crops, but found to FAIL CATASTROPHICALLY (flips the whole
      crop to solid black) if the crop includes even a sliver of the
      lighter background beyond the envelope's edge -- the global
      histogram splits between "that sliver" and "everything else"
      instead of between paper and ink.
    - 'adaptive': local (Gaussian-weighted) adaptive threshold. Robust
      to the background-sliver problem above and to lighting gradients
      across a large crop (found necessary for the multi-line tables),
      but performed worse than Otsu on the small bordered header
      fields in testing.
    - 'otsu_inv': Otsu, inverted. Needed for fields printed as light
      text on a dark filled block -- e.g. Excella's "25 Cents" price
      box is reversed (white-on-black), which reads as blank/garbage
      under either of the above.

    Every variant gets a white border added before returning -- found
    via testing on Pictorial's "7117" pattern number: tesseract
    performed noticeably worse (dropped digits, near-zero confidence)
    on a binarized crop with text touching the crop edges, and
    improved (though still imperfect -- see field_extraction/README.md)
    once given a margin.
    """
    otsu_thresh, otsu = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
    )
    _, otsu_inv = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    variants = {"otsu": otsu, "adaptive": adaptive, "otsu_inv": otsu_inv}
    return {
        name: cv2.copyMakeBorder(
            v, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255
        )
        for name, v in variants.items()
    }


def ocr_region(crop: np.ndarray, field_name: str) -> tuple[str, float | None]:
    """Returns (cleaned_text, average_confidence). Confidence is None if
    nothing above MIN_FIELD_OCR_CONFIDENCE was detected in ANY variant
    (blank region, or a crop tesseract genuinely can't read).

    Runs OCR against all of _prep_variants() and keeps whichever variant
    gives the highest mean confidence -- no single binarization strategy
    covered every field type found on the real photos (see
    _prep_variants' docstring), so this picks per-crop rather than
    committing to one globally.
    """
    import pytesseract

    psm = 6 if field_name in MULTILINE_FIELDS else 7
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop

    best_words: list[str] = []
    best_confs: list[int] = []
    best_mean = -1.0

    for variant in _prep_variants(gray).values():
        data = pytesseract.image_to_data(
            variant, config=f"--psm {psm}", output_type=pytesseract.Output.DICT
        )
        words: list[str] = []
        confs: list[int] = []
        for i in range(len(data["text"])):
            token = data["text"][i].strip()
            if not token:
                continue
            conf = int(data["conf"][i]) if data["conf"][i] != "-1" else -1
            if conf < MIN_FIELD_OCR_CONFIDENCE:
                continue
            words.append(token)
            confs.append(conf)

        mean_conf = float(np.mean(confs)) if confs else -1.0
        if mean_conf > best_mean:
            best_mean = mean_conf
            best_words = words
            best_confs = confs

    value = " ".join(best_words)
    confidence = float(np.mean(best_confs)) if best_confs else None
    return value, confidence


def extract_field_map(image: np.ndarray, field_map) -> dict[str, dict]:
    """Runs OCR over every FieldRegion in field_map.regions against
    image. Returns {field_name: {"value", "confidence", "bbox"}}.

    Does NOT correct whole-page rotation itself -- caller must pass an
    already-upright image (see apply_detected_rotation / run_test.py).
    DOES apply each region's own local_rotation, if set, to just that
    crop -- for text blocks that run sideways relative to the rest of
    an otherwise-upright page (see FieldRegion.local_rotation).
    """
    results: dict[str, dict] = {}
    for region in field_map.regions:
        crop = crop_fractional_region(image, region.bbox)
        if region.local_rotation:
            cv2_constant = _ROTATION_TO_CV2[region.local_rotation % 360]
            if cv2_constant is not None:
                crop = cv2.rotate(crop, cv2_constant)
        value, confidence = ocr_region(crop, region.name)
        results[region.name] = {
            "value": value,
            "confidence": confidence,
            "bbox": region.bbox,
        }
    return results
