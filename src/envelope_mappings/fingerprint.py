"""Combined visual fingerprint for envelope template matching.

Deliberately ONE combined signal rather than several independently
weighted detectors -- color, layout, illustration style, and font all
move together for a given company/era, so a single concatenated fingerprint
captures that gestalt "this looks the same" impression rather than trying
to hand-weight separate signals that are already correlated.

Built entirely from the same libraries as the digitization pipeline
(opencv, numpy, pytesseract) -- no new dependencies.

Components:
    - Color histogram (HSV, coarse bins)       -- the color-scheme signal
    - Downsampled edge/gradient map             -- overall layout AND
                                                    illustration-style
                                                    silhouette in one shot
    - Text block position/size layout           -- structural signature
                                                    from pytesseract boxes;
                                                    uses WHERE text sits,
                                                    not what it says, so
                                                    it's robust to OCR
                                                    misreads

Logo matching (ORB) is intentionally NOT part of this fingerprint -- see
logo.py. Logo identifies company (Stage 1); this fingerprint disambiguates
year-variant WITHIN a company (Stage 2).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

EDGE_MAP_SIZE = 32  # downsample target (square) for the edge/layout map
COLOR_HIST_BINS = (8, 8, 8)  # H, S, V bins


def _correlation_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two maps, clamped to [0, 1]. Used for
    edge_map/text_layout instead of raw elementwise difference -- see the
    note in EnvelopeFingerprint.distance() for why that matters.
    """
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)
    if a_flat.std() == 0 or b_flat.std() == 0:
        # No variation in one map (e.g. genuinely blank -- no edges, no
        # text found even after the PSM fallback) means there's no real
        # signal to correlate against. Treat as no-match rather than the
        # undefined/NaN result corrcoef would otherwise produce.
        return 0.0
    corr = np.corrcoef(a_flat, b_flat)[0, 1]
    return max(0.0, float(corr))


@dataclass
class EnvelopeFingerprint:
    color_hist: np.ndarray
    edge_map: np.ndarray
    text_layout: np.ndarray

    @classmethod
    def compute(cls, image_bgr: np.ndarray) -> "EnvelopeFingerprint":
        return cls(
            color_hist=_compute_color_hist(image_bgr),
            edge_map=_compute_edge_map(image_bgr),
            text_layout=_compute_text_layout(image_bgr),
        )

    def distance(self, other: "EnvelopeFingerprint") -> float:
        """Returns a similarity score in [0, 1] -- 1.0 = identical, not a
        true distance despite the name (kept for readability at call
        sites: `fingerprint.distance(other)` reads as "how well does this
        match other").

        WEIGHTS ARE NOT EQUAL, found via testing on real data in two
        stages:

        1. Comparing edge_map/text_layout via mean-absolute-difference
           scored two genuinely different companies as ~0.80-0.93
           similar (both maps are mostly blank, so elementwise diff is
           dominated by trivial agreement on blank regions). Fixed by
           switching to correlation, same approach already used for
           color_hist -- see _correlation_sim().

        2. Even with correlation, equal-weighting still had a real
           problem: a single JPEG re-save of the SAME envelope (an
           unavoidable, routine part of any real file-based workflow)
           dropped the combined score from 1.0 to 0.72 -- BELOW
           HIGH_THRESHOLD, meaning literally the same photo would fail
           to match itself after one lossy save. Root cause: layout_sim
           (built on OCR text-block detection) is far noisier than
           color/edge under small compression artifacts -- confirmed at
           0.20 similarity for the same envelope, vs. 0.97-0.999 for
           edge/color on the identical comparison. text_layout still
           carries real discriminating value between different companies
           (confirmed near 0.0 for two different real companies), so
           it's down-weighted rather than dropped -- just not trusted as
           much as the other two, more stable signals.

        These specific weights (0.45/0.45/0.10) are calibrated against
        exactly the two real envelopes used during development, not a
        larger validated sample -- revisit once more real templates
        exist to test against.
        """
        color_sim = cv2.compareHist(
            self.color_hist, other.color_hist, cv2.HISTCMP_CORREL
        )
        color_sim = max(0.0, color_sim)  # correlation can go negative

        edge_sim = _correlation_sim(self.edge_map, other.edge_map)
        layout_sim = _correlation_sim(self.text_layout, other.text_layout)

        return float(0.45 * color_sim + 0.45 * edge_sim + 0.10 * layout_sim)


def _compute_color_hist(image_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1, 2], None, list(COLOR_HIST_BINS),
        [0, 180, 0, 256, 0, 256],
    )
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist.flatten()


def _compute_edge_map(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    small = cv2.resize(edges, (EDGE_MAP_SIZE, EDGE_MAP_SIZE))
    return (small.astype(np.float32) / 255.0)


def _compute_text_layout(image_bgr: np.ndarray) -> np.ndarray:
    """Rasterizes text-block bounding boxes (position + size only, not
    content) onto a small grid -- a structural layout signature that
    doesn't depend on OCR actually reading the text correctly.

    IMPORTANT, found via testing on real envelope photos: tesseract's
    default page segmentation mode (PSM 3, "fully automatic") returned
    ZERO text detections on real envelope fronts -- large illustrations
    dominating the page apparently confuse its automatic layout analysis
    entirely, even though a tight crop of just the text worked fine.
    Confirmed reproducible on two different real envelopes (726 and 90
    tokens respectively) once switched to PSM 6 ("assume a single
    uniform block of text"). Without this fix, text_layout silently
    returns an all-zero grid on every real photo, and two all-zero grids
    trivially "match" each other perfectly -- which would have inflated
    the overall fingerprint's match confidence between genuinely
    different envelopes, not just failed loudly. PSM 11 (sparse text) is
    tried as a fallback if PSM 6 finds nothing, for layouts where even
    less of the page is text-like.
    """
    try:
        import pytesseract
    except ImportError:
        # text layout is one of three signals -- degrade gracefully to a
        # blank layer rather than hard-failing the whole fingerprint if
        # pytesseract/tesseract isn't installed in this environment.
        return np.zeros((EDGE_MAP_SIZE, EDGE_MAP_SIZE), dtype=np.float32)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    grid = np.zeros((EDGE_MAP_SIZE, EDGE_MAP_SIZE), dtype=np.float32)

    data = pytesseract.image_to_data(
        gray, config="--psm 6", output_type=pytesseract.Output.DICT
    )
    if not any(t.strip() for t in data["text"]):
        data = pytesseract.image_to_data(
            gray, config="--psm 11", output_type=pytesseract.Output.DICT
        )

    for i in range(len(data["text"])):
        if not data["text"][i].strip():
            continue
        conf = int(data["conf"][i]) if data["conf"][i] != "-1" else -1
        if conf < 30:
            continue
        x = data["left"][i]
        y = data["top"][i]
        bw = data["width"][i]
        bh = data["height"][i]
        gx1, gy1 = int(x / w * EDGE_MAP_SIZE), int(y / h * EDGE_MAP_SIZE)
        gx2 = min(EDGE_MAP_SIZE, int((x + bw) / w * EDGE_MAP_SIZE) + 1)
        gy2 = min(EDGE_MAP_SIZE, int((y + bh) / h * EDGE_MAP_SIZE) + 1)
        grid[gy1:gy2, gx1:gx2] = 1.0

    return grid
