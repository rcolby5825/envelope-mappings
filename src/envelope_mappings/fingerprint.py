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
        match other"). Combines the three components with equal weight;
        TODO once there's real data to tune against, these weights may
        want adjusting -- e.g. text_layout may deserve more weight than
        color if two companies happen to share a palette.

        IMPORTANT, found via testing on two real, genuinely different
        envelopes: comparing edge_map/text_layout via mean-absolute-
        difference scored them as ~0.80-0.93 similar even though they're
        different companies entirely. Both maps are mostly blank (sparse
        content on a mostly-empty page), so elementwise mean-abs-diff is
        dominated by both images trivially "agreeing" on blank regions,
        drowning out the real signal in the sparse content that actually
        differs. Switched to correlation (same approach already used for
        color_hist) instead -- confirmed on the same real pair that this
        correctly drops the cross-company score to ~0.25 while a self-
        match still scores a perfect 1.0.
        """
        color_sim = cv2.compareHist(
            self.color_hist, other.color_hist, cv2.HISTCMP_CORREL
        )
        color_sim = max(0.0, color_sim)  # correlation can go negative

        edge_sim = _correlation_sim(self.edge_map, other.edge_map)
        layout_sim = _correlation_sim(self.text_layout, other.text_layout)

        return float((color_sim + edge_sim + layout_sim) / 3.0)


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
