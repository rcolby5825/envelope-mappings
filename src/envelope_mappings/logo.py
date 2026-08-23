"""Company logo matching via ORB keypoints -- Stage 1 classification.

Logos are a company-level signal (stable across year-variants within a
company), so this is deliberately separate from EnvelopeFingerprint,
which handles year-variant disambiguation WITHIN a company (Stage 2).

Same library as the digitization pipeline's tracing work (opencv's ORB),
no new dependency.

REFERENCE IMAGES ARE NOT PART OF THIS REPO -- same convention as
EnvelopeTemplate. Set a path via set_reference_path(); the file is only
read (and keypoints computed + cached) the first time it's actually
needed, so you can define a CompanyLogo and point it at a path before
the image exists there yet.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

MIN_GOOD_MATCHES_FOR_SCORE = 4  # below this, treat score as 0 rather than noisy


class CompanyLogo:
    """One company's reference logo, ready to score candidate envelopes
    against. Add one of these per company to the classifier's `logos`
    list -- see EnvelopeTemplate for the matching "extend by appending"
    pattern.
    """

    def __init__(self, company: str):
        self.company = company
        self._orb = cv2.ORB_create()
        self._keypoints = None
        self._descriptors = None
        self._reference_path: Path | None = None
        self._loaded = False

    def set_reference(self, reference_image: np.ndarray) -> None:
        """Computes and stores ORB keypoints immediately from an
        in-memory image. Use this for tests/synthetic data; for real
        files use set_reference_path() instead, which defers loading
        until actually needed.
        """
        self._keypoints, self._descriptors = self._orb.detectAndCompute(
            reference_image, None
        )
        self._reference_path = None
        self._loaded = True

    def set_reference_path(self, path: str | Path) -> None:
        """Points this logo at a reference image file WITHOUT loading it
        yet -- the file doesn't need to exist at call time, only when
        match_score() is first actually called.
        """
        self._reference_path = Path(path)
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if self._reference_path is None:
            raise ValueError(
                f"CompanyLogo({self.company!r}) has no reference set -- "
                f"call set_reference() or set_reference_path() first."
            )
        if not self._reference_path.exists():
            raise FileNotFoundError(
                f"CompanyLogo({self.company!r}) reference image not found "
                f"at {self._reference_path} -- place the file there, or "
                f"call set_reference_path() with the correct location."
            )
        image = cv2.imread(str(self._reference_path))
        if image is None:
            raise ValueError(
                f"CompanyLogo({self.company!r}) reference image at "
                f"{self._reference_path} could not be read (corrupt file "
                f"or unsupported format?)."
            )
        self._keypoints, self._descriptors = self._orb.detectAndCompute(image, None)
        self._loaded = True

    def match_score(self, envelope_logo_region: np.ndarray) -> float:
        """Returns a similarity score in [0, 1]. 0 if either image has
        too few keypoints to compare meaningfully (blank/near-blank
        region, or a logo reference that didn't produce useful features).

        Raises FileNotFoundError/ValueError if a reference path was set
        but the file isn't there/readable -- callers driving a batch of
        logos (e.g. EnvelopeClassifier) should catch these per-logo
        rather than let one incomplete company break the whole batch.
        """
        self._ensure_loaded()

        if self._descriptors is None:
            return 0.0

        kp2, des2 = self._orb.detectAndCompute(envelope_logo_region, None)
        if des2 is None or len(kp2) < MIN_GOOD_MATCHES_FOR_SCORE:
            return 0.0

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(self._descriptors, des2)
        if len(matches) < MIN_GOOD_MATCHES_FOR_SCORE:
            return 0.0

        matches = sorted(matches, key=lambda m: m.distance)
        # Score from the best matches' distances (lower distance = better
        # match; ORB/Hamming distances top out around 256, in practice
        # good matches are well under 64) -- TODO: this normalization is
        # a reasonable starting point, not yet tuned against real data.
        best = matches[: max(MIN_GOOD_MATCHES_FOR_SCORE, len(matches) // 4)]
        avg_distance = float(np.mean([m.distance for m in best]))
        score = 1.0 - min(avg_distance / 64.0, 1.0)
        return max(0.0, score)
