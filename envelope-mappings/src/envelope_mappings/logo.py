"""Company logo matching via ORB keypoints -- Stage 1 classification.

Logos are a company-level signal (stable across year-variants within a
company), so this is deliberately separate from EnvelopeFingerprint,
which handles year-variant disambiguation WITHIN a company (Stage 2).

Same library as the digitization pipeline's tracing work (opencv's ORB),
no new dependency.
"""

from __future__ import annotations

import cv2
import numpy as np

MIN_GOOD_MATCHES_FOR_SCORE = 4  # below this, treat score as 0 rather than noisy


class CompanyLogo:
    """One company's reference logo, ready to score candidate envelopes
    against. Add one of these per company to the classifier's `logos`
    list -- see EnvelopeTemplate for the matching "extend by appending"
    pattern.
    """

    def __init__(self, company: str, reference_image: np.ndarray):
        self.company = company
        self._orb = cv2.ORB_create()
        self.keypoints, self.descriptors = self._orb.detectAndCompute(
            reference_image, None
        )

    def match_score(self, envelope_logo_region: np.ndarray) -> float:
        """Returns a similarity score in [0, 1]. 0 if either image has
        too few keypoints to compare meaningfully (blank/near-blank
        region, or a logo reference that didn't produce useful features).
        """
        if self.descriptors is None:
            return 0.0

        kp2, des2 = self._orb.detectAndCompute(envelope_logo_region, None)
        if des2 is None or len(kp2) < MIN_GOOD_MATCHES_FOR_SCORE:
            return 0.0

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(self.descriptors, des2)
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
