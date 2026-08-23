"""EnvelopeClassifier -- runs the two-stage classification.

Stage 1: cheap logo-only ORB match narrows to a company.
Stage 2: full fingerprint match, only within that company's templates,
         picks (or fails to pick) a specific year-variant.

Extending coverage is purely additive -- append to `logos` and
`templates`, nothing here changes:

    logos.append(CompanyLogo("Vogue", vogue_logo_ref))
    templates.append(Vogue1970())
    classifier = EnvelopeClassifier(logos, templates)
"""

from __future__ import annotations

import numpy as np

from envelope_mappings.logo import CompanyLogo
from envelope_mappings.results import AmbiguousMatch, NewTemplateNeeded, PatternRecord
from envelope_mappings.template import EnvelopeTemplate

LOGO_MIN_THRESHOLD = 0.5


class EnvelopeClassifier:
    def __init__(self, logos: list[CompanyLogo], templates: list[EnvelopeTemplate]):
        self.logos = logos
        self.templates = templates

    def classify(
        self, envelope: np.ndarray, logo_region: np.ndarray | None = None
    ) -> PatternRecord | AmbiguousMatch | NewTemplateNeeded:
        """`logo_region` defaults to the full envelope image if not given
        separately -- in practice you'll usually want to pass a cropped
        logo-area region for a cleaner Stage 1 match, but nothing here
        requires it.
        """
        if logo_region is None:
            logo_region = envelope

        # --- Stage 1: identify company via logo ---
        if not self.logos:
            return NewTemplateNeeded(envelope, stage="company", closest_scores=None)

        company_scores = sorted(
            ((logo.company, logo.match_score(logo_region)) for logo in self.logos),
            key=lambda pair: pair[1],
            reverse=True,
        )
        best_company, logo_score = company_scores[0]

        if logo_score < LOGO_MIN_THRESHOLD:
            return NewTemplateNeeded(
                envelope, stage="company", closest_scores=company_scores[:3]
            )

        # --- Stage 2: identify year-variant via fingerprint, within company ---
        candidates = [t for t in self.templates if t.company == best_company]
        if not candidates:
            # Logo matched a company with no templates registered for it yet
            return NewTemplateNeeded(envelope, stage="template", closest_scores=None)

        scores = []
        for template in candidates:
            if template.reference_fingerprint is None:
                # Template exists but has no reference image set yet --
                # can't match against it. Scored 0 rather than raising,
                # so one half-finished template doesn't break
                # classification of everything else in its company.
                scores.append((template, 0.0))
                continue
            # Uses THIS template's own fingerprint() method on the
            # envelope, not a shared one -- if a subclass overrides
            # fingerprint() (e.g. to crop to a specific region first),
            # both sides of the comparison need to go through that same
            # logic for the distance to mean anything.
            envelope_fp = template.fingerprint(envelope)
            scores.append(
                (template, template.reference_fingerprint.distance(envelope_fp))
            )
        scores.sort(key=lambda pair: pair[1], reverse=True)
        best_template, best_score = scores[0]

        if best_score >= EnvelopeTemplate.HIGH_THRESHOLD:
            fields = best_template.extract_fields(envelope)
            return PatternRecord(
                envelope=envelope,
                template_name=type(best_template).__name__,
                company=best_template.company,
                year_code=best_template.year_code,
                confidence=best_score,
                fields=fields,
            )
        elif best_score >= EnvelopeTemplate.LOW_THRESHOLD:
            return AmbiguousMatch(
                envelope=envelope, company=best_company, candidates=scores[:3]
            )
        else:
            return NewTemplateNeeded(
                envelope,
                stage="template",
                closest_scores=[(type(t).__name__, s) for t, s in scores[:3]],
            )
