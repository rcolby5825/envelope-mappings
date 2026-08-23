"""Envelope mappings package.

Two-stage classification of sewing pattern envelopes: a cheap logo match
narrows to a company, then a combined visual fingerprint (color, edge/
layout, text-block positions) narrows to a specific year-variant
template, which then extracts fields from its own known regions.

See classifier.py, template.py, logo.py, fingerprint.py, results.py.
"""

from envelope_mappings.classifier import EnvelopeClassifier
from envelope_mappings.fingerprint import EnvelopeFingerprint
from envelope_mappings.logo import CompanyLogo
from envelope_mappings.results import AmbiguousMatch, NewTemplateNeeded, PatternRecord
from envelope_mappings.template import EnvelopeTemplate

__version__ = "0.1.0"

__all__ = [
    "EnvelopeClassifier",
    "EnvelopeTemplate",
    "EnvelopeFingerprint",
    "CompanyLogo",
    "PatternRecord",
    "AmbiguousMatch",
    "NewTemplateNeeded",
]
