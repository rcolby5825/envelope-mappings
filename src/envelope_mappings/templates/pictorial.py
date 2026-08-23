"""Pictorial Printed Patterns (Pictorial Review Pattern Company, Inc.,
New York).

Validated against a real envelope (pattern #7117) during development:
the ORB logo match and combined fingerprint both correctly separated
this company from Excella Patterns on real photos -- see fingerprint.py
and logo.py docstrings for the specifics of what testing found and fixed.

YEAR: not printed on the envelope and not exactly known -- using a
placeholder anchor year for now (required for the classname parsing
convention, which needs strictly-digits-only). Update PLACEHOLDER_YEAR
below once you've pinned it down, or set year_range explicitly if only a
span is knowable. See EnvelopeTemplate's docstring for why year_code
(single anchor) and year_range (optional span) are kept separate.
"""

from envelope_mappings import CompanyLogo, EnvelopeTemplate

PLACEHOLDER_YEAR = 1900  # TODO: replace once the real year/era is known

# Fractional bounding boxes (0.0-1.0 of envelope width/height), left
# empty until you're ready to define exact field positions -- extraction
# isn't reachable until a template clears HIGH_THRESHOLD anyway, so this
# can be filled in incrementally without blocking classification testing.
FIELD_REGIONS: dict[str, tuple[float, float, float, float]] = {
    # "pattern_number": (x1, y1, x2, y2),
    # "size": (x1, y1, x2, y2),
}


class Pictorial(EnvelopeTemplate):
    """NOTE: renamed from the auto-derivable `Pictorial<year>` form since
    the real year isn't known yet -- company/year_code are set explicitly
    below instead of via classname parsing. Once a real year is known,
    consider renaming to e.g. `Pictorial1934` and removing the explicit
    overrides so it participates in the normal auto-derivation
    convention like other templates.
    """

    company = "Pictorial"
    year_code = PLACEHOLDER_YEAR
    field_regions = FIELD_REGIONS

    def extract_fields(self, envelope):
        # TODO: implement using field_regions once they're defined --
        # crop each region, OCR/validate per field_validators.
        raise NotImplementedError("Pictorial.extract_fields() not yet implemented")


def build_logo() -> CompanyLogo:
    """Call set_reference_path() on the result with wherever you keep
    the actual logo crop image, e.g.:

        logo = build_logo()
        logo.set_reference_path("/path/you/choose/pictorial_logo.jpg")
    """
    return CompanyLogo("Pictorial")


def build_template() -> Pictorial:
    """Same pattern as build_logo() -- call set_reference_path() on the
    result with wherever you keep the actual reference envelope photo.
    """
    return Pictorial()
