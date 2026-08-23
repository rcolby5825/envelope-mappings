"""Excella Patterns (Excella Corporation, New York).

Validated against a real envelope (pattern E3415) during development:
the ORB logo match and combined fingerprint both correctly separated
this company from Pictorial Printed Patterns on real photos -- see
fingerprint.py and logo.py docstrings for the specifics of what testing
found and fixed.

Structural note worth remembering when defining field_regions: unlike
Pictorial (front is illustration-only, instructions on the back),
Excella's front ALSO carries the "TO MAKE" instructions and materials
table alongside the illustration. Field positions won't transfer between
these two templates even loosely -- worth designing each independently.

YEAR: not printed on the envelope and not exactly known -- same
placeholder approach as pictorial.py, see that file for the full
rationale.
"""

from envelope_mappings import CompanyLogo, EnvelopeTemplate

PLACEHOLDER_YEAR = 1900  # TODO: replace once the real year/era is known

FIELD_REGIONS: dict[str, tuple[float, float, float, float]] = {
    # "pattern_number": (x1, y1, x2, y2),   # e.g. "E3415" near top
    # "size": (x1, y1, x2, y2),
}


class Excella(EnvelopeTemplate):
    """See Pictorial's docstring (pictorial.py) for why company/year_code
    are set explicitly here rather than via classname auto-derivation --
    same reasoning, real year not yet known.
    """

    company = "Excella"
    year_code = PLACEHOLDER_YEAR
    field_regions = FIELD_REGIONS

    def extract_fields(self, envelope):
        # TODO: implement using field_regions once they're defined.
        raise NotImplementedError("Excella.extract_fields() not yet implemented")


def build_logo() -> CompanyLogo:
    """Call set_reference_path() on the result with wherever you keep
    the actual logo crop image, e.g.:

        logo = build_logo()
        logo.set_reference_path("/path/you/choose/excella_logo.jpg")
    """
    return CompanyLogo("Excella")


def build_template() -> Excella:
    """Same pattern as build_logo() -- call set_reference_path() on the
    result with wherever you keep the actual reference envelope photo.
    """
    return Excella()
