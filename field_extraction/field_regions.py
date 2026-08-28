"""Field region definitions -- DATA ONLY, no extraction logic here.

Deliberately kept separate from extractor.py, same reasoning as the
production package's split between EnvelopeTemplate.field_regions (data)
and FieldExtractor (mechanism, in src/envelope_mappings/extraction.py):
so bboxes can be measured, tuned, and re-tested against real photos
without touching how extraction actually works.

This is a DEV/TEST harness, not the production field_regions. Once a
company's regions are confirmed good via run_test.py, copy the
confirmed tuples into the real template's FIELD_REGIONS dict in
src/envelope_mappings/templates/<company>.py -- this module doesn't
feed into the production package at all.

Every bbox is (x1, y1, x2, y2) as fractions of image width/height
(0.0-1.0) -- same convention as EnvelopeTemplate.field_regions. Measured
by hand against the real reference photos using a fractional grid
overlay; not guaranteed to transfer to a different photo of the same
envelope taken at a different crop/zoom.

ROTATION: rotation is now auto-detected per-photo at runtime (see
extractor.detect_rotation), not assumed from stored data. The
requires_rotation field below is kept as informational/historical
context only -- what was true for these specific reference photos when
measured -- but run_test.py no longer reads it to decide anything.
Found via testing that rotation genuinely varies photo to photo even
within "the same side of the same company": Excella's back photo
needed no rotation, Pictorial's back photo needed 90 degrees.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FieldRegion:
    name: str
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2) fractions
    note: str = ""
    # Clockwise degrees to rotate THIS crop before OCR, independent of
    # the whole-page rotation already applied. Not currently used by
    # either Excella or Pictorial -- both print every field right-side
    # up relative to the rest of their own page -- but kept as a field
    # on FieldRegion since it's cheap to support and some envelopes
    # print one text block sideways relative to everything else (e.g.
    # a pattern-number/size block running along a spine margin). 0
    # (the default) means no extra rotation.
    local_rotation: int = 0


@dataclass(frozen=True)
class EnvelopeFieldMap:
    """All field regions measured for one photographed side (front or
    back) of one reference envelope. Pure data plus a trivial lookup --
    no extraction behavior belongs on this class.
    """

    company: str
    pattern_number: str  # which reference envelope this was measured against
    side: str  # "front" or "back"
    requires_rotation: bool  # historical only -- see module docstring; not read at runtime
    regions: tuple[FieldRegion, ...]

    def get(self, field_name: str) -> FieldRegion | None:
        for region in self.regions:
            if region.name == field_name:
                return region
        return None


EXCELLA_E3415_FRONT = EnvelopeFieldMap(
    company="Excella",
    pattern_number="E3415",
    side="front",
    requires_rotation=True,
    regions=(
        FieldRegion("pattern_number", (0.25, 0.195, 0.40, 0.228)),
        FieldRegion("price", (0.39, 0.195, 0.555, 0.228)),
        FieldRegion("size", (0.55, 0.195, 0.715, 0.228)),
        FieldRegion("garment_description", (0.19, 0.495, 0.50, 0.585)),
        FieldRegion("to_make_instructions", (0.44, 0.235, 0.72, 0.585)),
        FieldRegion("material_required_table", (0.19, 0.58, 0.80, 0.67)),
        FieldRegion(
            "corresponding_measurements_table", (0.19, 0.685, 0.80, 0.735)
        ),
        FieldRegion("company_name", (0.19, 0.735, 0.80, 0.775)),
    ),
)

EXCELLA_E3415_BACK = EnvelopeFieldMap(
    company="Excella",
    pattern_number="E3415",
    side="back",
    requires_rotation=False,
    regions=(
        FieldRegion("general_instructions", (0.19, 0.365, 0.42, 0.565)),
        FieldRegion("piece_list", (0.395, 0.44, 0.55, 0.565)),
        FieldRegion("seam_allowance_note", (0.55, 0.735, 0.82, 0.758)),
        FieldRegion(
            "cutting_layout_label",
            (0.19, 0.565, 0.57, 0.605),
            note=(
                "right edge may still clip 'BUST'/'VIEW 1' -- check "
                "run_test.py output and widen if needed"
            ),
        ),
    ),
)

PICTORIAL_7117_FRONT = EnvelopeFieldMap(
    company="Pictorial",
    pattern_number="7117",
    side="front",
    requires_rotation=True,
    regions=(
        FieldRegion("pattern_number", (0.195, 0.205, 0.30, 0.235)),
        FieldRegion("size_bust_hip_table", (0.30, 0.18, 0.51, 0.245)),
        FieldRegion("price", (0.545, 0.205, 0.65, 0.235)),
        FieldRegion("materials_suitable", (0.16, 0.708, 0.68, 0.752)),
        FieldRegion("company_name", (0.16, 0.76, 0.68, 0.787)),
    ),
)

PICTORIAL_7117_BACK = EnvelopeFieldMap(
    company="Pictorial",
    pattern_number="7117",
    side="back",
    # Unlike Excella's back (shot already upright), this envelope's back
    # photo came out of the camera sideways too -- rotation need isn't a
    # fixed per-company constant, it depends on how that particular
    # photo was framed. Always check a fresh photo before assuming.
    requires_rotation=True,
    regions=(
        FieldRegion("material_required_table", (0.155, 0.213, 0.60, 0.352)),
        FieldRegion("garment_description", (0.155, 0.497, 0.60, 0.567)),
        FieldRegion("notion_guide", (0.155, 0.565, 0.60, 0.628)),
        FieldRegion("piece_count_label", (0.19, 0.64, 0.40, 0.663)),
    ),
)

# Keyed by (company, pattern_number, side) rather than just
# (company, side) -- a single company can have multiple reference
# envelopes with genuinely different layouts (different pattern
# numbers, different eras), so company+side alone isn't a safe unique
# key. Keeping the three-part key now avoids a silent-overwrite bug
# later, even though both companies below currently have just one
# reference envelope each.
ENVELOPE_FIELD_MAPS: dict[tuple[str, str, str], EnvelopeFieldMap] = {
    ("Excella", "E3415", "front"): EXCELLA_E3415_FRONT,
    ("Excella", "E3415", "back"): EXCELLA_E3415_BACK,
    ("Pictorial", "7117", "front"): PICTORIAL_7117_FRONT,
    ("Pictorial", "7117", "back"): PICTORIAL_7117_BACK,
}


def find_field_map(
    company: str, pattern_number: str, side: str
) -> tuple[EnvelopeFieldMap | None, bool]:
    """Looks up a field map for (company, pattern_number, side).

    Tries an exact match first. If that fails, falls back to ANY field
    map for the same (company, side) regardless of pattern_number --
    the whole point of measuring bboxes against one reference envelope
    (e.g. Excella E3415) is that the SAME visual template should apply
    to other envelopes from that company with a similar layout, even
    though their actual pattern number is different (e.g. an Excella
    E5000 photo). Requiring an exact pattern_number match would defeat
    that -- every new envelope would need its own from-scratch
    measurement even when an existing template already fits.

    Returns (field_map, was_exact_match). field_map is None if nothing
    matches even by company+side. was_exact_match tells the caller
    whether the returned map was actually measured against this exact
    pattern_number, or is a same-company/side template being reused for
    a different envelope -- worth surfacing to whoever's reading
    results, since a reused template's bboxes aren't guaranteed to line
    up as tightly on a photo they weren't measured against (different
    printing layout across years, a slightly different photo crop,
    etc).

    If more than one template exists for the same (company, side) --
    not the case yet for anything in ENVELOPE_FIELD_MAPS, but plausible
    once a company has multiple genuinely different-era layouts, the
    way McCall's 6600 and 8306 did in earlier testing -- this returns
    whichever one happens to be found first. No attempt is made to
    guess which template fits a new photo best; if that distinction
    ever matters, it needs its own explicit selection, not a silent
    guess here.
    """
    exact = ENVELOPE_FIELD_MAPS.get((company, pattern_number, side))
    if exact is not None:
        return exact, True

    for (map_company, _map_pattern, map_side), field_map in ENVELOPE_FIELD_MAPS.items():
        if map_company == company and map_side == side:
            return field_map, False

    return None, False


def list_templates() -> list[tuple[str, str, str, EnvelopeFieldMap]]:
    """Every unique (company, side) template available, each paired
    with which pattern_number's regions it was measured from and the
    EnvelopeFieldMap itself. Used by classify.py to try each template
    against a photo and see which one actually fits (see that module's
    docstring), instead of asking the person to pick one from a
    dropdown.

    If ENVELOPE_FIELD_MAPS ever has more than one pattern_number for
    the same (company, side), only the first one encountered is
    included here -- same simplification as find_field_map's fallback,
    for the same reason (no case requiring the distinction exists yet).
    """
    seen: dict[tuple[str, str], tuple[str, EnvelopeFieldMap]] = {}
    for (company, pattern_number, side), field_map in ENVELOPE_FIELD_MAPS.items():
        seen.setdefault((company, side), (pattern_number, field_map))
    return [
        (company, side, pattern_number, field_map)
        for (company, side), (pattern_number, field_map) in seen.items()
    ]
