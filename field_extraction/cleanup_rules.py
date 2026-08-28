"""Regex cleanup rules -- DATA ONLY, no logic here. Add to this file to
clean up recurring OCR mistakes without touching how extraction or
cleanup actually works (same data/mechanism split as field_regions.py
vs extractor.py).

Add a pattern here once you notice OCR making the SAME mistake more
than once -- e.g. if "Excella" keeps coming back as "FXCELLA" or
"Pxcella", add a rule that fixes it. Don't add a rule for a one-off
misread; regex rules are for patterns you expect to recur.

Structure: CLEANUP_RULES maps a field NAME to a list of (pattern,
replacement) pairs, applied in order with re.sub. A special key "*"
holds rules applied to EVERY field, before that field's own specific
rules run.

Patterns are plain Python regex (re.sub syntax) -- capture groups and
backreferences (\\1 etc) work normally. Keep replacement pairs small
and specific; a rule that's too broad can silently mangle a field it
wasn't meant for. If you're not sure a pattern is safe, test it against
a few past results.jsonl entries or SQLite rows before trusting it on
new photos.

Confirmed working example (tested against a real Excella E3415 photo,
where tesseract reliably adds a stray leading bracket to the pattern
number -- '[E3415' instead of 'E3415'):

    "pattern_number": [
        (r"^\\[", ""),
    ],

Add more entries the same way -- one field name (or "*") mapping to a
list of (pattern, replacement) tuples.
"""

from __future__ import annotations

CLEANUP_RULES: dict[str, list[tuple[str, str]]] = {
    "pattern_number": [
        (r"^\[", ""),  # strip stray leading bracket tesseract adds on Excella's pattern_number crop
    ],
}
