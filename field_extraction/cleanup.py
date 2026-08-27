"""Cleanup MECHANISM -- applies cleanup_rules.py's regex patterns to
raw OCR text. Kept separate from that data, same reasoning as every
other data/mechanism split in this folder.

This never modifies the raw OCR value in place -- callers keep both
the original and the cleaned version (see webapp.py and run_test.py),
so a bad regex rule never destroys information, only produces a bad
*cleaned* value sitting next to a still-intact raw one.
"""

from __future__ import annotations

import re

from cleanup_rules import CLEANUP_RULES


def clean_value(field_name: str, raw_value: str) -> str:
    """Applies CLEANUP_RULES["*"] (if any), then CLEANUP_RULES[field_name]
    (if any), in order, via re.sub. Returns the cleaned string -- never
    raises on a bad pattern in CLEANUP_RULES; a broken regex is skipped
    with a printed warning rather than crashing extraction, since a
    typo in a hand-edited rules file shouldn't take down the whole run.
    """
    value = raw_value
    for pattern, replacement in CLEANUP_RULES.get("*", []):
        value = _safe_sub(pattern, replacement, value, field_name)
    for pattern, replacement in CLEANUP_RULES.get(field_name, []):
        value = _safe_sub(pattern, replacement, value, field_name)
    return value


def _safe_sub(pattern: str, replacement: str, value: str, field_name: str) -> str:
    try:
        return re.sub(pattern, replacement, value)
    except re.error as exc:
        print(
            f"  [cleanup_rules: skipping bad pattern {pattern!r} for "
            f"field {field_name!r}: {exc}]"
        )
        return value
