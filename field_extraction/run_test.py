"""Local test runner for field_extraction. NOT part of the installed
envelope_mappings package -- run it directly by hand while tuning
bboxes/OCR settings:

    ~/envelope_env/bin/python field_extraction/run_test.py \
        /path/to/excella_E3415_front.jpg Excella E3415 front

Positional args are image, company, pattern_number, side.
pattern_number is the ACTUAL number printed on the photo you're
testing -- it does NOT need to match a reference envelope exactly.
Lookup tries an exact (company, pattern_number, side) match first,
then falls back to any template for the same (company, side) -- see
field_regions.find_field_map's docstring for why. This means any
Excella or Pictorial photo can be tested, not just the literal
reference envelopes those regions were originally measured against.

Rotation is auto-detected per photo (see extractor.detect_rotation) --
no flag needed. Prints each field's raw + cleaned OCR value (see
cleanup.py/cleanup_rules.py) and confidence to stdout, plus the full
raw result as JSON at the end. Results are also persisted via
storage.save_result() -- to a SQLite DB if FIELD_EXTRACTION_DB_PATH is
set in the environment, otherwise appended to
field_extraction/results/results.jsonl. See storage.py's docstring for
details. Nothing here touches the real package or its tests.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent))
from cleanup import clean_value  # noqa: E402
from extractor import apply_detected_rotation, extract_field_map  # noqa: E402
from field_regions import find_field_map  # noqa: E402
from storage import save_result  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_path", type=Path)
    parser.add_argument("company")
    parser.add_argument("pattern_number", help="actual pattern number on this photo")
    parser.add_argument("side", choices=["front", "back"])
    parser.add_argument(
        "--force-rotation",
        type=int,
        choices=[0, 90, 180, 270],
        default=None,
        help="Skip auto-detection and use this rotation instead "
        "(e.g. if detection gets a photo wrong).",
    )
    args = parser.parse_args()

    field_map, is_exact_template = find_field_map(
        args.company, args.pattern_number, args.side
    )
    if field_map is None:
        sys.exit(f"No template available for {args.company}/{args.side}.")

    image = cv2.imread(str(args.image_path))
    if image is None:
        sys.exit(f"Could not read image at {args.image_path}")

    if args.force_rotation is not None:
        from extractor import _ROTATION_TO_CV2

        cv2_constant = _ROTATION_TO_CV2[args.force_rotation]
        image = image if cv2_constant is None else cv2.rotate(image, cv2_constant)
        rotation_applied = args.force_rotation
    else:
        image, rotation_applied = apply_detected_rotation(image)

    results = extract_field_map(image, field_map)

    print(f"\n{args.company} {args.pattern_number} ({args.side})")
    if not is_exact_template:
        print(
            f"  using {args.company}/{field_map.pattern_number}'s regions as the "
            "closest template -- no exact match for this pattern number"
        )
    print(f"source: {args.image_path}")
    source_label = "forced" if args.force_rotation is not None else "detected"
    print(f"rotation applied: {rotation_applied} deg ({source_label})")
    print("-" * 72)
    confidences = []
    for name, r in results.items():
        conf_str = f"{r['confidence']:.0f}" if r["confidence"] is not None else "  —"
        note = field_map.get(name).note
        note_str = f"  [{note}]" if note else ""
        cleaned = clean_value(name, r["value"])
        cleaned_str = f"  (cleaned: {cleaned!r})" if cleaned != r["value"] else ""
        print(f"{name:35s} conf={conf_str:>4s}  {r['value']!r}{cleaned_str}{note_str}")
        if r["confidence"] is not None:
            confidences.append(r["confidence"])

    if not confidences or (sum(confidences) / len(confidences)) < 20:
        print(
            "\n  WARNING: every field came back empty or very low confidence. "
            "This usually means the rotation is wrong, not that these bboxes "
            "are bad -- try --force-rotation with a value 90 degrees off from "
            f"what was applied here ({rotation_applied})."
        )

    results_with_cleaned = {
        name: {**r, "cleaned_value": clean_value(name, r["value"])}
        for name, r in results.items()
    }

    saved_to = save_result(
        {
            "company": args.company,
            "pattern_number": args.pattern_number,
            "template_pattern_number": field_map.pattern_number,
            "is_exact_template": is_exact_template,
            "side": args.side,
            "image": str(args.image_path),
            "rotation_applied_degrees": rotation_applied,
            "rotation_source": source_label,
            "results": results_with_cleaned,
        }
    )

    print("\nRaw result (JSON):")
    print(json.dumps(results_with_cleaned, indent=2))
    print(f"\nSaved to {saved_to}")


if __name__ == "__main__":
    main()
