"""Local test runner for field_extraction. NOT part of the installed
envelope_mappings package -- run it directly by hand while tuning
bboxes/OCR settings:

    ~/envelope_env/bin/python field_extraction/run_test.py \\
        /path/to/excella_E3415_front.jpg Excella E3415 front

Positional args are image, company, pattern_number, side -- see
field_regions.ENVELOPE_FIELD_MAPS for the known (company,
pattern_number, side) combinations.

Rotation is auto-detected per photo (see extractor.detect_rotation) --
no flag needed. Prints each field's OCR value + confidence to stdout.
Results are also persisted via storage.save_result() -- to a SQLite DB
if FIELD_EXTRACTION_DB_PATH is set in the environment, otherwise
appended to field_extraction/results/results.jsonl. See storage.py's
docstring for details. Nothing here touches the real package or its
tests.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent))
from extractor import apply_detected_rotation, extract_field_map  # noqa: E402
from field_regions import ENVELOPE_FIELD_MAPS  # noqa: E402
from storage import save_result  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_path", type=Path)
    parser.add_argument("company")
    parser.add_argument("pattern_number")
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

    key = (args.company, args.pattern_number, args.side)
    field_map = ENVELOPE_FIELD_MAPS.get(key)
    if field_map is None:
        known = (
            ", ".join(f"{c}/{p}/{s}" for c, p, s in ENVELOPE_FIELD_MAPS)
            or "(none yet)"
        )
        sys.exit(f"No field map for {args.company}/{args.pattern_number}/{args.side}. Known: {known}")

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

    print(f"\n{field_map.company} {field_map.pattern_number} ({field_map.side})")
    print(f"source: {args.image_path}")
    source_label = "forced" if args.force_rotation is not None else "detected"
    print(f"rotation applied: {rotation_applied} deg ({source_label})")
    print("-" * 72)
    confidences = []
    for name, r in results.items():
        conf_str = f"{r['confidence']:.0f}" if r["confidence"] is not None else "  —"
        note = field_map.get(name).note
        note_str = f"  [{note}]" if note else ""
        print(f"{name:35s} conf={conf_str:>4s}  {r['value']!r}{note_str}")
        if r["confidence"] is not None:
            confidences.append(r["confidence"])

    if not confidences or (sum(confidences) / len(confidences)) < 20:
        print(
            "\n  WARNING: every field came back empty or very low confidence. "
            "This usually means the rotation is wrong, not that these bboxes "
            "are bad -- try --force-rotation with a value 90 degrees off from "
            f"what was applied here ({rotation_applied})."
        )

    saved_to = save_result(
        {
            "company": field_map.company,
            "pattern_number": field_map.pattern_number,
            "side": field_map.side,
            "image": str(args.image_path),
            "rotation_applied_degrees": rotation_applied,
            "rotation_source": source_label,
            "results": results,
        }
    )
    print(f"\nSaved to {saved_to}")


if __name__ == "__main__":
    main()
