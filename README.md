# Envelope Mappings
# Excella & Pictorial
Tools for identifying sewing-pattern envelopes and extracting printed
fields from envelope photos.

The repository contains two related layers:

- `src/envelope_mappings/` is the installable Python package. It defines
  company logos, visual fingerprints, envelope templates, classification
  result types, and a reusable OCR extraction mechanism.
- `field_extraction/` is a standalone development and test harness for
  measuring field regions and tuning OCR against real photos. It is not
  included in the installed package.

## Requirements

- Python 3.9+
- OpenCV
- NumPy
- Tesseract OCR

On macOS, install the Tesseract executable with Homebrew if needed:

```bash
brew install tesseract
```

## Install

Create a virtual environment and install the package with development
dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

The optional development dependencies provide `pytest` and `ruff`.
The browser harness also needs Flask:

```bash
python -m pip install flask
```

## Tests

```bash
python -m pytest
```

The tests cover package exports, logo matching, visual fingerprints,
template reference-image loading, classifier threshold outcomes, and
field extraction behavior.

## Package design

Classification is deliberately split into two stages:

1. `CompanyLogo` uses an ORB feature match to identify the likely
   company.
2. `EnvelopeFingerprint` compares color, edge, and text-layout features
   among templates for that company.

`EnvelopeClassifier.classify()` returns one of these result types:

- `PatternRecord`: a confident company and template match, with extracted
  fields.
- `AmbiguousMatch`: a company matched, but no template was sufficiently
  decisive.
- `NewTemplateNeeded`: no company or template matched confidently.

Templates are additive. Define an `EnvelopeTemplate` subclass, give it
fractional `field_regions` and optional `field_validators`, then register
it with an `EnvelopeClassifier`. Template names can derive `company` and
`year_code` from names such as `Vogue1970`; set those attributes explicitly
when the company has multiple words or the year is only a placeholder.

Reference images can be supplied in memory for tests or by path for real
files:

```python
from envelope_mappings import CompanyLogo, EnvelopeClassifier
from envelope_mappings.templates.excella import build_logo, build_template

logo = build_logo()
logo.set_reference_path("/path/to/excella-logo.jpg")

template = build_template()
# Reference images are loaded lazily when classification needs them.
template.set_reference_path("/path/to/excella-reference.jpg")

classifier = EnvelopeClassifier(logos=[logo], templates=[template])
```

`EnvelopeTemplate.extract_fields()` delegates to the shared
`FieldExtractor`. Each field region is cropped, OCR'd with Tesseract, and
returned as a `FieldResult` containing its value, validation state, and
confidence.

## Field-extraction harness

The harness currently has measured maps for:

- Excella E3415, front and back
- Pictorial 7117, front and back

The command-line runner explicitly accepts the company, actual pattern
number, and side. It first tries an exact pattern-number map, then falls
back to the closest company/side map:

```bash
python field_extraction/run_test.py \
  /path/to/photo.jpg Excella E3415 front
```

The positional arguments are `image_path company pattern_number side`.
Rotation is detected automatically. Override it for a difficult photo
with `--force-rotation 0`, `90`, `180`, or `270`.

The runner prints raw and cleaned OCR values with confidence scores and
persists the complete result. Repeated OCR mistakes can be added as
field-specific regular expressions in `field_extraction/cleanup_rules.py`;
the mechanism that applies them is in `cleanup.py`.

### Browser UI

Start the local Flask application from the repository root:

```bash
python field_extraction/webapp.py
```

Open <http://127.0.0.1:5151>. Upload an envelope photo, optionally enter
the pattern number printed on that photo, and submit it. The UI
auto-detects company and side, applies rotation detection, shows raw and
cleaned OCR values, and displays the rotated-image preview.

### Result storage

Without configuration, runs append JSON Lines to:

```text
field_extraction/results/results.jsonl
```

Use SQLite instead by setting the environment variable before running
the CLI or browser:

```bash
export FIELD_EXTRACTION_DB_PATH="$HOME/envelope-results.db"
```

Uploaded images are stored in `field_extraction/uploads/`. Results and
uploads are scratch data and are ignored by Git.

## Current status and limitations

The field-extraction harness is the active workflow for validating
regions and OCR against real envelope photos. Its auto-classification
currently handles the four known Excella/Pictorial reference layouts.

The package-level Excella and Pictorial templates contain company and
field-region scaffolding, but their template-specific `extract_fields()`
methods are still marked `NotImplementedError`. Their years are currently
represented by the placeholder value `1900` because the actual envelope
years are not known.

The package currently also has an import mismatch: `extraction.py` imports
`FieldResult` from `results.py`, but `results.py` does not yet define that
type. Resolve that mismatch before using the package-level classifier or
inherited template extraction.

## Layout

```text
proto/                            Protocol Buffer schema
src/envelope_mappings/            Installable package
src/envelope_mappings/templates/  Company-specific template scaffolds
field_extraction/                 OCR measurement and browser harness
tests/                            Unit tests
```
