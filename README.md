# field_extraction — dev/test harness

Standalone folder, NOT part of the installed `envelope_mappings`
package. Purpose: measure and test `field_regions` bboxes against real
envelope photos before copying confirmed values into
`src/envelope_mappings/templates/<company>.py`.

## Files

- `field_regions.py` — DATA ONLY. `FieldRegion`/`EnvelopeFieldMap`
  dataclasses, no methods beyond a trivial `.get()` lookup. Currently
  covers two reference envelopes: Excella E3415 and Pictorial 7117,
  both front and back.
- `extractor.py` — the extraction MECHANISM (rotation detection, crop,
  OCR), kept separate from the data above, same split as the
  production package's `template.py`/`extraction.py`.
- `run_test.py` — CLI you run by hand to test a field map against a
  photo; prints results and persists them via `storage.py`.
- `webapp.py` — local browser UI over the same mechanism as
  `run_test.py`, for adding/testing photos without typing CLI args.
  Pick the envelope from a dropdown, upload the photo, submit. Not
  installed as part of the package; requires `flask` (not a real
  package dependency, install it directly — see below).
- `storage.py` — shared persistence for both front doors above. Writes
  to a SQLite DB if `FIELD_EXTRACTION_DB_PATH` is set in the
  environment, otherwise appends to `results/results.jsonl`. See
  "Persistence" below.
- `results/` — either `results.jsonl` (one line per run) or nothing,
  if you're using the SQLite path instead. Gitignored (the images
  themselves aren't in the repo either).
- `uploads/` — photos uploaded via `webapp.py` land here (gitignored,
  same treatment as `results/`). `run_test.py` doesn't use this
  directory — it reads whatever path you pass it directly.

## Usage — command line

```
~/envelope_env/bin/pip install -e ".[dev]"   # if not already
~/envelope_env/bin/python field_extraction/run_test.py \
    /path/to/excella_E3415_front.jpg Excella E3415 front
```

Positional args are `image_path company pattern_number side`. Both
`company` and `pattern_number` must match a key in
`ENVELOPE_FIELD_MAPS` (case-sensitive) — currently `Excella`/`E3415`
or `Pictorial`/`7117`, each with `side` `front` or `back`.

The `(company, pattern_number, side)` key (rather than just
`(company, side)`) is deliberate: a single company can have multiple
reference envelopes with different layouts, so company+side alone
isn't a safe unique key even though both companies here currently have
just one reference envelope each.

Rotation is auto-detected per photo — no flag needed. Add
`--force-rotation {0,90,180,270}` only if detection gets a specific
photo wrong.

## Persistence

Both `run_test.py` and `webapp.py` save every result through
`storage.save_result()`, controlled by one environment variable:

```
FIELD_EXTRACTION_DB_PATH
```

- **Set** (e.g. `export FIELD_EXTRACTION_DB_PATH=~/milfoil/envelopes.db`)
  — results are written to a SQLite database at that path. Table
  (`extraction_results`) is created automatically on first use; no
  setup needed beyond setting the variable. SQLite specifically
  because that's already the planned DB for the rest of this
  pipeline.
- **Unset** — results are appended as JSON Lines to
  `results/results.jsonl` instead. This is the default, works with
  zero configuration.

Either way, the record shape is the same: `company`, `pattern_number`,
`side`, `image` (path), `rotation_applied_degrees`, `rotation_source`,
`results` (the full per-field dict), plus a `created_at` timestamp
`storage.py` adds automatically. In the SQLite case, `results` is
stored as a JSON-encoded `results_json` column — quick to query the
top-level columns directly, quick to load the full field detail back
out with `json.loads()`.

Switching the env var mid-session is fine — the CLI and the browser
both just check `os.environ` at save time, nothing is cached. Older
individual timestamped JSON files (if you have any from before this
existed) aren't touched or migrated by any of this.

## Usage — browser (no command line needed after setup)

One-time setup:

```
~/envelope_env/bin/pip install flask
```

Then run the server (this part still needs a terminal, once):

```
~/envelope_env/bin/python field_extraction/webapp.py
```

Open `http://127.0.0.1:5151` in a browser. Pick the envelope from the
dropdown (same `(company, pattern_number, side)` combos as the CLI),
choose a photo file, hit submit. Results render as a table with the
same value/confidence data the CLI prints, plus a preview of the
rotated image and a low-confidence warning banner if every field came
back weak (usually means the wrong rotation was applied — the banner
suggests the 180°-opposite to try via the "Force rotation" dropdown).

Every submission is saved through the same `storage.save_result()`
that `run_test.py` uses (see "Persistence" above), so both front doors
share one result history — switching between the CLI and the browser
mid-session doesn't lose anything, and switching `FIELD_EXTRACTION_DB_PATH`
on or off between runs is fine too.

**Known limitation**: the low-confidence warning is calibrated for
"every field came back completely empty," not "every field came back
garbled but still confidently-scored" — a wrong rotation sometimes
still produces moderate per-field confidence numbers on nonsense text,
in which case the banner won't fire even though the result is wrong.
Worth eyeballing the actual field values, not just the color-coding,
especially on a photo you haven't tested before.

# Envelope Mappings

Python tools for recognizing sewing-pattern envelopes and extracting
fields from their images. The project has two related parts:

- `src/envelope_mappings/` is the installable package. It provides a
  two-stage classifier that matches a company logo, then identifies a
  year or pattern variant with a visual fingerprint.
- `field_extraction/` is a standalone development harness for measuring
  field regions and testing OCR against real envelope photos. It is not
  included in the installed package.

## Installation

Requires Python 3.9 or newer, OpenCV, NumPy, and Tesseract OCR.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Install the Tesseract executable separately if it is not already
available. On macOS with Homebrew:

```bash
brew install tesseract
```

## Run the tests

```bash
python -m pytest
```

The test suite uses synthetic images to cover classification, logo
matching, fingerprints, lazy reference paths, and OCR field extraction.

## Package usage

Create `CompanyLogo` and `EnvelopeTemplate` instances with reference
images, then pass them to `EnvelopeClassifier`:

```python
import cv2

from envelope_mappings import CompanyLogo, EnvelopeClassifier
from envelope_mappings.templates.excella import Excella

envelope = cv2.imread("/path/to/envelope.jpg")
logo_region = cv2.imread("/path/to/logo-crop.jpg")

logo = CompanyLogo("Excella")
logo.set_reference_path("/path/to/excella-logo.jpg")

template = Excella()
template.set_reference_path("/path/to/excella-reference.jpg")

classifier = EnvelopeClassifier(logos=[logo], templates=[template])
result = classifier.classify(envelope, logo_region=logo_region)
print(result)
```

Classification returns one of:

- `PatternRecord` when both company and template confidence are high.
- `AmbiguousMatch` when the template score is between the configured
  low and high thresholds.
- `NewTemplateNeeded` when no registered logo or template is a match.

Templates define their own `field_regions` and can override
`fingerprint()` or `extract_fields()` for envelope-specific behavior.
Reference images may be assigned before they exist with
`set_reference_path()`; they are loaded lazily when first used.

## Field-extraction harness

The harness currently includes measured maps for Excella E3415 and
Pictorial 7117, each with front and back layouts. Run it from the
repository root after installing the package dependencies:

```bash
python field_extraction/run_test.py \
  /path/to/photo.jpg Excella E3415 front
```

The positional arguments are `image_path company pattern_number side`.
Rotation is detected automatically; use `--force-rotation 0`, `90`,
`180`, or `270` for a specific photo when needed.

For a local browser UI, install Flask and start the server:

```bash
python -m pip install flask
python field_extraction/webapp.py
```

Open <http://127.0.0.1:5151>, select an envelope layout, upload a photo,
and submit it for extraction. Uploaded images go to
`field_extraction/uploads/`.

## Result storage

By default, CLI and browser runs append JSON Lines to
`field_extraction/results/results.jsonl`. To use SQLite instead, set:

```bash
export FIELD_EXTRACTION_DB_PATH="$HOME/envelope-results.db"
```

The database and scratch uploads/results are ignored by Git.

## Project layout

```text
proto/                       Protocol Buffer schema
src/envelope_mappings/       Installable package
src/envelope_mappings/templates/  Envelope-specific templates
field_extraction/            OCR measurement and browser harness
tests/                       Unit tests
```

## Current scope

The classifier and extraction primitives are implemented and tested.
The checked-in Excella and Pictorial templates contain field-region
configuration, while production template-specific extraction remains
an extension point. The field-extraction harness is the place to
measure and validate new regions before promoting them into templates.
