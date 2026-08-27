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

## Current state (tested against the real Excella E3415 and Pictorial 7117 photos)

Rotation confirmed correct: front-cover photos on both envelopes came
out of the camera landscape with content sideways —
`cv2.ROTATE_90_CLOCKWISE` before applying the measured fractions is
right. Excella's back/flap photo needed no rotation; Pictorial's back
photo DID need it — rotation isn't a fixed per-company constant.

**Rotation is auto-detected**, not passed as a flag. Tesseract's own
orientation detection (`image_to_osd`) failed when run on the full
photo directly ("too few characters" — too much blank background
relative to the small printed area). Cropping first to just the
envelope's own bounding box (isolated via HSV saturation, since the
tan paper is more saturated than the gray background) before running
OSD fixed this: correct on both companies' front and back photos. See
`extractor.detect_rotation` for the implementation and full reasoning,
including a secondary edge-density-based bbox method and a
word-likeness tiebreak that exist to handle photos where the
saturation approach or raw OSD alone isn't enough (kept as general
robustness even though neither Excella nor Pictorial's photos actually
need the fallback path).

OCR quality by field, after multi-strategy thresholding (see
`extractor.py`'s `_prep_variants` docstring for the mechanism) and a
white-border pad before OCR (needed for tight crops like pattern
numbers, where tesseract badly mangles text touching the crop edge):

| Field | Excella E3415 | Pictorial 7117 |
|---|---|---|
| `pattern_number` | Good — "[E3415" (77 conf) | Good — exact "7117" (37 conf) |
| `price` | Good — "25 Cents" (70 conf) | Good — "45 Cents" (92 conf) |
| `company_name` | Good — "FXCELLA PATTERNS" (90 conf) | Good — full text (95 conf) |
| size/measurements | Partial — noisy but usable (~51-71 conf) | Good — (89 conf) |
| description/instructions | Good — legible, noisy punctuation (~79-87 conf) | n/a (Pictorial's front has no instructions block) |
| `materials_suitable` | n/a | Good (91 conf) |

Back/flap fields: Pictorial's back reads well across the board
(66-95 conf). Excella's back is mostly good except
`cutting_layout_label`, which is still clipping its right edge —
flagged in the field's `note`.

### What fixed it

1. **Background sliver breaks Otsu.** Any field bbox that clips even a
   little of the light background beyond the envelope's edge makes
   Otsu's global threshold flip the *entire* crop to solid black —
   not a partial-quality problem, a total failure. Several bboxes
   needed tightening because of this.
2. **No single threshold strategy works for every field.**
   `_prep_variants()` in `extractor.py` tries Otsu, adaptive (local)
   threshold, and inverted Otsu per crop and keeps whichever gives the
   highest mean OCR confidence. Needed because e.g. Excella's price is
   printed reversed (light-on-dark), which only the inverted variant
   handles.
3. **Text touching the crop edge hurt tesseract badly.** A tight crop
   with text touching the top/bottom edge gave near-zero confidence
   and dropped characters. Adding a 30px white border around every
   binarized variant (also in `_prep_variants()`) fixed this.
4. **Multi-line fields need PSM 6, not PSM 7.** Any field spanning more
   than one line needs to be listed in `MULTILINE_FIELDS` in
   `extractor.py`, or it silently gets single-line OCR mode and reads
   as noise regardless of image quality. This has bitten more than
   once when adding a new field — worth double-checking whenever a
   field is added.

### Not yet done

- Nothing here has been copied into the real
  `src/envelope_mappings/templates/excella.py` /
  `templates/pictorial.py` `FIELD_REGIONS` yet.
- `cutting_layout_label` on Excella's back still needs its right edge
  widened slightly.
- Text quality on the paragraph/table fields is legible but noisy
  (stray punctuation, occasional misread letters) — fine for a human
  to read and correct, not yet clean enough to trust unreviewed for
  structured extraction. `FieldResult.valid`/`confidence` in the real
  package's extraction mechanism is exactly the mechanism meant to
  flag that kind of "readable but needs a human glance" result.
