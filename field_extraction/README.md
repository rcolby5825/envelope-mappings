# field_extraction — dev/test harness

Standalone folder, NOT part of the installed `envelope_mappings`
package. Purpose: measure and test `field_regions` bboxes against real
envelope photos before copying confirmed values into
`src/envelope_mappings/templates/<company>.py`.

## Files

- `classify.py` — auto-classification MECHANISM used by `webapp.py`:
  tries every known template against a photo and picks whichever
  actually fits, instead of the person picking one from a dropdown.
  See "Auto-classification" below.
- `field_regions.py` — DATA ONLY. `FieldRegion`/`EnvelopeFieldMap`
  dataclasses, plus `find_field_map()` (CLI lookup) and
  `list_templates()` (enumeration used by `classify.py`). Currently
  covers two reference envelopes: Excella E3415 and Pictorial 7117,
  both front and back.
- `extractor.py` — the extraction MECHANISM (rotation detection, crop,
  OCR), kept separate from the data above, same split as the
  production package's `template.py`/`extraction.py`.
- `cleanup_rules.py` — DATA ONLY. Regex find/replace rules for
  recurring OCR mistakes, keyed by field name. Edit this directly to
  add a rule; see "Cleaning up OCR output" below.
- `cleanup.py` — the MECHANISM that applies `cleanup_rules.py`'s
  patterns. Kept separate from that data for the same reason as
  everything else here.
- `run_test.py` — CLI you run by hand to test one specific template
  against a photo (explicit company/pattern_number/side, no
  auto-detection); prints results (raw + cleaned) and persists them
  via `storage.py`.
- `webapp.py` — local browser UI, for adding/testing photos without
  typing CLI args. Upload a photo, submit — no dropdown; which
  template applies is auto-detected (see `classify.py`). Not installed
  as part of the package; requires `flask` (not a real package
  dependency, install it directly — see below).
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

## Auto-classification (browser UI)

`webapp.py` doesn't ask which company/side a photo is — `classify.py`
figures it out. Two stages:

1. **Company, via keyword match.** For any template with a
   `company_name` field, OCR just that field and check (fuzzily, to
   tolerate OCR noise) whether the expected company keyword appears.
   If exactly one company matches confidently, candidates narrow to
   that company's templates only.
2. **Side, via mean field confidence**, among whatever's left after
   stage 1.

Why not confidence alone for everything: tried that first and it
mis-classified an Excella front photo as Pictorial/back — these
envelopes are covered in dense text nearly everywhere, so even the
WRONG template's bboxes usually land on some real, legible text and
score a plausible confidence (the wrong guess scored within 1 point of
the right one). Keyword matching checks WHAT the text says, not just
how legible it looks, which is a much stronger signal where it's
available. Confirmed via testing: 4/4 known photos (both companies,
both sides) now classify correctly.

This is a lighter stand-in for the production package's real
logo/fingerprint classifier (`src/envelope_mappings/classifier.py`),
which currently can't even be imported due to an unrelated pre-existing
bug (`extraction.py` imports `FieldResult` from `results.py`, which
doesn't define it). See `classify.py`'s docstring for the full
reasoning and when it'd be worth switching to the real thing.

The results page always shows what was detected and its confidence
(`Auto-detected as Excella / front (classification confidence: 74)`),
so a wrong guess is visible immediately rather than silent.

## Testing photos beyond the exact reference envelope

Every field map was measured against ONE specific reference photo
(Excella E3415, Pictorial 7117), but the whole point of a template is
that the same bboxes should work reasonably well on OTHER envelopes
from that company with a similar layout — a different Excella pattern
number, for instance, not just E3415 itself. This is what makes
auto-classification useful at all: `classify.py` matches by
company/side, never by the exact reference pattern number.

The CLI's `run_test.py` supports the same idea explicitly:
`find_field_map()` in `field_regions.py` tries an exact `(company,
pattern_number, side)` match first, and if that fails, falls back to
any template for the same `(company, side)`. It prints a note when
this fallback kicks in: `using Excella/E3415's regions as the closest
template -- no exact match for this pattern number`.

This is a genuine "best effort, not guaranteed" situation — a
template's bboxes might not line up as tightly on a photo they weren't
measured against (different year, different print layout, a different
photo crop). Worth treating a non-exact match with a bit more scrutiny
than an exact one — the "pattern number" you actually typed/typed
into the browser is preserved separately from `template_pattern_number`
in every saved result specifically so you can tell them apart later.

If more than one template ever exists for the same `(company, side)` —
not the case yet, but plausible once a company has multiple
genuinely-different-era layouts (McCall's 6600 vs 8306 were like this
in earlier testing) — both `find_field_map()` and `list_templates()`
pick/include whichever one they find first, with no attempt to guess
which fits best. That's a known simplification; if it matters later it
needs explicit selection, not a silent guess.

## Cleaning up OCR output

`cleanup_rules.py` holds regex find/replace rules, keyed by field name
(or `"*"` for a rule applied to every field), applied via
`cleanup.clean_value()`. It ships with one confirmed-working example —
Excella's `pattern_number` reliably gets a stray leading `[` from
tesseract, which a rule there strips.

Add a rule once you notice OCR making the SAME mistake more than
once — not for a one-off misread. Both front doors always show the
*raw* OCR value plainly, alongside the *cleaned* one, regardless of
whether a rule exists for that field yet — the whole point is to be
able to look at the raw output and decide what rule to write, not to
have it hidden until a rule already exists. Both values are persisted
to `results.jsonl` / SQLite too, so nothing is silently lost to a bad
rule. A pattern that fails to compile is skipped with a printed
warning rather than crashing the run.

## Usage — command line

```
~/envelope_env/bin/pip install -e ".[dev]"   # if not already
~/envelope_env/bin/python field_extraction/run_test.py \
    /path/to/excella_E3415_front.jpg Excella E3415 front
```

Positional args are `image_path company pattern_number side`. This is
explicit, deliberate selection — no auto-classification here, that's
the browser UI's job (see "Auto-classification" above). `pattern_number`
is the ACTUAL number on the photo you're testing — it does not need to
match a reference envelope exactly; see "Testing photos beyond the
exact reference envelope" above.

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

Either way, the record shape is close to the same from both front
doors, with one difference: the CLI (explicit selection) includes
`is_exact_template` (was the typed pattern_number an exact match or a
fallback); the browser (auto-classification) includes
`classification_confidence` instead (how confident `classify.py` was).
Both include `company`, `pattern_number` (the actual one on the
photo), `template_pattern_number` (which reference template's regions
were actually used), `side`, `image` (path), `rotation_applied_degrees`,
`rotation_source`, `results` (the full per-field dict, each entry
including both `value` and `cleaned_value`), plus a `created_at`
timestamp `storage.py` adds automatically. In the SQLite case,
`results` is stored as a JSON-encoded `results_json` column — quick to
query the top-level columns directly, quick to load the full field
detail back out with `json.loads()`.

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

Open `http://127.0.0.1:5151` in a browser. Upload a photo of any
Excella or Pictorial pattern envelope (front or back), optionally type
in the actual pattern number printed on it, hit submit — no dropdown,
which template applies is auto-detected (see "Auto-classification"
above). Results render as:

- A banner showing what was auto-detected and how confident that
  classification was.
- A table with BOTH the raw OCR value and the cleaned value for every
  field, always — not hidden behind a rule needing to exist first, see
  "Cleaning up OCR output" above.
- A preview of the rotated image.
- A low-confidence warning banner if every field came back weak
  (usually means the wrong rotation was applied — the banner suggests
  the 180°-opposite to try via the "Force rotation" dropdown).
- A collapsible "Raw result (JSON)" block with the complete
  unformatted result — the same shape that gets persisted — for
  copy-pasting or a closer look.

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
