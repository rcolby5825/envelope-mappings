"""Local web UI for field_extraction -- upload a photo and let it
auto-detect which company/side template fits, instead of typing CLI
args or picking from a dropdown. Same underlying mechanism as
run_test.py (extractor.py, field_regions.py, classify.py); this is
just a different front door onto it. NOT part of the installed
envelope_mappings package.

One-time setup (Flask isn't a dependency of the real package, so it's
not in pyproject.toml -- install it directly in the same venv):

    ~/envelope_env/bin/pip install flask

Run it:

    ~/envelope_env/bin/python field_extraction/webapp.py

Then open http://127.0.0.1:5151 in a browser. Upload a photo of any
Excella or Pictorial pattern envelope (front or back) -- which
template applies is detected automatically (see classify.py), not
chosen from a dropdown. Uploaded photos are saved to
field_extraction/uploads/ (gitignored, same treatment as the reference
photos themselves) and results are persisted via storage.save_result()
-- to a SQLite DB if FIELD_EXTRACTION_DB_PATH is set in the
environment, otherwise appended to field_extraction/results/results.jsonl.
See storage.py's docstring for details.
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import cv2
from flask import Flask, abort, render_template_string, request

sys.path.insert(0, str(Path(__file__).parent))
from classify import auto_classify  # noqa: E402
from cleanup import clean_value  # noqa: E402
from extractor import _ROTATION_TO_CV2, apply_detected_rotation  # noqa: E402
from field_regions import list_templates  # noqa: E402
from storage import save_result  # noqa: E402

UPLOADS_DIR = Path(__file__).parent / "uploads"

app = Flask(__name__)

PAGE_STYLE = """
<style>
  body { font-family: system-ui, sans-serif; max-width: 760px; margin: 2rem auto; padding: 0 1rem; color: #222; }
  h1 { font-size: 1.4rem; }
  label { display: block; margin-top: 1.2rem; font-weight: 600; }
  .hint { font-weight: 400; color: #666; font-size: .85em; }
  select, input[type=file], input[type=text] { width: 100%; padding: .45rem; margin-top: .3rem; font-size: 1rem; box-sizing: border-box; }
  button { margin-top: 1.6rem; padding: .6rem 1.3rem; font-size: 1rem; cursor: pointer; }
  table { border-collapse: collapse; width: 100%; margin-top: 1rem; }
  th, td { text-align: left; padding: .4rem .6rem; border-bottom: 1px solid #ddd; vertical-align: top; }
  th { background: #f5f5f5; }
  .conf-good { color: #1a7f37; }
  .conf-low { color: #b35900; }
  .conf-none { color: #999; }
  .note { font-size: .85em; color: #999; }
  .raw-col { color: #555; font-family: ui-monospace, monospace; font-size: .9em; }
  .cleaned-col { font-weight: 600; }
  .unchanged { color: #999; font-weight: 400; }
  img.preview { max-width: 100%; margin-top: 1rem; border: 1px solid #ccc; }
  .warning { background: #fff3cd; border: 1px solid #ffe69c; padding: .8rem; margin-top: 1rem; border-radius: 4px; }
  .info { background: #e7f1ff; border: 1px solid #c3ddff; padding: .8rem; margin-top: 1rem; border-radius: 4px; }
  a.back { display: inline-block; margin-top: 1.5rem; }
  pre.rawjson { background: #f5f5f5; padding: 1rem; overflow-x: auto; font-size: .8rem; margin-top: .5rem; }
  details { margin-top: 1.5rem; }
  summary { cursor: pointer; font-weight: 600; }
</style>
"""

FORM_TEMPLATE = f"""
<!doctype html>
<title>field_extraction</title>
{PAGE_STYLE}
<h1>Test an envelope photo</h1>
<p class=hint>Which company/side template applies is auto-detected from the photo -- see classify.py.</p>
<form method=post action="/extract" enctype=multipart/form-data>
  <label>Photo
    <input type=file name=photo accept="image/*" required>
  </label>
  <label>Pattern number on this actual envelope
    <span class=hint>(optional -- what's printed on the photo, for the log; doesn't affect detection)</span>
    <input type=text name=actual_pattern_number placeholder="e.g. E5000, or leave blank">
  </label>
  <label>Rotation
    <select name=force_rotation>
      <option value="">Auto-detect (recommended)</option>
      <option value="0">Force 0&deg;</option>
      <option value="90">Force 90&deg;</option>
      <option value="180">Force 180&deg;</option>
      <option value="270">Force 270&deg;</option>
    </select>
  </label>
  <button type=submit>Extract fields</button>
</form>
"""

NO_TEMPLATES_TEMPLATE = f"""
<!doctype html>
<title>field_extraction</title>
{PAGE_STYLE}
<h1>Test an envelope photo</h1>
<div class=warning>
  <strong>No templates defined yet.</strong> ENVELOPE_FIELD_MAPS in
  field_regions.py is empty -- add at least one EnvelopeFieldMap before
  there's anything to auto-detect or test a photo against.
</div>
"""

RESULT_TEMPLATE = f"""
<!doctype html>
<title>field_extraction — results</title>
{PAGE_STYLE}
<h1>{{{{ company }}}} {{{{ display_pattern_number }}}} &mdash; {{{{ side }}}}</h1>
<div class=info>Auto-detected as <strong>{{{{ company }}}} / {{{{ side }}}}</strong> (classification confidence: {{{{ classify_confidence }}}}).</div>
<p>rotation applied: <strong>{{{{ rotation_applied }}}}&deg;</strong> ({{{{ rotation_source }}}})</p>
{{% if warning %}}<div class=warning>{{{{ warning }}}}</div>{{% endif %}}
<table>
  <tr><th>Field</th><th>Confidence</th><th>Raw OCR value</th><th>Cleaned value</th></tr>
  {{% for name, r in results.items() %}}
  <tr>
    <td>{{{{ name }}}}{{% if r.note %}}<br><span class=note>{{{{ r.note }}}}</span>{{% endif %}}</td>
    <td class="{{{{ r.conf_class }}}}">{{{{ r.conf_display }}}}</td>
    <td class=raw-col>{{{{ r.raw_value }}}}</td>
    <td class="{{{{ 'cleaned-col' if r.raw_value != r.cleaned_value else 'unchanged' }}}}">{{{{ r.cleaned_value }}}}</td>
  </tr>
  {{% endfor %}}
</table>
<img class=preview src="data:image/jpeg;base64,{{{{ preview_b64 }}}}" alt="rotated photo">
<details>
  <summary>Raw result (JSON)</summary>
  <pre class=rawjson>{{{{ raw_json }}}}</pre>
</details>
<p><a class=back href="/">&larr; test another photo</a></p>
<p class=note>Saved to {{{{ result_path }}}}</p>
"""


def _confidence_display(confidence: float | None) -> tuple[str, str]:
    if confidence is None:
        return "—", "conf-none"
    if confidence >= 60:
        return f"{confidence:.0f}", "conf-good"
    return f"{confidence:.0f}", "conf-low"


@app.route("/")
def index():
    if not list_templates():
        return render_template_string(NO_TEMPLATES_TEMPLATE)
    return render_template_string(FORM_TEMPLATE)


@app.route("/extract", methods=["POST"])
def extract():
    photo = request.files.get("photo")
    if photo is None or not photo.filename:
        abort(400, "No photo uploaded.")

    UPLOADS_DIR.mkdir(exist_ok=True)
    upload_path = UPLOADS_DIR / photo.filename
    photo.save(upload_path)

    image = cv2.imread(str(upload_path))
    if image is None:
        abort(400, f"Could not read uploaded file {photo.filename!r} as an image.")

    force_rotation = request.form.get("force_rotation") or None
    if force_rotation:
        rotation_applied = int(force_rotation)
        cv2_constant = _ROTATION_TO_CV2[rotation_applied]
        image = image if cv2_constant is None else cv2.rotate(image, cv2_constant)
        rotation_source = "forced"
    else:
        image, rotation_applied = apply_detected_rotation(image)
        rotation_source = "detected"

    try:
        company, side, field_map, raw_results, classify_confidence = auto_classify(image)
    except ValueError as exc:
        abort(404, str(exc))

    actual_pattern_number = request.form.get("actual_pattern_number", "").strip()
    display_pattern_number = actual_pattern_number or field_map.pattern_number

    display_results = {}
    confidences = []
    for name, r in raw_results.items():
        conf_display, conf_class = _confidence_display(r["confidence"])
        region = field_map.get(name)
        raw_value = r["value"] or "(empty)"
        cleaned = clean_value(name, r["value"]) or "(empty)"
        display_results[name] = {
            "raw_value": raw_value,
            "cleaned_value": cleaned,
            "conf_display": conf_display,
            "conf_class": conf_class,
            "note": region.note if region else "",
        }
        if r["confidence"] is not None:
            confidences.append(r["confidence"])

    warning = None
    if not confidences or (sum(confidences) / len(confidences)) < 20:
        opposite = (rotation_applied + 180) % 360
        warning = (
            "Every field came back empty or very low confidence. This usually "
            f"means the rotation is wrong -- try forcing {opposite}\u00b0 instead "
            f"of the {rotation_applied}\u00b0 applied here."
        )

    # Shared with run_test.py's CLI -- see storage.py for where this
    # actually lands (SQLite if FIELD_EXTRACTION_DB_PATH is set, JSONL
    # otherwise). Cleaned values are included alongside the raw OCR
    # output in the persisted record, not instead of it -- see
    # cleanup.py's docstring for why nothing overwrites the raw value.
    record = {
        "company": company,
        "pattern_number": display_pattern_number,
        "template_pattern_number": field_map.pattern_number,
        "classification_confidence": classify_confidence,
        "side": side,
        "image": str(upload_path),
        "rotation_applied_degrees": rotation_applied,
        "rotation_source": rotation_source,
        "results": {
            name: {**r, "cleaned_value": clean_value(name, r["value"])}
            for name, r in raw_results.items()
        },
    }
    result_path = save_result(record)

    preview_ok, preview_buf = cv2.imencode(".jpg", image)
    preview_b64 = base64.b64encode(preview_buf.tobytes()).decode("ascii") if preview_ok else ""

    return render_template_string(
        RESULT_TEMPLATE,
        company=company,
        display_pattern_number=display_pattern_number,
        side=side,
        classify_confidence=f"{classify_confidence:.0f}",
        rotation_applied=rotation_applied,
        rotation_source=rotation_source,
        results=display_results,
        warning=warning,
        preview_b64=preview_b64,
        result_path=result_path,
        raw_json=json.dumps(record["results"], indent=2),
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5151, debug=True)
