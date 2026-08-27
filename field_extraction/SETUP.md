# Setting up the field_extraction web UI

This is the quick-start for the browser-based tool that lets you test
envelope photos without typing CLI commands each time. For everything
else about `field_extraction` — the CLI, how the OCR mechanism works,
known limitations — see `README.md` in this same folder.

## 1. One-time setup

You need the same venv already used for the rest of `field_extraction`
(`~/envelope_env`), plus Flask, which isn't part of the regular
project dependencies:

```bash
~/envelope_env/bin/pip install flask
```

That's the only extra install. Everything else (opencv, pytesseract,
etc.) should already be there from the existing setup.

## 2. Start the server

From the repo root:

```bash
~/envelope_env/bin/python field_extraction/webapp.py
```

Optional: if you want results saved to a SQLite database instead of
the default JSONL file, set the env var before starting the server:

```bash
export FIELD_EXTRACTION_DB_PATH=~/wherever/envelopes.db
~/envelope_env/bin/python field_extraction/webapp.py
```

Leave this running in its terminal window — it's a local web server,
not a one-shot script. You'll see Flask's startup log; it's ready once
you see a line like:

```
Running on http://127.0.0.1:5151
```

## 3. Open it in a browser

Go to:

```
http://127.0.0.1:5151
```

From here on, everything is point-and-click — no terminal needed until
you want to stop the server.

## 4. Using it

1. **Pick a template** from the dropdown — this is grouped by company
   and side, not tied to one specific reference envelope, so any
   Excella or Pictorial photo can use it (e.g. "Excella — front
   (regions measured from E3415)" works for an E3415 photo just as
   well as some other Excella pattern number).
2. **Pattern number on this actual envelope** (optional) — type in
   what's actually printed on the photo you're uploading. Leave it
   blank if you don't know or it doesn't matter for this test. If you
   type something that doesn't match the template's own reference
   number, you'll see a blue info banner confirming it fell back to
   the closest available template rather than an exact match — that's
   expected, not an error.
3. **Choose a photo file** from your computer.
4. Leave **Rotation** on "Auto-detect" unless you already know a
   specific photo needs a forced rotation.
5. Click **Extract fields**.

You'll land on a results page showing:

- A table of every field for that envelope side, with its *cleaned*
  value (see `cleanup_rules.py` in `README.md`) and a confidence score
  (green = solid, amber = shaky, gray = nothing detected). If cleanup
  actually changed something, the original raw OCR text shows in gray
  underneath.
- A preview of the photo as the tool actually saw it, rotation
  applied — useful for a quick gut-check on whether it landed upright.
- A warning banner if every field came back weak, with a suggested
  rotation to try instead.
- A collapsible "Raw result (JSON)" section near the bottom with the
  complete unformatted result, if you want to copy the whole thing out
  rather than read the table.

Click "test another photo" to go back and do another one.

## 5. Stopping the server

Back in the terminal where it's running, press `Ctrl+C`.

## Where things get saved

- Uploaded photos: `field_extraction/uploads/`
- Extraction results: either `field_extraction/results/results.jsonl`
  (default, one line per test) or a SQLite database, if you've set the
  `FIELD_EXTRACTION_DB_PATH` environment variable before starting the
  server. See "Persistence" in `README.md` for details — the short
  version is: no setup needed for the JSONL default, or point that
  env var at a `.db` file path if you'd rather query results with SQL.

Both `uploads/` and the JSONL file are gitignored — treat them as
scratch space, not something to commit. If you also use the
`run_test.py` command-line tool, it saves through the exact same
mechanism, so both ways of testing share one history.
