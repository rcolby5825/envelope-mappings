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

1. **Pick the envelope** from the dropdown — this tells the tool which
   set of field regions to use (e.g. "Excella E3415 — front").
2. **Choose a photo file** from your computer.
3. Leave **Rotation** on "Auto-detect" unless you already know a
   specific photo needs a forced rotation.
4. Click **Extract fields**.

You'll land on a results page showing:

- A table of every field for that envelope side, with its extracted
  text and a confidence score (green = solid, amber = shaky, gray =
  nothing detected).
- A preview of the photo as the tool actually saw it, rotation
  applied — useful for a quick gut-check on whether it landed upright.
- A warning banner if every field came back weak, with a suggested
  rotation to try instead.

Click "test another photo" to go back and do another one.

## 5. Stopping the server

Back in the terminal where it's running, press `Ctrl+C`.

## Where things get saved

- Uploaded photos: `field_extraction/uploads/`
- Results (one JSON file per test, timestamped): `field_extraction/results/`

Both are gitignored — treat them as scratch space, not something to
commit. If you also use the `run_test.py` command-line tool, it writes
to the same `results/` folder, so both ways of testing share one
history.
