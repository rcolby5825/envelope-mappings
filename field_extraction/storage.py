"""Persistence for extraction results -- shared by run_test.py and
webapp.py so both front doors log to the same place with the same
record shape, rather than each writing its own files.

Controlled by one environment variable:

    FIELD_EXTRACTION_DB_PATH

If set, results are written to a SQLite database at that path (table
created automatically on first use, via a plain INSERT -- see
_save_to_db). If NOT set, results are appended as JSON Lines to
field_extraction/results/results.jsonl instead -- no setup needed,
works out of the box.

SQLite specifically (not Postgres/MySQL/etc): matches the DB already
planned for the rest of this pipeline (see Milfoil's tech stack -- 
SQLite behind the Spring Boot API). If a different engine is ever
needed later, only _save_to_db below needs to change -- the env-var
branching and the record shape stay the same either way.

This intentionally replaces the older one-timestamped-JSON-file-per-run
behavior. Old individual files under results/ aren't touched or
migrated -- if you need that history, it's still sitting there as
plain files; this just doesn't add to it going forward.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DB_PATH_ENV_VAR = "FIELD_EXTRACTION_DB_PATH"

RESULTS_DIR = Path(__file__).parent / "results"
JSONL_PATH = RESULTS_DIR / "results.jsonl"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS extraction_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    company TEXT NOT NULL,
    pattern_number TEXT NOT NULL,
    side TEXT NOT NULL,
    image_path TEXT NOT NULL,
    rotation_applied_degrees INTEGER NOT NULL,
    rotation_source TEXT NOT NULL,
    results_json TEXT NOT NULL
)
"""


def save_result(record: dict[str, Any]) -> str:
    """Persists one extraction record. Expected keys: company,
    pattern_number, side, image, rotation_applied_degrees,
    rotation_source, results (the per-field dict) -- the same shape
    run_test.py has always written to its JSON files. Adds created_at
    itself, so callers don't need to.

    Returns a short human-readable string describing where the record
    went, for callers to print/display -- e.g. "SQLite: /path/to/db"
    or the JSONL file's path.
    """
    record = {"created_at": datetime.now(timezone.utc).isoformat(), **record}

    db_path = os.environ.get(DB_PATH_ENV_VAR)
    if db_path:
        _save_to_db(db_path, record)
        return f"SQLite: {db_path}"

    _save_to_jsonl(record)
    return str(JSONL_PATH)


def _save_to_db(db_path: str, record: dict[str, Any]) -> None:
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(_SCHEMA)
        connection.execute(
            """
            INSERT INTO extraction_results (
                created_at, company, pattern_number, side, image_path,
                rotation_applied_degrees, rotation_source, results_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["created_at"],
                record["company"],
                record["pattern_number"],
                record["side"],
                record["image"],
                record["rotation_applied_degrees"],
                record["rotation_source"],
                json.dumps(record["results"]),
            ),
        )
        connection.commit()
    finally:
        connection.close()


def _save_to_jsonl(record: dict[str, Any]) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    with JSONL_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")
