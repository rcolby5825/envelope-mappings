# Envelope Mappings

A generic Python project template using a `src/` layout.

## Requirements

- Python 3.10 or newer

## Setup

Create and activate a virtual environment, then install the package in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Development

Run the tests:

```bash
python -m unittest discover -s tests
```

Run linting when the development extras are installed:

```bash
ruff check .
```

The package source lives in `src/envelope_mappings/`; add application code there and keep tests under `tests/`.

## Spring Boot integration

The gRPC contract is in [`proto/envelope_mappings.proto`](proto/envelope_mappings.proto). Generate Java and Spring gRPC stubs from this file in the Spring Boot repository. Send PNG or JPEG bytes in `envelope_image`; `logo_region` is optional and falls back to the full envelope image when omitted.
