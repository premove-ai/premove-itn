# Contributing

Premove ITN has one production path: Rust candidate generation, contextual
scoring, and exact decoding behind the `PremoveITN` Python API and CLI. Keep
changes focused and preserve this boundary.

## Pull requests

Create a short-lived branch from `main` and open one focused pull request.
Explain the problem, the protected invariant, and the validation evidence. Do
not push directly to `main`. Merge only after the required checks pass.

Bug reports and tests must use synthetic or de-identified text. Do not include
private transcripts, customer data, credentials, model weights, generated
training corpora, or new outputs derived from the frozen evaluation dataset.

## Validation

Run:

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest
cargo test --manifest-path rust/Cargo.toml
uv build
python scripts/inspect_release_artifact.py dist/*.whl dist/*.tar.gz
uv run python scripts/check_local_links.py
uv run python scripts/check_frozen_boundaries.py
```

Pull requests run the same core checks in GitHub Actions. Changes to package or
inference paths also run the supported platform matrix and frozen prediction
equivalence gate.

Add focused tests for changes to a candidate kind, parser route, decoder, model
loader, or public output. Update
[`docs/rust-candidate-coverage.md`](docs/rust-candidate-coverage.md) and the
README when supported forms change.

## Invariants

- Rust owns deterministic realization and candidate semantics.
- Python does not duplicate a Rust parser or kind definition.
- The model chooses among valid candidates; it does not generate arbitrary
  written text.
- Exact decoding selects compatible, non-overlapping edits.
- Frozen evaluation artifacts are evidence, not tuning input.
- Cleanup and refactoring must not change released normalization behavior.
