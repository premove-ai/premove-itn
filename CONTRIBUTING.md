# Contributing

Keep changes within the deterministic realization boundary.

## Validation

Run:

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest
cargo test --manifest-path rust/Cargo.toml
uv build
```

Add focused tests when a change affects a kind, parser routing, or output.

## Invariants

- An explicit kind selects exactly one deterministic realizer.
- A realizer consumes the complete input or returns `None`.
- Rust owns realization behavior.
- Python does not duplicate a Rust parser.
- New abstractions need a current runtime requirement.
- Contextual selection, model training, and data tooling are outside the
  current repository scope.
