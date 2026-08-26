# Premove ITN

Premove ITN currently contains deterministic inverse text normalization
primitives.

## Boundaries

- `rust/` owns every realization rule.
- `src/premove_itn/` is the small Python API over the Rust extension.
- `text-processing-rs` supplies the upstream English ITN and TN parsers.
- Do not add model, training, dataset, BIO-labeling, or contextual-selection
  code without a new architecture decision.

## Rules

- Read the relevant Rust code, direct callers, and tests before changes.
- Make the smallest coherent change.
- Keep one source of truth for supported kinds and realization behavior.
- Python must not duplicate a Rust realizer.
- Preserve unrelated work and generated artifacts.

## Verification

```bash
uv run ruff check .
uv run pytest
cargo test --manifest-path rust/Cargo.toml
uv build
```
