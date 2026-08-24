# Premove ITN

Premove ITN is a context-aware inverse text normalization runtime and its
private dataset pipeline.

## Boundaries

- `src/premove_itn/` and `rust/` are the runtime distribution.
- `tools/data_pipeline/` owns corpus parsing, selection, audit, enrichment, and
  dataset CLI commands.
- The dependency direction is `premove-itn-data -> premove-itn`. Runtime code
  must never import or package data-pipeline code.
- `data/` stores local source and generated artifacts. Do not package or commit
  generated corpora or model artifacts.

## Rules

- Read the relevant code, callers, and tests before changing behavior.
- Make the smallest coherent change. Keep one source of truth.
- Rust owns deterministic realization. Python may select a `SpanKind`, but it
  must not duplicate a Rust realizer.
- Keep dataset generation deterministic. Record stable provenance and seeds.
- Validate offsets, spans, BIO labels, and `expected_text` after record changes.
- Do not use an LLM to assign ground-truth labels.
- Preserve unrelated work and data artifacts.

## Verification

Run focused tests first, then:

```bash
uv run ruff check .
uv run pytest
uv run --package premove-itn-data pytest tools/data_pipeline/tests
cargo test --manifest-path rust/Cargo.toml
uv build --package premove-itn
uv build --package premove-itn-data
```
