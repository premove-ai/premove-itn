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

## Git branching

- Treat `main` as the stable, validated MVP branch.
- Create one `epic/<epic-name>` branch from `main` for each MVP slice.
- Create short-lived topic branches under the epic:
  `<type>/<epic-name>/<short-name>`, using prefixes such as `feat`, `fix`,
  `test`, `docs`, `refactor`, or `chore`.
- Merge topic branches into the epic. Merge an epic into `main` only after its
  MVP acceptance criteria and full validation pass.
- Do not open normal topic pull requests directly against `main`.
- Follow [`docs/branching-strategy.md`](docs/branching-strategy.md) for the
  branch lifecycle, merge modes, and protection rules.

## Verification

```bash
uv run ruff check .
uv run pytest
cargo test --manifest-path rust/Cargo.toml
uv build
```
