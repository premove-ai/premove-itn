# Contributing

Preserve the production path from Rust candidate generation through contextual
scoring and exact decoding. Keep one implementation behind the Python API and
CLI.

## Branching and pull requests

Use short-lived topic branches under one epic branch for each MVP slice:

```text
main
└── epic/<epic-name>
    └── <type>/<epic-name>/<short-name>
```

Open topic pull requests against the epic. Squash merge them into the epic.
Open the epic pull request against `main` only after the MVP acceptance
criteria and full validation pass. Merge that pull request with a merge commit
to preserve the MVP boundary. Do not push directly to `main` or send normal
topic pull requests to `main`.

See [`docs/branching-strategy.md`](docs/branching-strategy.md) for branch
naming, commands, merge gates, and recommended GitHub protection settings.

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

Pull requests also run these checks in GitHub Actions. The normal workflow
builds and installs a release wheel in a clean environment, verifies the
release Rust metadata, and checks `premove-itn --help` and
`premove-itn --version` outside the source checkout. It does not download the
large Hub model. The model-backed release smoke test is available from the
Actions `workflow_dispatch` menu.

Add focused tests when a change affects a kind, parser routing, or output.
Update [`docs/realizer-coverage.md`](docs/realizer-coverage.md) and the README
when a kind's supported forms change.

## Invariants

- An explicit kind selects exactly one deterministic realizer.
- A realizer consumes the complete input or returns `None`.
- Rust owns realization behavior.
- Python does not duplicate a Rust parser.
- New abstractions need a current runtime requirement.
- Contextual selection is the public runtime above the deterministic Rust
  layer. Model training must consume the Rust-owned candidate and graph
  interfaces and must not duplicate realization rules in Python.
- Dataset tooling remains an offline experiment concern. It must not become a
  runtime dependency.
