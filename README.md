# premove-itn

Deterministic inverse text normalization primitives for voice agents.

This repository is back at its base layer. It wraps the English realizers from
[`text-processing-rs`](https://github.com/FluidInference/text-processing-rs)
and adds focused local behavior where the upstream library does not provide the
required result.

It contains no model, BIO labeling code, training pipeline, or dataset.

## Current scope

The package exposes three operations:

```python
from premove_itn import SpanKind, normalize_sentence, realize, tn_normalize

realize(SpanKind.TIME, "four thirty")  # "04:30"
normalize_sentence("call me at nine one one")
tn_normalize("123")
```

`realize` forces one parser to consume the complete input. The supported kinds
are:

```text
DIGIT_SEQUENCE  CARDINAL     TIME       DATE
MONEY           DECIMAL      PHONE      ELECTRONIC
MEASUREMENT     ORDINAL      PUNCTUATION
WHITELIST       WORD
```

The local Rust code currently adds strict `DIGIT_SEQUENCE` realization and
more useful spoken clock handling. All other kind realization delegates to
`text-processing-rs`.

There is deliberately no contextual decision layer. A future candidate
lattice, scorer, and decoder must be designed as separate work after the
candidate-oracle experiment proves that the deterministic realizers can reach
the required outputs.

## Development

Requirements: Python 3.11 or newer, Rust, and
[`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --all-groups
uv run ruff check .
uv run pytest
cargo test --manifest-path rust/Cargo.toml
uv build
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) and the
[`branching strategy`](docs/branching-strategy.md) for the development
workflow.

## License

Original code is MIT licensed. Bundled third-party code retains the terms in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
