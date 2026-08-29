# premove-itn

Deterministic inverse text normalization primitives for voice agents.

This repository is back at its base layer. It wraps the English realizers from
[`text-processing-rs`](https://github.com/FluidInference/text-processing-rs)
and adds focused local behavior where the upstream library does not provide the
required result.

It contains no model, BIO labeling code, training pipeline, or dataset.

## Current scope

The package exposes four operations:

```python
from premove_itn import (
    SpanKind,
    normalize_sentence,
    realize,
    realize_options,
    representations_equivalent,
    tn_normalize,
)

realize(SpanKind.TIME, "four thirty")  # "04:30"
realize_options(SpanKind.CARDINAL, "two")  # ["2"]
realize_options(SpanKind.CARDINAL, "seven eighty eight")  # ["95", "788"]
representations_equivalent(SpanKind.CARDINAL, "12345", "12,345")  # True
representations_equivalent(SpanKind.DATE, "4 march 2014", "2014-03-04")  # True
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

`realize_options` returns deterministic semantic interpretations for the same
complete input. It does not add grouping, padding, Roman numerals, or other
rendering aliases. `CARDINAL` returns a plain ASCII decimal integer and adds
an alternate aviation reading only when it has a different numeric value.

`representations_equivalent` is an evaluation helper for `CARDINAL` and
`DATE`. It compares semantic values while ignoring display conventions such as
grouping, padding, Roman numerals, date field order, separators, month
abbreviations, ordinal suffixes, weekday display, and era punctuation. It does
not add these aliases to the runtime candidate graph.

The local Rust code adds strict `DIGIT_SEQUENCE` realization, semantic
`CARDINAL` coverage, compositional date and year handling, calendar validity
checks, and more useful spoken clock handling. Other realization delegates to
`text-processing-rs`.

There is deliberately no contextual decision layer. A future candidate
lattice, scorer, and decoder remain separate work. Dataset formatting must be
canonicalized before it is compared with these semantic candidates.

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
