# premove-itn

Context-aware inverse text normalization experiments for voice agents.

> [!IMPORTANT]
> This repository is a proof of concept. It does not provide a production-ready
> normalizer yet. The first goal is to prove that a small contextual model can
> select the correct deterministic parser for ambiguous spoken values.

## Why this exists

The phrase `four thirty` can mean different things:

```text
I'll arrive at four thirty.       -> TIME           -> 4:30
My order number is four thirty.   -> DIGIT_SEQUENCE -> 430
```

A fixed parser priority cannot use the words around the value. This project
tests a hybrid design:

```text
transcript
    -> contextual BIO tagger (find the span and select its class)
    -> deterministic class-specific realizer (produce the exact value)
    -> span-only text rewrite
```

The neural model never writes the normalized value. It only selects a class
such as `TIME`, `DATE`, `CARDINAL`, or `DIGIT_SEQUENCE`. Deterministic code does
the conversion. Low-confidence predictions can abstain and preserve the input.

## Current status

The repository contains the experiment foundation:

- the initial semantic class contract;
- the labeled-span data contract;
- architecture and data-format documentation;
- contribution guidance and issue templates;
- an accuracy-gated experiment roadmap.

Model training, calibration, ONNX export, and Rust realization are planned but
not implemented. See [ROADMAP.md](ROADMAP.md).

## Quick start

Requirements: Python 3.11 or newer and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/premove-ai/premove-itn.git
cd premove-itn
uv sync --all-groups
uv run pytest
uv run ruff check .
```

The first contribution milestone is a small, human-reviewed benchmark. Do not
start model training or large-scale data generation before that judge is fixed.

## Development commands

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv build
```

## Project layout

```text
src/premove_itn/   package and data contracts
tests/             focused tests added with behavior
docs/              architecture documentation
data/README.md     rules for generated and reviewed data
```

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) before making a change. Use the issue
forms for bugs and proposals. Pull requests should explain the protected
invariant and include focused tests for behavior changes.

## Success criteria

The proof continues only if a human-reviewed, held-out benchmark shows:

- at least 95% exact span-and-class accuracy on important transformed values;
- very low false-positive conversion, especially on ID/time/date collisions;
- byte-for-byte preservation of text outside predicted spans;
- an initial ONNX CPU p95 below about 50 ms on target hardware.

Precision is more important than coverage. Leaving ambiguous text unchanged is
safer than applying the wrong rewrite.

## License

[MIT](LICENSE)
