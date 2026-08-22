# premove-itn

Context-aware inverse text normalization for voice agents.

> [!IMPORTANT]
> This repository is a proof of concept. It does not yet provide a trained model
> or production normalizer. The benchmark decides whether the project proceeds.

## Why this exists

The same spoken phrase can require different written forms:

```text
I'll arrive at four thirty.       -> TIME           -> 4:30
My order number is four thirty.   -> DIGIT_SEQUENCE -> 430
```

A fixed parser order cannot use the words around the value. `premove-itn` tests
a hybrid design:

```text
transcript
    -> contextual BIO tagger
    -> structural span class
    -> forced deterministic Rust realizer
    -> normalized text plus exact edit provenance
```

The learned model finds spans and selects structural classes. It never writes a
normalized value. Existing `text-processing-rs` parsers perform cardinal, time,
date, money, decimal, and phone realization. A small local parser handles
one-by-one digit sequences.

## Project hypothesis

The project tests one question:

> Can a small contextual encoder select the correct span and deterministic
> parser on difficult voice-agent utterances better than fixed parser priority?

The proof succeeds only if it clearly improves hard contextual cases without
introducing destructive edits or unacceptable local latency.

## Current status

Completed:

- repository and contribution setup;
- initial package scaffold;
- architecture and data contracts;
- repository-local engineering skills.

Not yet implemented:

- reviewed benchmark records;
- Rust/PyO3 realization adapter;
- synthetic training-data tooling;
- contextual tagger, decoder, and normalizer;
- evaluation and latency reports;
- ONNX export or Premove integration.

See [ROADMAP.md](ROADMAP.md) for the gated implementation order.

## MVP scope

The proof uses all 13 structural span kinds supported by the forced Rust
realizer:

```text
DIGIT_SEQUENCE
CARDINAL
TIME
DATE
MONEY
DECIMAL
PHONE
ELECTRONIC
MEASUREMENT
ORDINAL
PUNCTUATION
WHITELIST
WORD
```

`O` is the BIO label for text outside a normalized span. Business concepts such
as `ORDER_ID` and `BOOKING_ID` are contextual evidence, not normalization
classes.

## Quick start

Requirements: Python 3.11 or newer and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/premove-ai/premove-itn.git
cd premove-itn
uv sync --all-groups
uv run pytest
uv run ruff check .
uv build
```

The build commands will gain Rust/Maturin steps when the approved package-boundary
phase is implemented.

## Repository layout

```text
src/premove_itn/   Python package and public contracts
rust/              PyO3 adapter and deterministic class realizers
data/              reviewed benchmarks and future template families
scripts/           future training and evaluation entry points
tests/             behavior tests at approved interfaces
docs/              architecture documentation
.agents/skills/    repository-local engineering skills
```

## Safety properties

- A model prediction alone cannot cause a rewrite.
- The selected deterministic parser must accept the complete span.
- Rejected or uncertain spans remain unchanged.
- Text outside successful edits remains byte-for-byte identical.
- Public spans always refer to original-text character offsets.
- Generated training templates never enter the reviewed benchmark.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) before making a change. The first data
milestone is a small, de-identified, human-reviewed benchmark. Do not start
large-scale generation or model training before that judge is approved.

## Acknowledgements

`premove-itn` is designed to reuse
[`text-processing-rs`](https://github.com/FluidInference/text-processing-rs),
created and maintained by
[`FluidInference`](https://github.com/FluidInference). It provides the tested
deterministic English ITN/TN parsers that this project will call for cardinal,
time, date, money, decimal, and telephone realization. Its sentence normalizer
also serves as the baseline for this proof of concept.

`premove-itn` adds contextual span classification, safe class routing,
provenance, abstention, streaming-prefix evaluation, and the missing strict
digit-sequence behavior. It does not claim authorship of the upstream grammars
or parsers.

`text-processing-rs` is distributed under the
[`Apache-2.0` license](https://github.com/FluidInference/text-processing-rs/blob/v0.3.0/LICENSE)
and remains subject to its own copyright and license terms. We are grateful to
its maintainers and contributors for making this work available.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the complete bundled
license and attribution record.

## License

The original `premove-itn` code is licensed under [MIT](LICENSE). Bundled
third-party software retains the terms listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
