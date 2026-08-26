# premove-itn

Context-aware inverse text normalization for voice agents.

> [!IMPORTANT]
> This repository is a proof of concept. It does not ship a trained model or a
> production normalizer. The frozen benchmark decides whether the project
> proceeds.

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

- Python/Rust runtime package and all 13 forced realizer kinds;
- frozen Golden V1 and deferred-class benchmark records;
- deterministic Dataset V1 construction and grouped train/validation split;
- private 21-label Model V1 training project;
- first DeBERTa-v3 Small training run and validation checkpoint selection.

Next:

- evaluate the selected checkpoint against frozen Golden V1;
- implement checkpoint loading, BIO decoding, and the hybrid normalizer;
- evaluate end-to-end Rust realization;
- benchmark local inference latency;
- ONNX export or Premove integration.

See [ROADMAP.md](ROADMAP.md) for the gated implementation order.

## Runtime and Model V1 scope

The forced Rust realizer supports 13 structural span kinds:

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

Model V1 trains ten of those kinds: `CARDINAL`, `DATE`, `DECIMAL`,
`DIGIT_SEQUENCE`, `ELECTRONIC`, `MEASUREMENT`, `MONEY`, `ORDINAL`, `PHONE`, and
`TIME`. Its versioned contract has 21 labels: `O` plus `B-` and `I-` for those
ten kinds. `PUNCTUATION`, `WHITELIST`, and `WORD` remain deferred from Model V1.

Business concepts such as `ORDER_ID` and `BOOKING_ID` are contextual evidence,
not normalization classes.

## Quick start

Requirements: Python 3.11 or newer and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/premove-ai/premove-itn.git
cd premove-itn
uv sync --all-packages --all-groups
uv run pytest
uv run --package premove-itn-data pytest tools/data_pipeline/tests
uv run --package premove-itn-training pytest tools/training/tests
uv run ruff check .
uv build --package premove-itn
uv build --package premove-itn-data
uv build --package premove-itn-training
```

## Project layout

The repository contains three Python distributions. Both private tool projects
depend on the runtime distribution:

```text
premove-itn-data -----\
                       -> premove-itn
premove-itn-training -/
```

`src/premove_itn/` and `rust/` form the runtime distribution. Dataset parsing,
selection, audit, and enrichment tools live in
`tools/data_pipeline/src/premove_itn_data/`. The runtime does not import or ship
the data pipeline. Model training lives in the private
`tools/training/src/premove_itn_training/` project. It reads only frozen dataset
artifacts and imports the runtime Model V1 label contract. The runtime imports
neither private tool project. The repository-level `data/` directory remains
the shared artifact location and is not part of any wheel.

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

Read [CONTRIBUTING.md](CONTRIBUTING.md) before making a change. Keep frozen
evaluation data independent from training decisions and generated corpora.

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
