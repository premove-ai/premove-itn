# premove-itn

Contextual inverse text normalization for English voice-agent transcripts.

Premove ITN is open-weight and open-source. The inference code and final
`model.safetensors` release are available under the MIT license.

## Install and use

The contextual API currently runs from a source checkout. Install the package
and model dependencies with `uv`:

```bash
uv sync --group model
```

Load the frozen `premove-itn/premove-itn` release and normalize text:

```python
from premove_itn import PremoveITN

itn = PremoveITN.from_pretrained()
result = itn.normalize("call me at four thirty")

print(result)  # call me at 04:30
```

The first load downloads the immutable Hub commit for release `v0.1.0`. Later
loads reuse the Hugging Face cache. The loader verifies the resolved commit,
release metadata, base model, and model-file digest before inference. One
`PremoveITN` instance keeps its model and tokenizer in memory across all
`normalize()` calls. The default `device="auto"` selects CUDA, then Apple MPS,
then CPU. Use `device="cpu"`, `device="mps"`, or `device="cuda"` to select a
device explicitly.

PyPI installation and a contextual optional dependency extra are release
packaging work and are not available yet.

The optional [comparison runner](benchmarks/README.md) reproduces First
Evaluation against text-processing-rs and Thutmose. It records release build
metadata, warm-up measurements, per-record latency, and semantic results.
Read the [First Evaluation results](eval/voice_agent_itn/results/first-evaluation/REPORT.md)
for overall results, voice-agent domains, and measurement limitations.

This repository is back at its base layer. It wraps the English realizers from
[`text-processing-rs`](https://github.com/FluidInference/text-processing-rs)
and adds focused local behavior where the upstream library does not provide the
required result.

The runtime has no model dependency. An optional model dependency group owns
the contextual candidate-scoring and structured-training experiment. Exact
maximum-score decoding is available as a separate optional model layer.

## Current scope

The package exposes deterministic realization operations and candidate
enumeration:

```python
from premove_itn import (
    build_candidate_graph,
    build_gold_graph,
    SpanKind,
    normalize_sentence,
    realize,
    realize_options,
    representations_equivalent,
    target_is_reachable,
    tn_normalize,
)

realize(SpanKind.TIME, "four thirty")  # "04:30"
realize_options(SpanKind.CARDINAL, "two")  # ["2"]
realize_options(SpanKind.CARDINAL, "seven eighty eight")  # ["95", "788"]
representations_equivalent(SpanKind.CARDINAL, "12345", "12,345")  # True
representations_equivalent(SpanKind.DATE, "4 march 2014", "2014-03-04")  # True
representations_equivalent(SpanKind.TIME, "04:30 p.m.", "4.30 PM")  # True
representations_equivalent(SpanKind.MONEY, "$1000000", "$1M")  # True
representations_equivalent(SpanKind.DECIMAL, "1,212.3", "1212.30")  # True
representations_equivalent(SpanKind.DIGIT_SEQUENCE, "77152-", "77152")  # True
normalize_sentence("call me at nine one one")
tn_normalize("123")
build_candidate_graph("booking id seven three")
target_is_reachable("booking id seven three", "booking id 73")  # True
build_gold_graph("booking id seven three", "booking id 73")
```

`build_candidate_graph` calls every Rust realizer for every contiguous,
word-or-punctuation token span. It returns half-open token offsets and every
valid edit with its half-open character offsets. Exact no-op realizations are
omitted because `KEEP` owns unchanged text. Results with the same span and
replacement form one candidate with all equivalent `SpanKind` derivations.
Candidates have a stable `(token_start, token_end, replacement)` order. The
graph does not enumerate complete sentence paths. `target_is_reachable` uses
exact dynamic programming over candidate replacements and unchanged source
characters.

`build_gold_graph` uses forward and backward source-target reachability to
retain all states, candidate transitions, and exact `KEEP` transitions that
participate in at least one complete derivation of the expected output. It
returns `None` when the output is unreachable. The packed graph preserves
multiple valid derivations without enumerating complete paths or selecting one
arbitrary gold segmentation.

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

The supported forms are documented in
[`docs/realizer-coverage.md`](docs/realizer-coverage.md). Local Rust extensions
currently cover strict and grouped-ID digit sequences, signed and large cardinals, compositional
dates with calendar checks, spoken clocks including military forms, meridiems,
relative times, durations, and recognized timezones, compositional money with
major and minor currency units and common scale forms, signed decimals with
fractions, named scales, scientific notation, preserved negative zero, and
bounded canonical expansion, and strict measurements with signed/scaled
quantities, fractions, feet-and-inches heights, bare degrees, compound rate
units, and canonical unit output. Phone area-code prefixes, spoken separators,
extensions, and mixed spoken alphanumeric words are also handled. The
electronic and phone delegations also reject unknown trailing words. Ordinals
accept optional
articles, hyphenated/conjunctive words, numeric suffixes, ordinal scales, and
canonical Roman numerals.

ELECTRONIC preserves uppercase `O` inside spelled-letter runs while spoken
`oh` and lowercase `o` retain digit-zero semantics.
Punctuation accepts common aliases such as `full stop`, `bang`, quote names,
paired delimiters, ellipses, and en/em dashes.
Whitelist replacements remain sentence-level and now require safe word
boundaries; ambiguous or non-ASCII input fails closed.
WORD accepts spelled-letter plus number forms and one attached punctuation
mark. It preserves explicitly spoken leading-zero groups inside identifiers,
supports canonical `v`-prefixed multipart versions, and keeps large cardinal
values and conjunctions arithmetic. Explicit CLI flags, spoken underscores,
and `all caps` environment-variable forms are also supported.

`representations_equivalent` is an evaluation helper for `CARDINAL`, `DATE`,
`DECIMAL`, `DIGIT_SEQUENCE`, `MEASUREMENT`, `MONEY`, `ORDINAL`, `PHONE`, and
`TIME`. It compares semantic values while ignoring display
conventions such as grouping, padding, Roman numerals, date field order,
separators, month abbreviations, ordinal suffixes, weekday display, era
punctuation, clock padding, AM/PM punctuation, timezone case, duration
fraction padding, currency placement, grouping, symbols, ISO codes, and scale
abbreviations while preserving currency identity. It does not add these aliases
to the runtime candidate graph. Measurement comparison also preserves
case-sensitive unit identity: `m` is not `min`, and bits are not bytes.

The implementation reuses `text-processing-rs` for its upstream English
parsers. Local Rust grammar and complete-span checks extend those parsers for
the supported edge cases; they do not replace them with a second Python
realizer. Update the coverage document and focused tests whenever a kind
changes.

The optional contextual scorer encodes a padded sentence batch once, pools each
candidate span, projects its multi-hot Rust kinds, and pools the proposed
replacement's first, last, and mean vectors from the encoder's shared input
embedding table. It returns one scalar per candidate. Replacement pooling does
not run the contextual encoder a second time. The scorer consumes the
deterministic candidate graph without changing the runtime realization rules.
The optional training-batch adapter connects scorer outputs to exact
source-target structured loss. `prepare_training_batches` can length-bucket and
chunk prepared examples to reduce padding. The optimizer training loop uses
fused AdamW, writes end-of-epoch and periodic mid-epoch checkpoints, and records
epoch metrics plus the caller-supplied per-kind training distribution in each
checkpoint. Checkpointing requires that distribution so exposure metadata
cannot be omitted accidentally. Checkpoint loading retains the destination
optimizer's `fused` and `foreach` execution settings while restoring legacy
Adam moments. It also verifies that fused Adam step tensors share their
parameter device.
`decode_candidates` uses exact maximum-score interval dynamic programming over
the same source-character path definition as training, then applies the chosen
replacements with the runtime's spacing rules. These model and training layers
remain optional and are not part of the deterministic package API.
Dataset formatting must be
canonicalized before it is compared with semantic candidates.
Corpus audits are regression checks only;
realization rules are generic and must not depend on a particular dataset
sentence or annotation token.

## Datasets and evaluation

Training is complete. The repository retains one production checkpoint and one
frozen VoiceAgent ITN evaluation dataset. See the
[`production model record`](docs/model-provenance.md) for the training
composition, kind distribution, selection evidence, and known limitations.
See the
[`VoiceAgent ITN specification`](docs/evaluations/voice-agent-itn-spec.md) for
the frozen benchmark contract. The benchmark must not be used for training.
The inference-only model artifact and its release/verification procedure are
documented in [`docs/inference-artifact.md`](docs/inference-artifact.md).

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
