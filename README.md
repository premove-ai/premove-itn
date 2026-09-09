# Premove ITN

Contextual inverse text normalization for English voice-agent transcripts.

Premove turns spoken-form ASR text into structured written text:

```text
call me at four thirty  →  call me at 04:30
the total is twenty dollars  →  the total is $20
```

Deterministic Rust realizers propose valid written forms. A
DeBERTa-v3-large contextual scorer uses the complete transcript to choose among
them, and an exact decoder produces compatible, non-overlapping edits.

On the retained First Evaluation, Premove achieved **99.50% semantic accuracy
on 400 dedicated voice-agent rows** and **89.70% overall semantic accuracy on
1,500 frozen stress-suite rows**. Mean warm request latency was **56.49 ms** on
an Apple M4 using MPS and batch size one. See the [full First Evaluation report](eval/voice_agent_itn/results/first-evaluation/REPORT.md).

## Installation

The package is prepared for:

```bash
pip install premove-itn
```

The PyPI v0.1.0 release is not published yet. It will be published after
automated release gates and platform validation are complete. Contributors can
currently build and install the release wheel locally:

```bash
uv build
pip install dist/premove_itn-*.whl
```

The model weights are downloaded separately from the frozen
[`premove-ai/premove-itn`](https://huggingface.co/premove-ai/premove-itn)
Hugging Face release on first use.

## Python quickstart

```python
from premove_itn import PremoveITN

itn = PremoveITN.from_pretrained()

print(itn.normalize("call me at four thirty"))
# call me at 04:30
```

Create one `PremoveITN` instance and keep it resident:

```python
texts = [
    "call me at four thirty",
    "the total is twenty dollars",
    "the last account digits are zero eight two zero six three",
]

for text in texts:
    print(itn.normalize(text))
```

Model initialization is expensive. Warm normalization calls on an existing
instance are much faster than loading a new instance for each request.

## Command-line interface

Normalize one transcript:

```bash
premove-itn "call me at four thirty"
```

```text
call me at 04:30
```

Process newline-delimited transcripts:

```bash
printf 'call me at four thirty\nthe total is twenty dollars\n' | premove-itn
```

```text
call me at 04:30
the total is $20
```

Stdin mode loads the model once, then processes every input line in order. It
is the correct CLI mode for files and long-lived transcript pipelines.

```text
stdin process → one model load → line 1 → line 2 → line 3 → ...
```

The CLI supports `--device auto`, `--device cpu`, `--device mps`,
`--device cuda`, `--version`, and `--help`. Normal stdout contains only
normalized transcripts. Diagnostics and errors use stderr.

## Why contextual ITN?

Obvious spoken values can often be normalized with fixed rules:

```text
twenty dollars → $20
```

The harder cases have several plausible written forms. For example, a number
sequence can represent a time, an identifier, a count, or part of a phone
number. Premove generates valid alternatives with deterministic rules, then
uses the surrounding sentence to score the intended interpretation.

```text
ASR transcript
      ↓
structured Rust candidates
      ↓
full-sentence contextual scores
      ↓
exact interval decoding
      ↓
written transcript
```

## How it works

```text
Spoken ASR text
      │
      ▼
Rust candidate generation
      │  TIME / MONEY / PHONE / ID / ...
      ▼
DeBERTa-v3-large contextual scorer
      │
      ▼
Exact maximum-score decoder
      │
      ▼
Written transcript
```

**Rust candidate generation** deterministically proposes valid written
representations. It does not choose the intended interpretation.

**Contextual scoring** uses the complete sentence to score competing
candidates.

**Exact decoding** selects a compatible set of scored replacements without
overlapping edits.

The public runtime has one implementation path. Both the Python API and CLI
call `PremoveITN`; neither duplicates candidate generation or decoding.

## Performance

### Dedicated voice-agent rows

These results use only the 400 `group=voice_agent` rows: 50 rows in each of
eight voice-agent domains.

| Backend | Correct entities | Semantic accuracy | Mean latency |
| --- | ---: | ---: | ---: |
| **Premove ITN** | **398/400** | **99.50%** | 57.41 ms |
| Thutmose | 268/400 | 67.00% | 16.04 ms |
| text-processing-rs | 273/400 | 68.25% | 0.15 ms |

### Overall frozen benchmark

| Backend | Semantic accuracy | Strict exact | Mean latency |
| --- | ---: | ---: | ---: |
| **Premove ITN** | **89.70%** | **40.53%** | 56.49 ms |
| Thutmose | 59.39% | 22.13% | 15.98 ms |
| text-processing-rs | 55.79% | 16.53% | 0.14 ms |

Premove led semantic accuracy in this evaluation. It did not lead latency.

### Methodology

- The dataset is a frozen, balanced synthetic stress suite with 1,500 rows and
  1,640 declared entities.
- The voice-agent result uses only the 400 dedicated voice-agent rows. Generic
  multi-entity rows do not enter domain claims.
- Semantic entity accuracy is the main structured-value metric. It ignores
  allowed display differences while preserving value, currency, unit, digit
  order, phone form, and identifier case policy.
- Strict exact match scores the full canonical output and is reported
  separately.
- Latency used sequential batch-one requests on an Apple M4 MacBook Air, MPS
  completion, an optimized Rust extension, and eight Rayon workers.
- Models were initialized and warmed before request latency was measured.
- The benchmark is not an IID sample of live production traffic.
- The backend evaluation was blind: every backend received only transcript text
  and did not receive gold spans, categories, domains, difficulty, or expected
  output. Independent human gold adjudication is a separate task and remains
  pending.

Read the [full report](eval/voice_agent_itn/results/first-evaluation/REPORT.md),
[detailed tables](eval/voice_agent_itn/results/first-evaluation/DETAILS.md), and
[reproduction guide](benchmarks/README.md).

## Model lifecycle and latency

Three different costs must not be combined:

| Phase | Meaning |
| --- | --- |
| First download | Fetches about 1.6 GB from Hugging Face; duration depends on the network. |
| Cached initialization | Resolves cached files, builds the model, loads weights, and transfers the model to the device; this took several seconds on the tested system. |
| Warm normalization | Processes one transcript after loading and warm-up; the retained release-build mean was 56.49 ms on Apple M4/MPS. |

The benchmark excludes model download and initialization. A one-shot CLI
timing includes process startup and model initialization, so it is not
comparable to warm request latency. Services and transcript streams should load
one normalizer and reuse it.

## Device selection

`device="auto"` selects CUDA when available, then Apple MPS, then CPU. Python
users can pass `device="cpu"`, `device="mps"`, or `device="cuda"` to
`PremoveITN.from_pretrained()`. The CLI exposes the same choices through
`--device`.

The current release candidate has been validated end-to-end on macOS Apple
Silicon with Python 3.11 and MPS. CPU, CUDA, other operating systems, and other
Python versions will be validated before the public release. API availability
does not imply that a platform has completed release validation.

## Limitations

- English only.
- A 435.6M-parameter DeBERTa-v3-large contextual scorer.
- About a 1.6 GB first model download.
- Multi-second model initialization.
- A custom candidate-scoring architecture; it is not a generic
  `AutoModel.from_pretrained()` model.
- A synthetic stress benchmark, not observed live-traffic accuracy.
- Weaker measured categories include URL, MONEY, CARDINAL, TIME,
  REFERENCE_ID, and VERSION. See the report for exact per-category results.
- Independent human gold adjudication and broader contamination checks remain
  incomplete.
- End-to-end release validation currently covers only the environment stated
  above.

## Model and weights

The public inference artifact is
[`premove-ai/premove-itn`](https://huggingface.co/premove-ai/premove-itn),
release `v0.1.0`, at the immutable commit
`80bda5e2e1fe9542aa628597090242df57c1a157`. `PremoveITN.from_pretrained()`
uses that commit by default and verifies the resolved revision, release
metadata, base model, and model-file digest before inference.

The artifact is inference-only. It excludes optimizer state, scheduler state,
training counters, training data, and evaluation rows. See the
[artifact release record](docs/inference-artifact.md) and
[training provenance](docs/model-provenance.md).

## Training data and model selection

The production DeBERTa-v3-large scorer was trained on **418,000 examples**
across **15 training kinds**. Training ran in three sequential stages:

| Stage | Examples | Role |
| --- | ---: | --- |
| Google text normalization | 378,000 | General English ITN coverage |
| Conversational adaptation | 20,000 | Spoken-dialogue style and context |
| Structured-value adaptation | 20,000 | Identifiers, phones, electronic values, and hard context |
| **Total** | **418,000** | |

The frozen 1,500-row VoiceAgent benchmark was not used for training or model
selection. The bulk training corpora were removed after this composition was
recorded; the table below is the retained distribution of every example seen
by the selected checkpoint. Counts are examples, not individual spans.

| Kind | Google | Conversational | Structured | Total |
| --- | ---: | ---: | ---: | ---: |
| CARDINAL | 32,472 | 853 | 876 | 34,201 |
| DATE | 78,643 | 315 | 398 | 79,356 |
| DECIMAL | 12,033 | 465 | 701 | 13,199 |
| DIGIT_SEQUENCE | 3,258 | 398 | 674 | 4,330 |
| ELECTRONIC | 0 | 12 | 3,030 | 3,042 |
| KEEP | 100,000 | 11,429 | 4,150 | 115,579 |
| MEASUREMENT | 16,647 | 300 | 349 | 17,296 |
| MONEY | 23,688 | 258 | 271 | 24,217 |
| MULTI | 82,490 | 3,259 | 2,078 | 87,827 |
| ORDINAL | 21,484 | 258 | 271 | 22,013 |
| PHONE | 2,761 | 260 | 1,994 | 5,015 |
| PUNCTUATION | 624 | 258 | 272 | 1,154 |
| TIME | 3,894 | 1,879 | 876 | 6,649 |
| WHITELIST | 6 | 17 | 17 | 40 |
| WORD | 0 | 39 | 4,043 | 4,082 |
| **Total** | **378,000** | **20,000** | **20,000** | **418,000** |

The final 20,000-example structured stage contained candidate-bearing KEEP
examples (4,000), conversational replay (2,850), Google replay (3,000),
targeted electronic values (3,000), hard KEEP cases (150), identifiers (4,000),
numeric IDs (750), and phone values (2,250). The source material combined the
Google Text Normalization training partition with conversational examples from
SLURP, Schema-Guided Dialogue, SpokenWOZ, and Taskmaster-1.

The Google stage used AdamW with learning rate `2e-5`, weight decay `0.01`,
batch size `8`, and length bucketing. Both adaptation stages used AdamW with
learning rate `5e-6`, weight decay `0.01`, batch size `8`, one epoch, and fresh
optimizer state; the structured stage used microbatch size `2`. The final
structured-value checkpoint was selected for the best product-relevant balance
of structured-value and conversational results. Full selection evidence and
development results are in the [production model record](docs/model-provenance.md).

## Reproducibility

The repository retains the frozen VoiceAgent ITN dataset, detailed per-record
First Evaluation outputs, comparison adapters, metric summaries, and release
artifact checks. The frozen benchmark was not used for training or checkpoint
selection. Bulk training corpora were deleted after their composition and kind
distribution were documented.

The benchmark runner is optional and is not part of normal inference. See
[`benchmarks/README.md`](benchmarks/README.md).

## Repository

| Path | Purpose |
| --- | --- |
| `src/premove_itn/` | Python runtime and public API |
| `rust/` | Deterministic realization rules |
| `benchmarks/` | Optional comparison and artifact-verification tools |
| `eval/` | Frozen benchmark and retained results |
| `examples/` | Minimal Python and stdin examples |
| `scripts/` | Release checks and development utilities |
| `tests/` | Python regression tests |
| `docs/` | Architecture, provenance, and evaluation documentation |

Deterministic realization coverage and lower-level APIs are documented in
[`docs/realizer-coverage.md`](docs/realizer-coverage.md). Contributor workflow
is documented in [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Development

Requirements: Python 3.11 or newer, Rust, and
[`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --all-groups
uv run ruff format --check .
uv run ruff check .
uv run pytest
cargo test --manifest-path rust/Cargo.toml
uv build
```

## License and attribution

Premove ITN source code and model weights are MIT licensed. The contextual
scorer uses [`microsoft/deberta-v3-large`](https://huggingface.co/microsoft/deberta-v3-large)
at a pinned revision. The architecture derives from the
[DeBERTaV3 paper](https://arxiv.org/abs/2111.09543).

The Rust realization layer uses
[`text-processing-rs`](https://github.com/FluidInference/text-processing-rs),
which is Apache-2.0 licensed. Required third-party licenses and notices are in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and [`LICENSES/`](LICENSES/).
