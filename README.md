# Premove ITN

[![PyPI](https://img.shields.io/pypi/v/premove-itn.svg)](https://pypi.org/project/premove-itn/)
[![CI](https://github.com/premove-ai/premove-itn/actions/workflows/ci.yml/badge.svg)](https://github.com/premove-ai/premove-itn/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/premove-ai/premove-itn.svg)](https://github.com/premove-ai/premove-itn/blob/main/LICENSE)
[![Model weights](https://img.shields.io/badge/model%20weights-Hugging%20Face-yellow)](https://huggingface.co/premove-ai/premove-itn)

Open-source, context-aware inverse text normalization for conversational
voice-agent transcripts, with open weights.

## The ambiguity

| System | Output |
| --- | --- |
| Input | `the room code is one oh five` |
| Expected | `the room code is 105` |
| **Premove ITN** | `the room code is 105` ✓ |
| [NVIDIA Thutmose BIO tagger](https://catalog.ngc.nvidia.com/orgs/nvidia/nemo/models/itn_en_thutmose_bert/-) | `the room code is 1 oh 5` ✗ |
| [`text-processing-rs`](https://github.com/FluidInference/text-processing-rs) | `the room code is 01:05` ✗ |

The spoken form is ambiguous. Context tells us that `one oh five` is an
identifier, not a time.

## Results

On our frozen benchmark, Premove reaches **99.50% semantic accuracy on the
voice-agent subset**, compared with 68.25% for `text-processing-rs` and 67.00%
for NVIDIA Thutmose.

| Backend | Voice-agent semantic | Overall semantic | Mean warm latency |
| --- | ---: | ---: | ---: |
| **Premove ITN** | **99.50%** | **89.70%** | 56.49 ms |
| [NVIDIA Thutmose](https://catalog.ngc.nvidia.com/orgs/nvidia/nemo/models/itn_en_thutmose_bert/-) | 67.00% | 59.39% | 15.98 ms |
| [`text-processing-rs`](https://github.com/FluidInference/text-processing-rs) | 68.25% | 55.79% | **0.14 ms** |

The benchmark is a frozen, balanced synthetic stress suite with 1,500 rows and
includes a dedicated 400-row voice-agent subset. Semantic accuracy checks
whether the structured value is correct while allowing approved formatting
differences. Latency is warm, sequential batch-one inference on an Apple M4
MacBook Air; every backend received transcript text only. This is a synthetic
stress benchmark, not a sample of live production traffic.

[Full report](eval/voice_agent_itn/results/first-evaluation/REPORT.md) ·
[Detailed results](eval/voice_agent_itn/results/first-evaluation/DETAILS.md) ·
[Reproduce the benchmark](benchmarks/README.md)

## Quick start

Install from PyPI:

```bash
pip install premove-itn
```

```python
from premove_itn import PremoveITN

itn = PremoveITN.from_pretrained()

print(itn.normalize("the room code is one oh five"))
# the room code is 105
```

`from_pretrained()` downloads the frozen model weights from
[`premove-ai/premove-itn`](https://huggingface.co/premove-ai/premove-itn) on
first use and caches them locally.

Create one `PremoveITN` instance and reuse it across requests. Model loading is
expensive; warm normalization calls are much faster.

For reproducible deployments, pin the package version:

```bash
pip install premove-itn==0.1.0
```

## Why this exists

I ran into this problem while building a voice agent. Deepgram gave me
transcripts in spoken form, but the tools behind the agent needed normalized
values. A person can read `one hundred twenty three` and know it means `123`; an
API usually cannot.

My first solution was [`text-processing-rs`](https://github.com/FluidInference/text-processing-rs).
It was extremely fast and handled straightforward normalization well. But some
spoken forms are impossible to normalize correctly without the sentence around
them. `one oh five` might mean `105`, `1:05`, or something else entirely. Rules
can generate plausible answers, but they cannot always know which one the
speaker meant.

That sent me looking for context-aware ITN. I tried [NVIDIA Thutmose](https://catalog.ngc.nvidia.com/orgs/nvidia/nemo/models/itn_en_thutmose_bert/-),
but on the structured values I cared about for tool calls, I still found
surprisingly simple failures.

Premove came from a different idea: do not ask the model to perform the entire
normalization. Generate the valid written forms first, then train the model
only to decide which one fits the context. An exact decoder handles the rest.

## How Premove ITN works

Premove splits normalization into three steps: **generate, score, decode**.

```text
input
enter the digits four one seven two zero one two
                         │
                         ▼
1. GENERATE

Rust enumerates valid written forms for spans in the transcript.

[four]                               → 4
[four one]                           → 41 | 04:01
[one seven two]                      → 172 | 17:02
[four one seven two zero one two]    → 4172012 | 417-2012
                         │
                         ▼
2. SCORE

DeBERTa encodes the transcript once.

Each candidate gets one score from:
contextual source span + candidate type + proposed replacement
                         │
                         ▼
3. DECODE

Candidates can overlap, so Premove uses exact dynamic programming
to find the highest-scoring compatible path through the transcript.
                         │
                         ▼
output
enter the digits 4172012
```

The rules decide **what can be written**. The model decides **what fits the
context**. The decoder decides **which edits can coexist**.

## Supported forms

Premove covers English structured values commonly needed by voice agents:

| Form | Example |
| --- | --- |
| Numbers | `one hundred twenty` → `120` |
| Digit sequences | `zero eight two zero six three` → `082063` |
| Times | `four thirty` → `04:30` |
| Dates | `march fifth twenty twenty four` → `march 5 2024` |
| Money | `twenty dollars` → `$20` |
| Decimals | `one point five` → `1.5` |
| Measurements | `ninety percent` → `90 %` |
| Ordinals | `the eighth` → `8th` |
| Phone numbers | `eight one four two three one four` → `814-2314` |
| Email and URLs | `support at example dot com` → `support@example.com` |
| Versions and identifiers | `v three dot one dot nine` → `v3.1.9` |
| Punctuation | `comma` → `,` |
| Abbreviations | `doctor` → `dr.` |

### Not supported

Premove currently targets English structured text. It does not provide
first-class normalization for non-English speech, street addresses, free-form
rewriting, or arbitrary application-specific formats.

Text outside the supported candidate grammar is left unchanged.

See the [full candidate coverage](docs/rust-candidate-coverage.md) for the exact
forms supported by each realizer.

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
- Independent human gold adjudication and broader contamination checks are
  outside the v0.1.0 release gate and remain separate follow-up work.
- CUDA, Windows, macOS Intel, Linux ARM64, and other accelerators are not
  validated v0.1.0 support claims.

## Documentation

- [Getting started](docs/getting-started.md)
- [Architecture](docs/architecture.md)
- [Candidate coverage](docs/rust-candidate-coverage.md)
- [Model card](docs/model-card.md)
- [Model provenance](docs/model-provenance.md)
- [Inference artifact](docs/inference-artifact.md)
- [Platform support](docs/platform-support.md)
- [Benchmark reproduction](benchmarks/README.md)
- [Contributing](CONTRIBUTING.md)

## Model, citation, and license

### Model and weights

The current public release is **v0.1.0**:

- [PyPI package](https://pypi.org/project/premove-itn/)
- [GitHub release](https://github.com/premove-ai/premove-itn/releases/tag/v0.1.0)
- [Hugging Face model](https://huggingface.co/premove-ai/premove-itn/tree/v0.1.0)

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

### Citation

The contextual scorer uses
[`microsoft/deberta-v3-large`](https://huggingface.co/microsoft/deberta-v3-large)
at a pinned revision. The architecture derives from the
[DeBERTaV3 paper](https://arxiv.org/abs/2111.09543).

### License and attribution

Premove ITN source code and model weights are MIT licensed. The contextual
scorer uses [`microsoft/deberta-v3-large`](https://huggingface.co/microsoft/deberta-v3-large)
at a pinned revision. The architecture derives from the
[DeBERTaV3 paper](https://arxiv.org/abs/2111.09543).

The Rust realization layer uses
[`text-processing-rs`](https://github.com/FluidInference/text-processing-rs),
which is Apache-2.0 licensed. Required third-party licenses and notices are in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and [`LICENSES/`](LICENSES/).
