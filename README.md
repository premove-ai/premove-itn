# Premove ITN

[![PyPI](https://img.shields.io/pypi/v/premove-itn.svg)](https://pypi.org/project/premove-itn/)
[![CI](https://github.com/premove-ai/premove-itn/actions/workflows/ci.yml/badge.svg)](https://github.com/premove-ai/premove-itn/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/premove-ai/premove-itn.svg)](https://github.com/premove-ai/premove-itn/blob/main/LICENSE)
[![Model weights](https://img.shields.io/badge/model%20weights-Hugging%20Face-yellow)](https://huggingface.co/premove-ai/premove-itn)

Open-source, context-aware inverse text normalization for conversational
voice-agent transcripts, with open weights.

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

The frozen evaluation contains 1,500 synthetic stress cases, including a
dedicated 400-row voice-agent subset. Semantic accuracy checks whether the
structured value is correct while allowing approved formatting differences.
Latency is warm, sequential batch-one inference on an Apple M4 MacBook Air;
every backend received transcript text only. This is not a sample of live
production traffic.

[Benchmark details and reproduction →](eval/voice_agent_itn/results/first-evaluation/REPORT.md)

## Quick start

```bash
pip install premove-itn
```

```python
from premove_itn import PremoveITN

itn = PremoveITN.from_pretrained()
print(itn.normalize("the room code is one oh five"))
# the room code is 105
```

Create one `PremoveITN` instance and reuse it across requests; model loading is
expensive compared with warm normalization.

[Installation, devices, model lifecycle, and CLI usage →](docs/getting-started.md)

## Why this exists

I ran into this problem while building a voice agent. Deepgram gave me
transcripts in spoken form, but the tools behind the agent needed normalized
values. A person can read `one hundred twenty three` and know it means `123`;
an API usually cannot.

My first solution was [`text-processing-rs`](https://github.com/FluidInference/text-processing-rs).
It was extremely fast and handled straightforward normalization well. But some
spoken forms are impossible to normalize correctly without the sentence around
them. `one oh five` might mean `105`, `1:05`, or an identifier. Rules can
generate plausible answers, but they cannot always know which one the speaker
meant.

I then tried [NVIDIA Thutmose](https://catalog.ngc.nvidia.com/orgs/nvidia/nemo/models/itn_en_thutmose_bert/-)
for context-aware ITN, but still found simple failures on structured values used
in tool calls. Premove came from a different idea: generate valid written forms
first, train the model only to choose what fits the context, then let an exact
decoder assemble the final output.

## How Premove ITN works

Premove divides normalization into three jobs:

```text
GENERATE
Rust produces valid written candidates
        ↓
SCORE
DeBERTa encodes the transcript once and scores each candidate
        ↓
DECODE
Exact dynamic programming chooses the best compatible edits
```

The rules decide **what can be written**. The model decides **what fits the
context**. The decoder decides **which edits can coexist**.

[Full runtime architecture →](docs/architecture.md)

## Supported forms

Premove covers English structured values commonly found in voice-agent
transcripts:

| Form | Example |
| --- | --- |
| Numbers | `one hundred twenty` → `120` |
| Times | `four thirty` → `04:30` |
| Money | `twenty dollars` → `$20` |
| Identifiers | `d l t two nine eight two` → `DLT2982` |
| Phone numbers | `eight one four two three one four` → `814-2314` |
| Email | `support at example dot com` → `support@example.com` |

Premove also supports dates, digit sequences, decimals, measurements,
ordinals, versions, URLs, punctuation, and approved abbreviations.
[Full coverage →](docs/rust-candidate-coverage.md)

Premove currently targets English structured text. Non-English normalization,
street addresses, free-form rewriting, and arbitrary application-specific
formats are outside the v0.1.0 scope. Text outside the supported candidate
grammar is left unchanged.

## Limitations

- English only.
- The 435.6M-parameter contextual model is materially slower and heavier than
  deterministic ITN.
- Accuracy is measured on a synthetic stress benchmark, not live production
  traffic.

[Full model and evaluation limitations →](docs/model-card.md#limitations)

## Documentation

- [Getting started](docs/getting-started.md)
- [Architecture](docs/architecture.md)
- [Coverage](docs/rust-candidate-coverage.md)
- [Benchmarks](benchmarks/README.md)
- [Model card](docs/model-card.md)

## Model and license

Open weights are published at
[`premove-ai/premove-itn`](https://huggingface.co/premove-ai/premove-itn).
The source code and model weights are MIT licensed. The scorer uses
[`microsoft/deberta-v3-large`](https://huggingface.co/microsoft/deberta-v3-large)
and derives from the [DeBERTaV3 paper](https://arxiv.org/abs/2111.09543). The
Rust realization layer uses Apache-2.0-licensed
[`text-processing-rs`](https://github.com/FluidInference/text-processing-rs).
See [third-party notices](THIRD_PARTY_NOTICES.md) and
[contributing](CONTRIBUTING.md).
