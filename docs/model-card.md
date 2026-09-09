---
library_name: premove-itn
language:
  - en
base_model:
  - microsoft/deberta-v3-large
tags:
  - inverse-text-normalization
  - text-normalization
  - speech-processing
  - voice-agents
  - structured-prediction
  - candidate-ranking
  - safetensors
  - custom-code
license: mit
---

# Premove ITN v0.1.0

Premove ITN is an open-weight contextual inverse text normalization system for
English voice-agent transcripts. It turns spoken-form ASR text into structured
written text:

```text
call me at four thirty  →  call me at 04:30
the total is twenty dollars  →  the total is $20
```

Deterministic Rust realizers propose valid written forms. A
DeBERTa-v3-large contextual scorer uses the complete sentence to score those
candidates, and an exact decoder selects compatible, non-overlapping edits.

This repository contains the frozen, inference-only v0.1.0 model artifact. It
does not contain optimizer state, scheduler state, training counters, training
data, or evaluation rows. Source code and retained evaluation evidence are in
[`premove-ai/premove-itn`](https://github.com/premove-ai/premove-itn).

## Loading the model

This is a custom candidate-scoring architecture. Do not load it with
`AutoModel.from_pretrained()`.

```python
from premove_itn import PremoveITN

itn = PremoveITN.from_pretrained()
print(itn.normalize("call me at four thirty"))
# call me at 04:30
```

Install the package with `pip install premove-itn==0.1.0`. Create one
`PremoveITN` instance and reuse it; model initialization is expensive compared
with warm normalization.

`device="auto"` selects CUDA when available, then Apple MPS, then CPU. The
Release wheels are validated on macOS 14+ arm64 and `manylinux_2_28` x86_64
for Python 3.11–3.13. Real frozen-model inference is validated on Apple
Silicon MPS and Linux CPU. Other environments require release validation. The source repository retains the current
[platform support matrix](https://github.com/premove-ai/premove-itn/blob/main/docs/platform-support.md).

## Architecture

```text
Spoken ASR text
      ↓
deterministic Rust candidates
      ↓
DeBERTa-v3-large contextual scores
      ↓
exact maximum-score decoder
      ↓
written transcript
```

The scorer has 435,594,145 parameters. The artifact contains the complete
trained state in `model.safetensors`, the pinned DeBERTa configuration, and the
tokenizer files required by the release.

Supported candidate kinds are `DIGIT_SEQUENCE`, `CARDINAL`, `TIME`, `DATE`,
`MONEY`, `DECIMAL`, `PHONE`, `ELECTRONIC`, `MEASUREMENT`, `ORDINAL`,
`PUNCTUATION`, `WHITELIST`, and `WORD`.

## First Evaluation

The retained First Evaluation used a frozen, balanced synthetic stress suite.
Semantic entity accuracy is the primary structured-value metric. Strict exact
match separately measures the complete canonical output.

### Dedicated voice-agent rows

| Backend | Correct entities | Semantic accuracy | Mean latency |
| --- | ---: | ---: | ---: |
| **Premove ITN** | **398/400** | **99.50%** | 57.41 ms |
| Thutmose | 268/400 | 67.00% | 16.04 ms |
| text-processing-rs | 273/400 | 68.25% | 0.15 ms |

### Overall 1,500-row benchmark

| Backend | Semantic accuracy | Strict exact | Mean latency |
| --- | ---: | ---: | ---: |
| **Premove ITN** | **89.70%** | **40.53%** | 56.49 ms |
| Thutmose | 59.39% | 22.13% | 15.98 ms |
| text-processing-rs | 55.79% | 16.53% | 0.14 ms |

Premove led measured semantic accuracy overall and on the 400 dedicated
voice-agent rows. It did not lead latency.

Latency used sequential batch-one requests on an Apple M4 MacBook Air with
MPS, an optimized Rust extension, and eight Rayon workers. Models were loaded
and warmed before request latency was measured. Download and initialization
are excluded. The Premove timing used the release Rust extension. The backend
evaluation was blind: each backend received only transcript text.
Independent human gold adjudication is a separate task and remains pending.

See the
[`First Evaluation report`](https://github.com/premove-ai/premove-itn/blob/main/eval/voice_agent_itn/results/first-evaluation/REPORT.md)
and
[`detailed tables`](https://github.com/premove-ai/premove-itn/blob/main/eval/voice_agent_itn/results/first-evaluation/DETAILS.md).

## Intended use

Use Premove for English voice-agent transcripts in which numbers, dates,
times, money, phone values, identifiers, URLs, and related structured values
need sentence-level disambiguation. The runtime receives only transcript text.
Candidate metadata is generated internally.

## Model lifecycle

- The first use downloads about 1.6 GB; duration depends on the network.
- Cached initialization takes several seconds on the tested system.
- Warm normalization averaged 56.49 ms with the retained release Rust build in
  the MPS benchmark.
- Services and transcript streams should keep one normalizer resident.

## Limitations

- English only.
- A 435.6M-parameter model with an approximately 1.6 GB download.
- Multi-second initialization.
- Not compatible with generic `AutoModel.from_pretrained()` loading.
- The benchmark is synthetic and does not measure live production traffic.
- Weaker measured categories include URL, MONEY, CARDINAL, TIME,
  REFERENCE_ID, and VERSION.
- Independent human gold adjudication and broader contamination checks remain
  incomplete. The backend evaluation itself was blind.
- CUDA, Windows, macOS Intel, Linux ARM64, and other accelerators are not
  validated v0.1.0 support claims.

## Release identity and provenance

- Artifact version: `v0.1.0`
- Architecture: `premove-candidate-scorer-v1`
- Required package version: `0.1.0`
- Hub repository: `premove-ai/premove-itn`
- Immutable model commit: `80bda5e2e1fe9542aa628597090242df57c1a157`
- Base model: `microsoft/deberta-v3-large`
- Base model revision: `64a8c8eab3e352a784c658aef62be1662607476f`
- Model SHA-256: `119c0f19767b61446e04da1f8f01a001edf97a47a66965e7146db2483b4937a1`

The package pins the immutable model commit and verifies its release metadata,
base-model identity, and model digest before inference. Full training
composition and checkpoint selection evidence are in the
[`production model record`](https://github.com/premove-ai/premove-itn/blob/main/docs/model-provenance.md).

## License and attribution

Premove ITN source code and model weights are MIT licensed. The scorer uses
[`microsoft/deberta-v3-large`](https://huggingface.co/microsoft/deberta-v3-large)
at the revision above. Its architecture derives from the
[`DeBERTaV3` paper](https://arxiv.org/abs/2111.09543).

The Rust realization layer uses
[`text-processing-rs`](https://github.com/FluidInference/text-processing-rs),
which is Apache-2.0 licensed. Required notices are retained in the source
repository.
