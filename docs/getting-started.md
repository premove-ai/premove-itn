# Getting started

## Installation

Install the latest release from PyPI:

```bash
pip install premove-itn
```

For a reproducible deployment, pin the current release:

```bash
pip install premove-itn==0.1.0
```

## Python API

```python
from premove_itn import PremoveITN

itn = PremoveITN.from_pretrained()

print(itn.normalize("the room code is one oh five"))
# the room code is 105
```

`from_pretrained()` downloads the frozen model weights from
[`premove-ai/premove-itn`](https://huggingface.co/premove-ai/premove-itn) on
first use and caches them through the normal Hugging Face cache. It can also
load a local inference-artifact directory for offline use.

Create one `PremoveITN` instance and reuse it across requests:

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

Stdin mode loads the model once, then processes every input line in order:

```text
stdin process → one model load → line 1 → line 2 → line 3 → ...
```

The CLI supports `--device auto`, `--device cpu`, `--device mps`,
`--device cuda`, `--version`, and `--help`. Normal stdout contains only
normalized transcripts. Diagnostics and errors use stderr.

## Model lifecycle

Do not combine these three costs:

| Phase | Meaning |
| --- | --- |
| First download | Fetches about 1.6 GB from Hugging Face; duration depends on the network. |
| Cached initialization | Resolves cached files, builds the model, loads weights, and transfers the model to the device; this took several seconds on the tested system. |
| Warm normalization | Processes one transcript after loading and warm-up; the retained release-build mean was 56.49 ms on Apple M4/MPS. |

The benchmark excludes model download and initialization. A one-shot CLI
timing includes process startup and model initialization, so it is not
comparable to warm request latency. Services and transcript streams should keep
one normalizer resident.

## Device selection

`device="auto"` selects CUDA when available, then Apple MPS, then CPU. Python
users can pass `device="cpu"`, `device="mps"`, or `device="cuda"` to
`PremoveITN.from_pretrained()`. The CLI exposes the same choices through
`--device`.

Release wheels are validated on macOS 14+ arm64 and `manylinux_2_28` x86_64
for Python 3.11–3.13. Real frozen-model inference is validated on Apple Silicon
MPS and Linux CPU. See the [platform support matrix](platform-support.md) for
the complete support boundary.

## More examples

| Transcript | Premove ITN output |
| --- | --- |
| `can you look up order d l t two nine eight two` | `can you look up order DLT2982` |
| `I need to change flight m d o three five one` | `I need to change flight MDO351` |
| `the meeting starts at seven thirty six` | `the meeting starts at 7:36` |
| `the cash price in dollars was seven thirty six` | `the cash price in dollars was $7.36` |
| `my verified number is eight one four two three one four` | `my verified number is 814-2314` |

The last two rows use the same spoken value. Sentence context selects a time or
a dollar amount without exposing categories or gold metadata to the model.
