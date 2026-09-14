---
title: Getting started with Premove ITN
description: Install Premove ITN and normalize your first voice-agent transcript in Python or the CLI.
---

## Installation

Install the latest release from PyPI. Release wheels target Python 3.11–3.13
on the validated macOS and Linux platforms:

```bash
pip install premove-itn
```

To update an existing installation to the current library and matching model
release:

```bash
python -m pip install --upgrade premove-itn
```

For a reproducible deployment, pin the current release:

```bash
pip install premove-itn==0.2.0
```

## Python API

```python
from premove_itn import PremoveITN

itn = PremoveITN.from_pretrained()

print(itn.normalize("the room code is one oh five"))
# the room code is 105
```

`from_pretrained()` downloads the current frozen model weights from
[`premove-ai/premove-itn`](https://huggingface.co/premove-ai/premove-itn) on
first use and caches them through the normal Hugging Face cache. The package
pins the model to the release revision, so upgrading the package automatically
selects the matching model snapshot on the next initialization. It can also
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
instance are much faster than loading a new instance for each request. See the
[Python API](/itn/docs/python-api) for the method contract.

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
normalized transcripts. Diagnostics and errors use stderr. See the
[CLI reference](/itn/docs/cli) for usage details.

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
MPS and Linux CPU. See the [platform support matrix](/itn/docs/internals/platform-support) for
the complete support boundary.

## More examples

| Transcript | Premove ITN output |
| --- | --- |
| `the total is twenty dollars` | `the total is $20` |
| `the last account digits are zero eight two zero six three` | `the last account digits are 082063` |
| `email support at example dot com` | `email support@example.com` |
| `my order id is seven eight three two nine` | `my order id is 78329` |

These outputs were checked with the pinned v0.2.0 model artifact. They are
examples, not guarantees for every sentence. The frozen benchmark records
cases where contextual ranking selected the wrong format. See the
[benchmark results](/itn/benchmarks) before using the output as a tool argument.

Next, check [supported forms](/itn/docs/supported-forms) or read
[how Premove ITN works](/itn/docs/how-it-works).
