---
title: Python API
description: Load Premove ITN and normalize English voice-agent transcripts in Python.
---

The public contextual API has one implementation: `PremoveITN`.

## Load the model

```python
from premove_itn import PremoveITN

itn = PremoveITN.from_pretrained()
```

`from_pretrained()` downloads the pinned public model on first use and uses the
Hugging Face cache on later loads. Create one instance and reuse it. Model
initialization is expensive.

### Parameters

```text
PremoveITN.from_pretrained(
    model_id="premove-ai/premove-itn",
    *,
    revision="80bda5e2e1fe9542aa628597090242df57c1a157",
    device="auto",
)
```

- `model_id` accepts the public model ID or a local inference-artifact
  directory.
- `revision` accepts the pinned commit or the verified `v0.1.0` release tag.
- `device` accepts `auto`, `cpu`, `mps`, or `cuda`.

The loader verifies release metadata and the model-file digest before
inference. An arbitrary Hub repository is not accepted. A local artifact must
match the frozen release contract.

`device="auto"` selects CUDA when available, then Apple MPS, then CPU. CUDA is
an API option, but it is not a validated v0.1.0 platform claim.

## Normalize a transcript

```python
result = itn.normalize("call me at four thirty")
print(result)
# call me at 04:30
```

`normalize()` accepts one string and returns one string. It raises `TypeError`
for a non-string input. Empty and whitespace-only strings are returned
unchanged. If the text produces candidates, inputs longer than 512 DeBERTa
encoder tokens are rejected instead of being truncated. Text with no
candidates returns unchanged before tokenization.

## Reuse one instance

```python
texts = [
    "the total is twenty dollars",
    "email support at example dot com",
    "the room code is one oh five",
]

for text in texts:
    print(itn.normalize(text))
```

For model lifecycle and supported environments, see [Deployment](/docs/deployment).
For a streaming shell workflow, see the [CLI](/docs/cli).
