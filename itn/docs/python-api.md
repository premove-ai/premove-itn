---
title: Premove ITN Python API
sidebarTitle: Python API
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
    revision="e42a6ad5f58d3fde9cb6cf1f81f7fe40b9d99526",
    device="auto",
    context=None,
)
```

- `model_id` accepts the public model ID or a local inference-artifact
  directory.
- `revision` accepts the pinned commit or the verified `v0.2.0` release tag.
- `device` accepts `auto`, `cpu`, `mps`, or `cuda`.
- `context` accepts an optional default `NormalizationContext` for later calls.

The loader verifies release metadata and the model-file digest before
inference. An arbitrary Hub repository is not accepted. A local artifact must
match the frozen release contract.

`device="auto"` selects CUDA when available, then Apple MPS, then CPU. CUDA is
an API option, but it is not a validated v0.2.0 platform claim.

## Update an existing installation

Upgrade the package with the normal Python package manager:

```bash
python -m pip install --upgrade premove-itn
```

After the upgrade, `PremoveITN.from_pretrained()` uses the current package
default and downloads the matching pinned model snapshot if it is not cached.
Hugging Face keeps snapshots by revision, so an older cached release is not
overwritten. There is no silent background package update.

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

## Inspect structured normalization

Use `normalize_structured()` when a downstream system needs the exact edits
selected by ITN:

```python
result = itn.normalize_structured("pay twenty dollars")

print(result.text)
# pay $20

span = result.spans[0]
print(span.source_text, span.normalized_text)
# twenty dollars $20
```

The result is an immutable `NormalizationResult` with:

- `text`: readable normalized text.
- `resolved_text`: the machine-resolved text view.
- `spans`: selected edits as immutable `NormalizedSpan` values.

Each span retains exact half-open source and normalized character offsets,
source and normalized text, every contributing `SpanKind`, and an optional
`resolved_value`. These invariants hold for every selected span:

```python
source[span.source_start : span.source_end] == span.source_text
result.text[span.normalized_start : span.normalized_end] == span.normalized_text
```

The decoder renders the selected path once. The normalized text and offsets
come from that same render; the API does not recover alignment with a later
diff or a second inference pass.

The initial contextual resolver supports `today`, `tomorrow`, `yesterday`,
`day after tomorrow`, and `day before yesterday`. It keeps these expressions
in readable `text` and adds ISO calendar values to `resolved_value` and
`resolved_text` when `reference_datetime` is available. Without a reference
datetime, the expressions remain annotated but unresolved.

Selected named-month `DATE` spans without a year are also resolved with the
reference year. Explicit years always win. Invalid calendar dates remain
unresolved.

Selected numeric `DATE` spans are resolved when their interpretation is
structurally unique. Ambiguous dates require `DateOrder`; otherwise they stay
unresolved. Numeric dates without a year require the reference year. The
resolver accepts slash, dash, dot, and space-separated fields without adding
format-specific context fields. When the year is omitted, `DateOrder` uses the
relative position of day and month; all six orders therefore collapse to
either day-month or month-day interpretation.

## Supply normalization context

Create an immutable context from explicit caller-owned facts:

```python
from datetime import datetime

from premove_itn import DateOrder, NormalizationContext, PremoveITN

context = NormalizationContext(
    reference_datetime=datetime(2026, 9, 20, 12, 0),
    timezone="Asia/Kolkata",
    locale="en-IN",
    date_order=DateOrder.DMY,
)

itn = PremoveITN.from_pretrained(context=context)
```

The instance context is a default. A per-call context replaces it completely:

```python
result = itn.normalize_structured(
    "show my meetings tomorrow",
    context=NormalizationContext(
        reference_datetime=datetime(2026, 10, 1, 9, 0),
        timezone="America/New_York",
        locale="en-US",
        date_order=DateOrder.MDY,
    ),
)
```

Contexts are not merged field by field. Premove ITN does not discover the
current time, timezone, locale, or date order. The resolver consumes only the
explicit context supplied by the caller.

Pass an empty `NormalizationContext()` to replace an instance default with no
contextual facts for one call. Passing `context=None` uses the instance default.

`DateOrder` describes only the positional order of day, month, and year. It
supports all six permutations: `DMY`, `DYM`, `MDY`, `MYD`, `YDM`, and `YMD`.
It does not encode separators or surface formats. For example, `30/09/2026`,
`30-09-2026`, and `30 09 2026` all use `DateOrder.DMY`. Named-month dates do
not need a date order when their fields are already unambiguous.

## Return the resolved text view

`normalize_resolved()` is the string-only view of the same internal result:

```python
resolved = itn.normalize_resolved("pay twenty dollars")
print(resolved)
# pay $20
```

It does not run a separate inference or reconstruction path. Its output will
differ from `normalize()` when a supported contextual expression has a
deterministic value to render:

```python
resolved = itn.normalize_resolved(
    "call me tomorrow",
    context=NormalizationContext(
        reference_datetime=datetime(2026, 9, 20, 12, 0),
    ),
)
print(resolved)
# call me 2026-09-21
```

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

For model lifecycle and supported environments, see [Deployment](/itn/docs/deployment).
For a streaming shell workflow, see the [CLI](/itn/docs/cli).
