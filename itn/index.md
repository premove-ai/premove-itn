---
title: "Premove ITN: Context-aware inverse text normalization"
sidebarTitle: Premove ITN
description: Ready-to-use inverse text normalization for English voice-agent transcripts.
---

A released Python package turns spoken ASR text into written values such as
phone numbers, dates, times, amounts, email addresses, URLs, and identifiers.
It uses sentence context to choose between valid interpretations.

```text
the room code is one oh five
→ the room code is 105
```

In the [frozen evaluation](https://github.com/premove-ai/premove-itn/blob/main/eval/voice_agent_itn/results/first-evaluation/REPORT.md),
the same spoken value could be mistaken for a time. Premove selected the room
code in this example.

## Why voice agents need ITN

Speech recognition can return spoken-form text, while downstream systems often
expect written forms. A tool often needs a structured value instead: `seven
eight three two nine` becomes an order ID, and `twenty dollars` becomes an
amount. A formatting mistake can become a failed lookup or an invalid tool
argument.

[Learn where ITN fits in a voice-agent pipeline](/itn/docs/learn/itn-for-voice-agents).

## How Premove works

```text
Generate → Score → Decode
```

Rust generates valid written candidates. A frozen DeBERTa scorer uses the full
sentence to rank them. An exact decoder selects compatible edits and can leave
text unchanged when an edit is not justified. The model cannot output a form
that the deterministic candidate layer did not generate.

[Read the illustrated explanation](/itn/docs/how-it-works).

## What it can write

| Spoken form | Possible written form |
| --- | --- |
| `one hundred twenty` | `120` |
| `four thirty` | `04:30` |
| `twenty dollars` | `$20` |
| `support at example dot com` | `support@example.com` |
| `v three dot one dot nine` | `v3.1.9` |

These are supported candidate forms, not a promise that every sentence will
select that candidate. Context determines the final output. See
[supported forms and boundaries](/itn/docs/supported-forms).

## Measured results

Premove ITN scored **398/400 (99.50%)** on the voice-agent subset of its frozen
synthetic stress benchmark. Its overall semantic entity accuracy was **89.70%**
across the 1,500-row suite. Mean warm latency was **56.49 ms** on the measured
Apple M4/MPS setup. The benchmark is not a production-traffic accuracy
estimate; Premove was also slower than both comparison backends.

[See the full comparison, method, and limitations](/itn/benchmarks).

## Install and try it

```bash
pip install premove-itn
```

```python
from premove_itn import PremoveITN

itn = PremoveITN.from_pretrained()
print(itn.normalize("the room code is one oh five"))
# the room code is 105
```

The first use downloads about 1.6 GB of model files. Load one instance and
reuse it for later requests. The frozen-model release is validated on Apple
Silicon MPS and Linux x86-64 CPU; see [deployment](/itn/docs/deployment) for the
complete platform boundary.

## Open source and open weights

- [Get started](/itn/docs/getting-started) and [browse the documentation](/itn/docs)
- [Inspect source and release evidence](https://github.com/premove-ai/premove-itn)
- [Install the Python package](https://pypi.org/project/premove-itn/)
- [View the model weights](https://huggingface.co/premove-ai/premove-itn)
- [Read the build story](https://www.aryamantodkar.com/blog/how-i-built-an-open-source-itn-model-that-beat-nvidia-thutmose-on-voice-agent-transcripts/)
