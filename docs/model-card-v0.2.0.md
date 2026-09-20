---
library_name: premove-itn
language: [en]
base_model: [microsoft/deberta-v3-large]
tags: [inverse-text-normalization, speech-processing, voice-agents]
license: mit
---

# Premove ITN v0.2.0

This release updates the contextual candidate scorer used by `premove-itn`.

The v0.3.0 Python package reuses this immutable model artifact. Its structured
result and deterministic temporal features are runtime additions and do not
change the trained scorer described here.
The deterministic Rust candidate generators and exact decoder are unchanged.

Load it through the matching Python package:

```python
from premove_itn import PremoveITN

itn = PremoveITN.from_pretrained(revision="v0.2.0")
print(itn.normalize("meet me at two thirty"))
# meet me at 02:30
```

## Training

The v0.1.0 scorer was adapted for three epochs on 7,440 controlled,
single-collision TIME-versus-identifier records. No replay or dual-collision
examples were used. Epoch 3 was selected using a separate 200-row development
set before any frozen test was inspected.

## Results

| Metric | v0.1.0 | v0.2.0 |
| --- | ---: | ---: |
| Context record | 51.2% | **86.2%** |
| Counterfactual pair | 6.4% | **72.4%** |
| Identifier | 29.6% | **98.0%** |
| Time | 72.8% | **74.4%** |
| Dual span | 45.0% | **96.0%** |
| Dual sentence exact | 16.0% | **92.0%** |
| Broad strict exact | 40.5% | **43.7%** |

The contextual and dual benchmarks are synthetic controlled evaluations. They
do not estimate production voice-agent accuracy. The broad benchmark gained 86
new exact rows and lost 38 previously exact rows. Localized losses were most
visible in ORDINAL, MONEY, URL, and DIGIT_SEQUENCE.

## Limitations

- TIME recall on the controlled single-collision test is 74.4%, substantially
  below the 98.0% identifier result.
- The model is English-only and requires the `premove-itn` candidate graph and
  decoder. It is not a generic Transformers model.
- The approximately 435.6M-parameter scorer has a large download and
  multi-second initialization cost.

## Release identity

- Artifact version: `v0.2.0`
- Model-release package version: `0.2.0` (also consumed by package `0.3.0`)
- Hub repository: `premove-ai/premove-itn`
- Base model: `microsoft/deberta-v3-large`
- Base revision: `64a8c8eab3e352a784c658aef62be1662607476f`
- Source checkpoint SHA-256: `{{CHECKPOINT_SHA256}}`
- Model SHA-256: `{{MODEL_SHA256}}`

Source and evaluation evidence are available from
[`premove-ai/premove-itn`](https://github.com/premove-ai/premove-itn).
