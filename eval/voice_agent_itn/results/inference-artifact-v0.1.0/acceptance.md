# Inference artifact acceptance

The acceptance gate compares the frozen production checkpoint with the
inference-only `premove-itn-contextual` artifact. Both use the same
candidate builder and decoder. No training is performed.

- Artifact: `artifacts/premove-itn-contextual-v0.1.0`
- Artifact SHA-256: `119c0f19767b61446e04da1f8f01a001edf97a47a66965e7146db2483b4937a1`
- Tensor state identical: **True**

| Corpus | Rows | Identical predictions | Artifact expected matches |
|---|---:|---:|---:|
| `golden` | 136 | 136/136 | 115/136 |
| `frozen_voice_agent` | 1500 | 1500/1500 | 608/1500 |

A release is acceptable only when both corpora report all rows
identical and the tensor state is identical.
