---
title: Premove ITN benchmark
description: Frozen 1,500-row synthetic ITN stress benchmark, including 400 voice-agent cases, comparison results, latency, methodology, and limitations.
---

Premove ITN reached **99.50% semantic entity accuracy (398/400)** on the
voice-agent subset of a frozen synthetic stress benchmark. It led the measured
semantic accuracy comparison, but it did not lead latency. These results do
not estimate accuracy on live production traffic.

## Comparison

| Backend | Voice-agent semantic accuracy | Overall semantic entity accuracy | Mean warm latency |
| --- | ---: | ---: | ---: |
| **Premove ITN** | **99.50% (398/400)** | **89.70%** | 56.49 ms |
| NVIDIA Thutmose | 67.00% (268/400) | 59.39% | 15.98 ms |
| `text-processing-rs` | 68.25% (273/400) | 55.79% | **0.14 ms** |

The voice-agent figures score the 400 rows in `group=voice_agent`. The overall
semantic figures pool 1,640 declared entities across all 1,500 rows. The
overall mean latency uses all 1,500 requests. For voice-agent rows alone,
Premove's mean warm latency was 57.41 ms.

## What the metric measures

Semantic entity accuracy checks whether a declared structured value is
correct. It allows approved formatting differences but preserves properties
such as numeric value, currency, unit, digit order, leading zeroes, and phone
dialing form. It is not full-sentence exact match. Premove's overall strict
exact match was **40.53%** in the same run.

The [frozen benchmark specification](https://github.com/premove-ai/premove-itn/blob/main/docs/evaluations/voice-agent-itn-spec.md)
defines the scoring contract. The [retained report](https://github.com/premove-ai/premove-itn/blob/main/eval/voice_agent_itn/results/first-evaluation/REPORT.md)
contains the complete result tables.

## Test design

- The balanced synthetic suite has **1,500 rows**, including **400 dedicated
  voice-agent rows** across eight domains.
- It was held out from training and checkpoint selection.
- Every backend received only transcript text. It did not receive gold spans,
  labels, domains, difficulty, or expected outputs.
- The Premove timing used the retained release Rust build. Requests were
  sequential, batch one, on an Apple M4 MacBook Air with MPS completion and
  eight Rayon workers. Models were loaded and warmed before latency was
  measured. Download and initialization were excluded.

See the [reproduction instructions](https://github.com/premove-ai/premove-itn/blob/main/benchmarks/README.md)
and [run metadata](https://github.com/premove-ai/premove-itn/blob/main/eval/voice_agent_itn/results/first-evaluation/run.json).

## Limits of the result

This suite deliberately stresses structured values and ambiguity. It is not a
random sample of customer traffic. Independent human gold adjudication remains
pending, and some similarity-contamination checks are incomplete. The retained
report also records collision and `KEEP` errors. Do not infer universal ITN
superiority from the headline voice-agent result.

For a concrete failure, the retained run predicted
`the cash price in dollars was 07:36` for the input
`the cash price in dollars was seven thirty six`, where the expected written
amount was `$7.36`. The model also changed
`the appointment starts at one oh five` to `the appointment starts at 105`
instead of the expected time `1:05`. These cases show why application-side
validation remains necessary.

`text-processing-rs` is an upstream ablation used by Premove's Rust candidate
layer, not a fully independent architecture. The Thutmose comparison uses the
NVIDIA model artifact through an isolated weight-compatible loader, not the
current NeMo API. These differences matter when interpreting the comparison.

Premove also carries a deployment tradeoff: about a 1.6 GB initial download,
multi-second initialization, and higher warm latency than either comparison
backend. Read [deployment guidance](/docs/deployment) and
[how the system works](/docs/how-it-works) before choosing it for a service.
