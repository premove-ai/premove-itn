# Model V1 validation review

Completed: 2026-08-26, before the first Golden V1 model evaluation.

This document records the selection and manual validation review for the first
DeBERTa-v3 Small contextual BIO model. It is an audit record, not a relabeled
benchmark or an adjusted score.

## Frozen inputs

```text
selected checkpoint: models/contextual-bio-v1/checkpoint-3500
checkpoint SHA-256:  6b89b1ec07467c8cae603cbcbbb12f872967a958a12ed3caa355ea12049df9f8
validation records:  4,034
validation SHA-256:  b8c29bd4e17079d29a2d8eb6e95f347a1af8fa1cb995f2c906d5488c508f7428
```

The checkpoint contains the selected step-3500 weights. The copied `best`
model has the same model SHA-256. Dataset V1 train and validation artifacts
remain unchanged.

## Raw validation result

| Metric | Result |
|---|---:|
| Strict span precision | 98.05% |
| Strict span recall | 98.37% |
| Strict span micro-F1 | 98.21% |
| Sentence exact BIO accuracy | 97.79% |
| Context-only exact accuracy | 99.06% |
| Multi-span sentence exact accuracy | 93.68% |

Strict span scoring requires an exact start, end, and `SpanKind`. These are the
unmodified Dataset V1 validation metrics. No post-review adjustment is applied.

## Per-class result

| Kind | Precision | Recall | F1 |
|---|---:|---:|---:|
| `CARDINAL` | 94.23% | 92.65% | 93.43% |
| `DATE` | 97.89% | 98.24% | 98.07% |
| `DECIMAL` | 97.44% | 98.71% | 98.07% |
| `DIGIT_SEQUENCE` | 99.57% | 99.57% | 99.57% |
| `ELECTRONIC` | 100.00% | 100.00% | 100.00% |
| `MEASUREMENT` | 95.32% | 96.01% | 95.67% |
| `MONEY` | 99.26% | 99.75% | 99.51% |
| `ORDINAL` | 94.55% | 97.59% | 96.05% |
| `PHONE` | 99.55% | 100.00% | 99.78% |
| `TIME` | 99.62% | 100.00% | 99.81% |

The critical identifier and collision classes do not show a systematic
failure. Six error records contain an error event involving `TIME`, `PHONE`,
`DIGIT_SEQUENCE`, or `ELECTRONIC`; three were judged clear model errors. No
`ELECTRONIC` error occurred.

## Full disagreement review

All 89 validation sentences with a BIO mismatch were reviewed. The diagnostic
adjudication was:

| Judgment | Records | Meaning |
|---|---:|---|
| Clear model errors | 42 | The predicted span or class is genuinely wrong. |
| Source-label or Dataset V1 policy conflicts | 17 | The frozen annotation is wrong or inconsistent with Premove's ontology. |
| Previously unresolved semantic/style ambiguities | 29 | The project had not yet frozen the desired canonical policy. |
| BIO-only grammar error | 1 | `I-DATE` starts a span, but deterministic repair recovers the correct strict span. |
| Total | 89 | Every BIO-mismatched validation sentence was reviewed. |

These record counts are diagnostic. They are not an adjusted F1, and one
record can contain multiple span-error events.

The semantic and style ambiguities motivated the frozen
[Model V1 semantic policy](semantic-policy.md). Under that policy, literal
values are normalizable regardless of tool relevance. Cases such as these are
therefore Premove-policy disagreements when Dataset V1 leaves them as `O`:

```text
Alexander was twenty three.  -> Alexander was 23.
Liverpool finished second.   -> Liverpool finished 2nd.
Season five                  -> Season 5
Objective one                -> Objective 1
```

The review also confirmed source-ontology conflicts such as a population
`one thousand` labeled `DATE`, and number-only `CARDINAL` spans where Premove
requires the complete number-plus-unit `MEASUREMENT` span.

## Genuine error backlog

The 42 clear model errors remain evidence for Dataset V2. Dominant patterns
include:

- incomplete `MEASUREMENT` number-plus-unit boundaries;
- missed ordinary `CARDINAL` values;
- difficult multi-span segmentation;
- rare `DIGIT_SEQUENCE` boundary failures;
- some `DATE` and `CARDINAL` boundary mistakes.

These failures do not justify tuning against Dataset V1 validation. They must
not be added to V1 training or used to retrain the selected checkpoint. If the
same patterns recur on independent evaluation, they become candidates for
Dataset V2 supervision.

## Source concentration

| Primary source | Records | Sentence disagreements | Rate |
|---|---:|---:|---:|
| Enrichment | 1,544 | 2 | 0.13% |
| Google TN | 2,490 | 87 | 3.49% |

Natural Google material contains almost all residual disagreements. Generated
enrichment is substantially easier, so enrichment performance alone cannot
establish readiness.

## Selection decision

Keep checkpoint 3500. Do not use checkpoint 4000 merely because it is later.

The validation curve peaks at step 3500. Step 4000 is 0.1802 percentage points
below the best strict span F1 and has lower precision. The selected checkpoint
is frozen by the SHA-256 above.

Do not:

- modify Dataset V1 train or validation records;
- regenerate the Dataset V1 split;
- add reviewed validation failures to training;
- retrain or tune against these validation disagreements;
- change the model, hyperparameters, training code, or Rust realizer as a
  result of this review.

Confirmed annotation corrections and policy improvements belong to Dataset V2.

## Runtime requirements confirmed by review

One prediction begins a date span with `I-DATE`. The validation decoder's
deterministic repair produces the correct strict span. Future runtime inference
must use the same BIO repair behavior described in
[architecture.md](architecture.md#bio-decoding).

Numeric predictions must also be decoded through checkpoint `config.id2label`
after validation against `ModelV1LabelContract`. A generic runtime enum order
must never reinterpret Model V1 IDs.

## Golden isolation

Golden V1 was not used for checkpoint selection, and no model prediction was
run against it during this validation adjudication. The semantic policy was
frozen before the first Golden V1 model evaluation.

If a frozen Golden annotation conflicts with the semantic policy, report both:

1. the unchanged frozen-Golden metric; and
2. a separate semantic-policy adjudication.

Never edit Golden after observing predictions.

## Decision record

```text
Dataset V1:       immutable
checkpoint-3500:  selected and frozen by SHA-256
training:         no rerun
model/runtime:    unchanged
Golden V1:        not evaluated during selection or this review
next experiment:  evaluate the existing checkpoint on Golden V1
```
