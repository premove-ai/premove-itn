# Checkpoint evaluation registry

This registry preserves the stable comparison results for each major model
checkpoint. Every comparison must show the deterministic Rust normalizer from
`text-processing-rs` as the baseline. Raw metrics remain under
`data/generated/`; they include per-kind counts, ablations, error categories,
samples, source hashes, model revision, and checkpoint metadata.

All model evaluations use DeBERTa revision
`64a8c8eab3e352a784c658aef62be1662607476f`. Evaluation data must not be used
for training.

## Current release comparison

These are the two retained model checkpoints. Google, conversational, Golden,
and PolyNorm use strict sentence exact accuracy. Numb3rs uses the separately
defined semantic accuracy.

| Benchmark | Deterministic Rust | 378k baseline | **378k + 20k** |
| --- | ---: | ---: | ---: |
| Google validation exact | 61.19% | 95.24% | **95.41%** |
| Conversational validation exact | 89.11% | 89.21% | **95.89%** |
| Golden exact | 65.44% | 74.26% | **77.94%** |
| NVIDIA Numb3rs semantic | 66.79% | 77.74% | **77.78%** |
| Apple PolyNorm exact, all 540 | 17.04% | 20.93% | **21.30%** |
| Apple PolyNorm exact, reachable 161 | 52.80% | 70.19% | **71.43%** |
| VoiceCodeBench entity exact, all 1,482 | 22.47% | 20.11% | **25.51%** |
| VoiceCodeBench entity exact, reachable 711 | 43.88% | 41.91% | **53.16%** |
| PolyNorm semantic, mapped 140 | 77.14% | **82.86%** | **82.86%** |
| VoiceCodeBench semantic, mapped 545 | **69.36%** | 51.38% | **58.53%** |

PolyNorm's reachable-only row measures selection among targets the current
Rust graph can construct. It is not a replacement for the full 540-row result.
The 161-row graph includes the first Rust coverage stage for structured `WORD`
versions and identifiers.

VoiceCodeBench uses the supplied entity-level acoustic and canonical text. It
is a Premove component diagnostic, not an official raw-audio VoiceCodeBench
score. The +20k checkpoint improves on both the 378k model and deterministic
Rust, while the 47.98% oracle shows that Rust coverage is now the main limit.
Semantic results and their exact category mappings are documented in
[`semantic-validation-report.md`](semantic-validation-report.md).

## Recorded results

| Benchmark | System | Exact | Positive | KEEP | MULTI | Raw metrics |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Golden | Deterministic Rust (`text-processing-rs`) | 65.44% | 67.71% | 60.00% | 50.00% | Both Golden metric files |
| Golden | 1k model | 69.85% | 64.58% | 82.50% | 37.50% | `data/generated/first_run_1000/golden_metrics.json` |
| Golden | 10k model | 70.59% | 64.58% | 85.00% | 37.50% | `data/generated/google_validation_10k/golden_metrics.json` |
| Golden | 82k model | 64.71% | 57.29% | 82.50% | 25.00% | `data/generated/full_google_train/evaluations/golden_082000/metrics.json` |
| Golden | 130k model | 64.71% | 57.29% | 82.50% | 25.00% | `data/generated/full_google_train/evaluations/golden_130000/metrics.json` |
| Golden | 210k model | 68.38% | 64.58% | 77.50% | 37.50% | `data/generated/full_google_train/evaluations/golden_210000/metrics.json` |
| Golden | 250k model | 70.59% | 77.08% | 55.00% | 62.50% | `data/generated/full_google_train/evaluations/golden_250000/metrics.json` |
| Golden | 298k model | **74.26%** | **79.17%** | 62.50% | 62.50% | `data/generated/full_google_train/evaluations/golden_298000/metrics.json` |
| Golden | 346k model | 72.79% | 77.08% | 62.50% | 75.00% | `data/generated/full_google_train/evaluations/golden_346000/metrics.json` |
| Golden | 378k model | 74.26% | 76.04% | 70.00% | 75.00% | `data/generated/full_google_train/evaluations/golden_378000/metrics.json` |
| Golden | 402k model | 73.53% | 73.96% | 72.50% | 62.50% | `data/generated/full_google_train/evaluations/golden_402000/metrics.json` |
| Golden | 450k model | 73.53% | 75.00% | 70.00% | 50.00% | `data/generated/full_google_train/evaluations/golden_450000/metrics.json` |
| Golden | 482k model | 72.79% | 75.00% | 67.50% | 50.00% | `data/generated/full_google_train/evaluations/golden_482000/metrics.json` |
| Golden | 498k model | 69.85% | 69.79% | 70.00% | 25.00% | `data/generated/full_google_train/evaluations/golden_498000/metrics.json` |
| Golden | 711,135 model | 68.38% | 68.75% | 67.50% | 37.50% | `data/generated/full_google_train/evaluations/golden_711135/metrics.json` |
| Google validation | Deterministic Rust (`text-processing-rs`) | 61.19% | 56.83% | 87.87% | 41.53% | Both Google validation metric files |
| Google validation | 1k model | 80.81% | 78.04% | 97.71% | 70.63% | `data/generated/google_validation_1k_full/metrics.json` |
| Google validation | 10k model | 88.35% | 86.63% | 98.92% | 81.80% | `data/generated/google_validation_10k/metrics.json` |
| Google validation | 250k model | 93.67% | 93.90% | 92.28% | 92.08% | `data/generated/full_google_train/evaluations/google_validation_250000/metrics.json` |
| Google validation | 298k model | 95.04% | 95.27% | 93.63% | 93.20% | `data/generated/full_google_train/evaluations/google_validation_298000/metrics.json` |
| Google validation | 346k model | 95.09% | 95.66% | 91.56% | 94.49% | `data/generated/full_google_train/evaluations/google_validation_346000/metrics.json` |
| Google validation | 378k model | 95.24% | 95.87% | 91.40% | 94.35% | `data/generated/full_google_train/evaluations/google_validation_378000/metrics.json` |
| Google validation | 402k model | 95.27% | 95.77% | 92.19% | 94.61% | `data/generated/full_google_train/evaluations/google_validation_402000/metrics.json` |
| Google validation | 450k model | **95.43%** | **96.09%** | 91.38% | **94.58%** | `data/generated/full_google_train/evaluations/google_validation_450000/metrics.json` |
| Google validation | 498k model | 94.87% | 95.44% | 91.38% | 94.34% | `data/generated/full_google_train/evaluations/google_validation_498000/metrics.json` |
| Google validation | 711,135 model | 95.11% | 95.66% | 91.76% | 94.35% | `data/generated/full_google_train/evaluations/google_validation_711135/metrics.json` |
| Conversational validation | Deterministic Rust (`text-processing-rs`) | 89.11% | 32.98% | 92.98% | 32.54% | `data/generated/full_google_train/evaluations/conversational_validation_001000/metrics.json` |
| Conversational validation | 1k model | **92.57%** | 42.64% | **96.01%** | 67.16% | `data/generated/full_google_train/evaluations/conversational_validation_001000/metrics.json` |
| Conversational validation | 10k model | 88.91% | 27.76% | 93.12% | 45.56% | `data/generated/full_google_train/evaluations/conversational_validation_010000/metrics.json` |
| Conversational validation | 250k model | 86.52% | **75.46%** | 87.28% | **78.40%** | `data/generated/full_google_train/evaluations/conversational_validation_250000/metrics.json` |
| Conversational validation | 298k model | 88.81% | 66.10% | 90.37% | 61.83% | `data/generated/full_google_train/evaluations/conversational_validation_298000/metrics.json` |
| Conversational validation | 346k model | 88.84% | 74.08% | 89.85% | 77.81% | `data/generated/full_google_train/evaluations/conversational_validation_346000/metrics.json` |
| Conversational validation | 378k model | 89.21% | 69.48% | 90.57% | 72.78% | `data/generated/full_google_train/evaluations/conversational_validation_378000/metrics.json` |
| Conversational validation | 402k model | 87.86% | 72.09% | 88.94% | 76.63% | `data/generated/full_google_train/evaluations/conversational_validation_402000/metrics.json` |
| Conversational validation | 450k model | 82.28% | 64.42% | 83.51% | 68.05% | `data/generated/full_google_train/evaluations/conversational_validation_450000/metrics.json` |
| Conversational validation | 498k model | 87.62% | 66.56% | 89.07% | 70.41% | `data/generated/full_google_train/evaluations/conversational_validation_498000/metrics.json` |
| Conversational validation | 711,135 model | 88.26% | 63.19% | 89.99% | 65.68% | `data/generated/full_google_train/evaluations/conversational_validation_711135/metrics.json` |
| NVIDIA Numb3rs | Deterministic Rust (`text-processing-rs`) | 39.43% | n/a | n/a | n/a | `data/generated/numb3rs/10k/metrics.json` |
| NVIDIA Numb3rs | 1k model | 22.19% | n/a | n/a | n/a | `data/generated/numb3rs/1k/metrics.json` |
| NVIDIA Numb3rs | 10k model | 25.33% | n/a | n/a | n/a | `data/generated/numb3rs/10k/metrics.json` |
| NVIDIA Numb3rs | 82k model | 34.45% | n/a | n/a | n/a | `data/generated/full_google_train/evaluations/numb3rs_082000/metrics.json` |
| NVIDIA Numb3rs | 130k model | 34.45% | n/a | n/a | n/a | `data/generated/full_google_train/evaluations/numb3rs_130000/metrics.json` |
| NVIDIA Numb3rs | 210k model | 29.96% | n/a | n/a | n/a | `data/generated/full_google_train/evaluations/numb3rs_210000/metrics.json` |
| NVIDIA Numb3rs | 250k model | 38.24% | n/a | n/a | n/a | `data/generated/full_google_train/evaluations/numb3rs_250000/metrics.json` |
| NVIDIA Numb3rs | 298k model | 37.95% | n/a | n/a | n/a | `data/generated/full_google_train/evaluations/numb3rs_298000/metrics.json` |
| NVIDIA Numb3rs | 346k model | 38.68% | n/a | n/a | n/a | `data/generated/full_google_train/evaluations/numb3rs_346000/metrics.json` |
| NVIDIA Numb3rs | 378k model | 38.91% | n/a | n/a | n/a | `data/generated/full_google_train/evaluations/numb3rs_378000/metrics.json` |
| NVIDIA Numb3rs | 402k model | 38.78% | n/a | n/a | n/a | `data/generated/full_google_train/evaluations/numb3rs_402000/metrics.json` |
| NVIDIA Numb3rs | 450k model | 38.95% | n/a | n/a | n/a | `data/generated/full_google_train/evaluations/numb3rs_450000/metrics.json` |
| NVIDIA Numb3rs | 482k model | 38.81% | n/a | n/a | n/a | `data/generated/full_google_train/evaluations/numb3rs_482000/metrics.json` |
| NVIDIA Numb3rs | 498k model | 38.63% | n/a | n/a | n/a | `data/generated/full_google_train/evaluations/numb3rs_498000/metrics.json` |
| NVIDIA Numb3rs | 711,135 model | 36.34% | n/a | n/a | n/a | `data/generated/full_google_train/evaluations/numb3rs_711135/metrics.json` |

## NVIDIA Numb3rs accuracy comparison

The primary Numb3rs accuracy is semantic accuracy. It accepts different
formatting only when value-preserving Rust equivalence proves that both outputs
have the same number, currency, unit, date, time, ordinal, or digit sequence.
Exact-reference results remain in the raw metric files for audit only.

| System | Semantic accuracy | Correct |
| --- | ---: | ---: |
| Deterministic Rust (`text-processing-rs`) | 66.79% | 6,766 / 10,131 |
| Google 378k model | 77.74% | 7,876 / 10,131 |
| **378k + 20k conversational adaptation** | **77.78%** | **7,880 / 10,131** |

`ADDRESS`, `FRACTION`, and `PLAIN` fail closed because Premove does not have a
safe equivalence implementation for those Numb3rs categories. Their outputs
remain incorrect unless they match the reference exactly. These semantic
scores are therefore conservative. The raw metric files preserve exact and
semantic counts and per-category results.

## Apple PolyNorm-Bench en-US

PolyNorm is evaluated in the reverse direction from normalized spoken text to
the original written text. Every run evaluates both retained checkpoints and
the deterministic Rust normalizer. After the first structured-identifier
coverage stage, the Rust candidate graph can construct the exact target for
**161/540** rows, up from 150.

| System | All rows | Reachable-only |
| --- | ---: | ---: |
| Deterministic Rust | 92/540 = 17.04% | 85/161 = 52.80% |
| 378k baseline | 113/540 = 20.93% | 113/161 = 70.19% |
| **378k + 20k** | **115/540 = 21.30%** | **115/161 = 71.43%** |

| Category | All rows | Reachable-only | Reachable |
| --- | ---: | ---: | ---: |
| URL or Email | 5/20 | 5/17 | 17/20 |
| Phone Number | 11/20 | 11/13 | 13/20 |
| Time | 7/20 | 7/7 | 7/20 |
| Cardinal | 13/20 | 13/17 | 17/20 |
| Currency | 9/20 | 9/9 | 9/20 |
| Decimal | 8/20 | 8/15 | 15/20 |
| Unit | 12/20 | 12/17 | 17/20 |
| Vehicle or Product Code | 0/20 | 0/1 | 1/20 |
| License Plate or Serial Numbers | 0/20 | 0/2 | 2/20 |
| Version Numbers | 10/20 | 10/10 | 10/20 |

All-row exact is the primary result. Reversed text normalization is not always
uniquely recoverable, and exact scoring includes Apple's written-form policy.
Reachable-only accuracy isolates contextual selection only; it must retain its
161-row denominator. The source is evaluation-only and is not redistributed.
The current raw metrics are stored at
`data/generated/polynorm/google_378000_rust_stage1/metrics.json` and
`data/generated/polynorm/checkpoint_020000_rust_stage1/metrics.json`.

## VoiceCodeBench

VoiceCodeBench is pinned at revision
`3ccea73877a159eb2a8b17304148c325c5fe5061`. This component evaluation uses
all 1,482 annotated entities from its 300-row test split.

| System | All entities | Reachable-only |
| --- | ---: | ---: |
| Deterministic Rust | 333/1,482 = 22.47% | 312/711 = 43.88% |
| 378k baseline | 298/1,482 = 20.11% | 298/711 = 41.91% |
| **378k + 20k** | **378/1,482 = 25.51%** | **378/711 = 53.16%** |

The exact oracle is 711/1,482 = 47.98%. Major zero-reachability classes are
CLI flags, code symbols, email addresses, environment variables, file paths,
phone extensions, dates, times, and versions under the benchmark's canonical
policy. This first result is frozen as the pre-adaptation baseline. Raw metrics
are stored under `data/generated/voicecodebench/`.

The complete 50k milestone curve, per-kind results, final Google validation,
and checkpoint-selection analysis are in
[`full-google-milestone-analysis.md`](full-google-milestone-analysis.md).

## Dataset identities

| Benchmark | Source identity |
| --- | --- |
| Golden | SHA-256 `271ab423ebcb0fc041b3dc0fbb144ef021b8078553a4b9ef69b2e4f9a4a0466a` |
| Google validation | SHA-256 `1da9671c18e4916f3700244d9c100b25b3e8e0113b1336ba338b2f11b63541c0` |
| Conversational validation | SHA-256 `4be85e98559a79d55e9a5afa6d7366e91d8f1c165dd49875bb70839668085ced` |
| NVIDIA Numb3rs | revision `192908075e1bd293914cc7b508f4a183ba6ef2b8`; metadata SHA-256 `d6c6020fc8d3ccf8404395814307dbf0aca415305ed6964543e63fb3aa5df377` |
| VoiceCodeBench | revision `3ccea73877a159eb2a8b17304148c325c5fe5061`; metadata SHA-256 `a69c40387303ea70fe46160239266cdb64b6ad9a68ca160c7e12640b4afb74f5` |

Add each completed run to the table without replacing an earlier result.
Keep benchmark inputs, source revisions, and checkpoint identities unchanged
when comparing checkpoints.
