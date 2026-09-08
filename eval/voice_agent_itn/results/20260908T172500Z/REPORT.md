# First Evaluation — original accuracy evidence

The original per-record outputs and metrics are preserved. The Rust extension
used for this run was a development build. Its latency tables are historical
diagnostics, not the release performance comparison. See
[First Evaluation](../first-evaluation/REPORT.md) for corrected release latency
and the candidate/output equivalence audit. The model checkpoint is unchanged.

Run completed: `2026-09-08T12:01:21.735793Z`  
Dataset: `eval/voice_agent_itn/voice_agent_eval.jsonl`  
Rows: **1500**  

## Executive summary

This is a blind, backend-neutral run of the frozen 1,500-row VoiceAgent ITN dataset.
Each adapter received only the row's `text` field. Gold spans, categories, domains,
difficulty, and expected output were withheld from every backend. The per-record JSONL
files are the source of truth for every aggregate below.

The Premove ITN entry is the current production structured-value 20k checkpoint named
by `data/models/production.json`. The text-processing-rs entry is the compiled upstream
sentence normalizer. Thutmose is the official `itn_en_thutmose_bert` artifact, run in
an isolated worker with its own Python environment.

## Backend comparison

| Backend | Rows | Strict exact | Semantic entity | All-entities rows | Mean ms | p50 ms | p95 ms | p99 ms | Throughput/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **text-processing-rs** | 1500 | 0.17 | 0.56 | 0.58 | 0.73 | 0.54 | 2.17 | 3.48 | 1364.77 |
| **premove-itn** | 1500 | 0.41 | 0.90 | 0.88 | 240.93 | 135.90 | 833.24 | 1604.14 | 4.15 |
| **thutmose** | 1500 | 0.22 | 0.59 | 0.63 | 16.45 | 16.44 | 18.08 | 19.23 | 60.79 |

Strict exact is the canonical full-sentence match. Semantic entity accuracy is
entity-level and category-aware; it preserves numeric value, currency, unit, digit
order, phone dialing form, and identifier case policy. No-span rows do not enter the
semantic-entity denominator. `all-entities rows` is the fraction of rows for which
every declared entity was semantically correct.
Structured identifiers ignore case and harmless whitespace/separator formatting for
semantic scoring; strict exact accuracy still requires the complete expected sentence.

## Detailed per-backend results

### text-processing-rs

Compiled upstream English sentence normalizer exposed by premove_itn.

- Records: **1500**
- Initialization: **5.36 ms**
- Warm-up: **0.00 ms** (excluded from per-record latency)
- Per-record output: [`eval/voice_agent_itn/results/20260908T172500Z/text-processing-rs/records.jsonl`](eval/voice_agent_itn/results/20260908T172500Z/text-processing-rs/records.jsonl)
- Runtime: `{"runtime": "premove_itn._rust.baseline_normalize_sentence"}`

#### Per kind

| Slice | Rows | Strict exact | Semantic entities | All entities | Mean ms | p50 ms | p95 ms | p99 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `CARDINAL` | 53 | 0.58 | 0.58 | 0.58 | 0.40 | 0.34 | 0.69 | 0.73 |
| `DATE` | 123 | 0.00 | 1.00 | 1.00 | 0.63 | 0.68 | 0.87 | 1.17 |
| `DECIMAL` | 38 | 1.00 | 1.00 | 1.00 | 0.46 | 0.49 | 0.70 | 0.73 |
| `DIGIT_SEQUENCE` | 63 | 0.00 | 0.37 | 0.37 | 0.57 | 0.56 | 0.84 | 0.87 |
| `EMAIL` | 48 | 0.00 | 0.00 | 0.00 | 0.35 | 0.36 | 0.39 | 0.48 |
| `FLIGHT_ID` | 38 | 0.00 | 0.50 | 0.50 | 0.88 | 0.82 | 1.65 | 1.72 |
| `IP` | 37 | 0.19 | 0.19 | 0.19 | 0.74 | 0.53 | 1.79 | 2.31 |
| `KEEP` | 100 | 0.38 | — | — | 0.22 | 0.23 | 0.38 | 0.65 |
| `MEASUREMENT` | 47 | 1.00 | 1.00 | 1.00 | 0.46 | 0.41 | 0.73 | 0.76 |
| `MONEY` | 83 | 0.12 | 0.12 | 0.12 | 0.94 | 0.85 | 1.78 | 2.06 |
| `MULTI` | 200 | 0.01 | 0.30 | 0.14 | 2.00 | 1.88 | 3.70 | 4.22 |
| `ORDER_ID` | 48 | 0.00 | 0.62 | 0.62 | 0.67 | 0.60 | 1.16 | 1.25 |
| `ORDINAL` | 53 | 1.00 | 1.00 | 1.00 | 0.20 | 0.18 | 0.32 | 0.42 |
| `PERCENT` | 38 | 0.00 | 1.00 | 1.00 | 0.28 | 0.27 | 0.46 | 0.48 |
| `PHONE` | 93 | 0.00 | 0.51 | 0.51 | 0.96 | 0.92 | 1.95 | 1.99 |
| `REFERENCE_ID` | 138 | 0.03 | 0.57 | 0.57 | 0.56 | 0.53 | 1.03 | 1.19 |
| `TIME` | 153 | 0.00 | 0.82 | 0.82 | 0.38 | 0.34 | 0.67 | 0.69 |
| `URL` | 62 | 0.02 | 0.45 | 0.45 | 0.48 | 0.43 | 1.29 | 1.54 |
| `VERSION` | 38 | 0.45 | 1.00 | 1.00 | 0.38 | 0.32 | 0.85 | 0.91 |
| `ZIP_CODE` | 47 | 0.00 | 1.00 | 1.00 | 0.54 | 0.59 | 0.68 | 0.86 |

#### Per benchmark group

| Slice | Rows | Strict exact | Semantic entities | All entities | Mean ms | p50 ms | p95 ms | p99 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `collision` | 350 | 0.16 | 0.74 | 0.74 | 0.31 | 0.27 | 0.60 | 0.63 |
| `multi` | 200 | 0.01 | 0.30 | 0.14 | 2.00 | 1.88 | 3.70 | 4.22 |
| `negative` | 50 | 0.72 | — | — | 0.18 | 0.13 | 0.45 | 0.69 |
| `normal` | 500 | 0.22 | 0.57 | 0.57 | 0.56 | 0.46 | 1.25 | 1.75 |
| `voice_agent` | 400 | 0.11 | 0.68 | 0.68 | 0.75 | 0.69 | 1.55 | 1.98 |

#### Per domain

| Slice | Rows | Strict exact | Semantic entities | All entities | Mean ms | p50 ms | p95 ms | p99 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `banking` | 75 | 0.00 | 0.33 | 0.36 | 1.41 | 1.15 | 3.08 | 3.71 |
| `customer_support` | 75 | 0.01 | 0.36 | 0.40 | 1.10 | 0.83 | 2.34 | 3.53 |
| `ecommerce` | 75 | 0.00 | 0.68 | 0.65 | 1.14 | 0.86 | 3.18 | 3.71 |
| `general` | 900 | 0.23 | 0.64 | 0.64 | 0.44 | 0.34 | 1.10 | 1.65 |
| `healthcare_admin` | 75 | 0.13 | 0.53 | 0.48 | 1.36 | 0.85 | 3.37 | 3.76 |
| `logistics` | 75 | 0.13 | 0.60 | 0.65 | 1.02 | 0.68 | 2.90 | 3.08 |
| `scheduling` | 75 | 0.01 | 0.47 | 0.43 | 1.24 | 1.00 | 3.05 | 4.32 |
| `technical_support` | 75 | 0.17 | 0.44 | 0.45 | 1.02 | 0.64 | 2.70 | 3.34 |
| `travel` | 75 | 0.13 | 0.45 | 0.57 | 1.07 | 0.80 | 2.48 | 2.63 |

#### Per difficulty

| Slice | Rows | Strict exact | Semantic entities | All entities | Mean ms | p50 ms | p95 ms | p99 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `easy` | 611 | 0.21 | 0.81 | 0.81 | 0.59 | 0.60 | 0.98 | 1.57 |
| `hard` | 573 | 0.11 | 0.42 | 0.45 | 0.72 | 0.34 | 2.77 | 3.83 |
| `medium` | 316 | 0.17 | 0.39 | 0.29 | 1.04 | 0.95 | 2.27 | 2.88 |

#### Per declared entity category

| Slice | Rows | Strict exact | Semantic entities | All entities | Mean ms | p50 ms | p95 ms | p99 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `CARDINAL` | 145 | 0.21 | 0.32 | 0.23 | 1.51 | 1.42 | 3.62 | 3.93 |
| `DATE` | 135 | 0.00 | 0.87 | 0.92 | 0.81 | 0.69 | 2.52 | 3.01 |
| `DECIMAL` | 80 | 0.50 | 0.51 | 0.56 | 1.29 | 1.27 | 3.25 | 3.52 |
| `DIGIT_SEQUENCE` | 76 | 0.00 | 0.31 | 0.30 | 0.86 | 0.58 | 3.06 | 3.71 |
| `EMAIL` | 65 | 0.00 | 0.00 | 0.00 | 0.69 | 0.37 | 2.22 | 2.54 |
| `FLIGHT_ID` | 50 | 0.00 | 0.50 | 0.42 | 1.09 | 0.87 | 2.34 | 2.77 |
| `IP` | 54 | 0.13 | 0.09 | 0.13 | 1.43 | 0.72 | 3.97 | 4.34 |
| `MEASUREMENT` | 64 | 0.75 | 0.79 | 0.81 | 0.82 | 0.52 | 2.07 | 2.60 |
| `MONEY` | 95 | 0.12 | 0.17 | 0.12 | 1.18 | 0.88 | 2.74 | 4.11 |
| `ORDER_ID` | 61 | 0.00 | 0.48 | 0.56 | 0.96 | 0.68 | 2.32 | 3.16 |
| `ORDINAL` | 145 | 0.37 | 0.53 | 0.50 | 1.20 | 1.12 | 3.02 | 4.26 |
| `PERCENT` | 50 | 0.00 | 0.91 | 0.88 | 0.45 | 0.34 | 1.62 | 2.10 |
| `PHONE` | 106 | 0.00 | 0.46 | 0.46 | 1.11 | 1.01 | 2.27 | 2.89 |
| `REFERENCE_ID` | 151 | 0.03 | 0.48 | 0.52 | 0.69 | 0.60 | 1.83 | 2.58 |
| `TIME` | 165 | 0.00 | 0.77 | 0.78 | 0.49 | 0.35 | 1.61 | 2.14 |
| `URL` | 79 | 0.01 | 0.28 | 0.35 | 0.78 | 0.59 | 2.00 | 3.21 |
| `VERSION` | 55 | 0.31 | 0.69 | 0.71 | 1.06 | 0.36 | 3.29 | 3.59 |
| `ZIP_CODE` | 64 | 0.00 | 0.74 | 0.84 | 0.91 | 0.64 | 2.28 | 3.27 |

#### Collision and compositional metrics

- Collision pairs: **175**
- Strict pair accuracy: **0.01**
- Semantic pair accuracy: **0.58**
- Multi-entity span accuracy: **0.30**
- Multi-entity all-correct accuracy: **0.14**
- Voice-agent group accuracy (strict): **0.11**
- Standalone negative KEEP preservation: **0.72**
- All no-span KEEP preservation: **0.38**

Collision-family metrics:

| Family | Rows | Strict exact | Semantic entities | Mean ms | p95 ms |
|---|---:|---:|---:|---:|---:|
| `date_vs_identifier` | 50 | 0.00 | 1.00 | 0.27 | 0.28 |
| `false_positive_keep` | 50 | 0.72 | — | 0.18 | 0.45 |
| `identifier_vs_time` | 50 | 0.08 | 0.54 | 0.28 | 0.36 |
| `military_time_vs_cardinal` | 50 | 0.50 | 0.50 | 0.32 | 0.41 |
| `money_vs_time` | 50 | 0.00 | 0.50 | 0.47 | 0.63 |
| `ordinal_vs_plain` | 50 | 0.54 | 1.00 | 0.22 | 0.37 |
| `phone_vs_digits` | 50 | 0.00 | 0.92 | 0.41 | 0.59 |
| `url_vs_plain` | 50 | 0.00 | 1.00 | 0.20 | 0.30 |

#### Error categories

`{"false_normalization": 62, "wrong_rewrite": 1190}`

### premove-itn

Production structured-value 20k contextual candidate scorer and decoder.

- Records: **1500**
- Initialization: **7258.41 ms**
- Warm-up: **624.79 ms** (excluded from per-record latency)
- Per-record output: [`eval/voice_agent_itn/results/20260908T172500Z/premove-itn/records.jsonl`](eval/voice_agent_itn/results/20260908T172500Z/premove-itn/records.jsonl)
- Runtime: `{"checkpoint": "data/models/structured_value_selected_20k/checkpoint.pt", "device": "mps", "model_name": "microsoft/deberta-v3-large"}`

#### Per kind

| Slice | Rows | Strict exact | Semantic entities | All entities | Mean ms | p50 ms | p95 ms | p99 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `CARDINAL` | 53 | 0.53 | 0.53 | 0.53 | 99.51 | 88.73 | 143.23 | 163.54 |
| `DATE` | 123 | 0.00 | 1.00 | 1.00 | 133.12 | 138.15 | 171.09 | 216.34 |
| `DECIMAL` | 38 | 1.00 | 1.00 | 1.00 | 115.79 | 118.50 | 162.61 | 166.72 |
| `DIGIT_SEQUENCE` | 63 | 1.00 | 1.00 | 1.00 | 138.38 | 137.71 | 167.07 | 192.96 |
| `EMAIL` | 48 | 0.96 | 0.96 | 0.96 | 384.46 | 372.19 | 511.72 | 527.53 |
| `FLIGHT_ID` | 38 | 0.00 | 0.95 | 0.95 | 179.39 | 163.21 | 306.15 | 309.54 |
| `IP` | 37 | 1.00 | 1.00 | 1.00 | 429.48 | 434.98 | 576.14 | 587.18 |
| `KEEP` | 100 | 0.81 | — | — | 59.23 | 72.41 | 100.92 | 223.35 |
| `MEASUREMENT` | 47 | 0.30 | 0.98 | 0.98 | 115.58 | 114.76 | 157.54 | 167.53 |
| `MONEY` | 83 | 0.63 | 0.69 | 0.69 | 179.39 | 164.59 | 312.36 | 374.33 |
| `MULTI` | 200 | 0.42 | 0.93 | 0.89 | 756.25 | 614.97 | 1715.64 | 2827.70 |
| `ORDER_ID` | 48 | 0.00 | 0.94 | 0.94 | 145.29 | 139.21 | 216.33 | 229.78 |
| `ORDINAL` | 53 | 0.92 | 1.00 | 1.00 | 71.04 | 69.77 | 83.84 | 95.18 |
| `PERCENT` | 38 | 0.00 | 1.00 | 1.00 | 84.01 | 85.72 | 101.50 | 110.18 |
| `PHONE` | 93 | 0.00 | 1.00 | 1.00 | 233.30 | 232.38 | 445.25 | 456.06 |
| `REFERENCE_ID` | 138 | 0.01 | 0.80 | 0.80 | 126.98 | 118.23 | 209.76 | 229.41 |
| `TIME` | 153 | 0.06 | 0.85 | 0.85 | 99.47 | 90.71 | 137.95 | 143.64 |
| `URL` | 62 | 0.60 | 0.60 | 0.60 | 485.91 | 585.93 | 933.62 | 936.21 |
| `VERSION` | 38 | 0.55 | 0.84 | 0.84 | 96.43 | 88.86 | 164.01 | 166.37 |
| `ZIP_CODE` | 47 | 1.00 | 1.00 | 1.00 | 123.50 | 120.73 | 151.34 | 162.48 |

#### Per benchmark group

| Slice | Rows | Strict exact | Semantic entities | All entities | Mean ms | p50 ms | p95 ms | p99 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `collision` | 350 | 0.26 | 0.60 | 0.60 | 87.64 | 78.87 | 137.70 | 145.30 |
| `multi` | 200 | 0.42 | 0.93 | 0.89 | 756.25 | 614.97 | 1715.64 | 2827.70 |
| `negative` | 50 | 0.94 | — | — | 41.47 | 13.42 | 126.70 | 223.73 |
| `normal` | 500 | 0.50 | 0.97 | 0.97 | 191.09 | 127.69 | 567.79 | 827.26 |
| `voice_agent` | 400 | 0.34 | 0.99 | 0.99 | 204.62 | 145.97 | 459.72 | 928.22 |

#### Per domain

| Slice | Rows | Strict exact | Semantic entities | All entities | Mean ms | p50 ms | p95 ms | p99 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `banking` | 75 | 0.45 | 1.00 | 1.00 | 395.53 | 224.81 | 1307.81 | 1579.33 |
| `customer_support` | 75 | 0.25 | 0.99 | 0.99 | 339.94 | 216.15 | 773.54 | 1836.15 |
| `ecommerce` | 75 | 0.47 | 0.99 | 0.99 | 276.91 | 164.59 | 926.78 | 1318.23 |
| `general` | 900 | 0.43 | 0.83 | 0.83 | 142.55 | 98.62 | 400.84 | 741.66 |
| `healthcare_admin` | 75 | 0.21 | 0.98 | 0.97 | 365.77 | 174.33 | 935.48 | 1921.03 |
| `logistics` | 75 | 0.39 | 0.96 | 0.95 | 363.08 | 139.84 | 1584.59 | 1980.57 |
| `scheduling` | 75 | 0.28 | 0.95 | 0.93 | 466.40 | 317.47 | 1344.71 | 2190.32 |
| `technical_support` | 75 | 0.64 | 0.90 | 0.92 | 505.68 | 342.30 | 1603.61 | 1980.39 |
| `travel` | 75 | 0.23 | 0.92 | 0.92 | 394.67 | 162.69 | 1359.75 | 1863.95 |

#### Per difficulty

| Slice | Rows | Strict exact | Semantic entities | All entities | Mean ms | p50 ms | p95 ms | p99 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `easy` | 611 | 0.31 | 0.98 | 0.98 | 130.67 | 135.12 | 195.69 | 263.34 |
| `hard` | 573 | 0.45 | 0.78 | 0.74 | 356.03 | 102.03 | 1370.94 | 1970.85 |
| `medium` | 316 | 0.50 | 0.96 | 0.94 | 245.40 | 224.85 | 650.84 | 817.34 |

#### Per declared entity category

| Slice | Rows | Strict exact | Semantic entities | All entities | Mean ms | p50 ms | p95 ms | p99 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `CARDINAL` | 145 | 0.45 | 0.86 | 0.77 | 576.71 | 456.95 | 1613.36 | 3061.11 |
| `DATE` | 135 | 0.00 | 1.00 | 1.00 | 177.44 | 139.23 | 605.00 | 740.44 |
| `DECIMAL` | 80 | 0.80 | 0.89 | 0.88 | 541.97 | 230.76 | 1614.22 | 2151.99 |
| `DIGIT_SEQUENCE` | 76 | 0.96 | 1.00 | 1.00 | 261.98 | 139.98 | 937.12 | 1612.41 |
| `EMAIL` | 65 | 0.75 | 0.82 | 0.85 | 617.33 | 425.34 | 1610.57 | 1748.33 |
| `FLIGHT_ID` | 50 | 0.00 | 0.95 | 0.94 | 292.49 | 181.89 | 714.87 | 1789.99 |
| `IP` | 54 | 0.93 | 1.00 | 1.00 | 667.47 | 455.22 | 1595.44 | 1876.70 |
| `MEASUREMENT` | 64 | 0.34 | 0.98 | 0.97 | 221.15 | 136.50 | 690.40 | 867.66 |
| `MONEY` | 95 | 0.66 | 0.76 | 0.72 | 266.48 | 168.29 | 722.17 | 2045.64 |
| `ORDER_ID` | 61 | 0.00 | 0.92 | 0.90 | 326.65 | 159.13 | 1630.69 | 2052.51 |
| `ORDINAL` | 145 | 0.61 | 0.98 | 0.97 | 342.69 | 215.78 | 1080.20 | 1683.11 |
| `PERCENT` | 50 | 0.00 | 1.00 | 1.00 | 123.39 | 89.59 | 333.55 | 627.64 |
| `PHONE` | 106 | 0.00 | 0.99 | 0.99 | 314.57 | 245.95 | 968.70 | 1944.96 |
| `REFERENCE_ID` | 151 | 0.01 | 0.84 | 0.82 | 213.18 | 134.78 | 513.48 | 2680.82 |
| `TIME` | 165 | 0.05 | 0.84 | 0.82 | 129.10 | 92.00 | 341.90 | 782.96 |
| `URL` | 79 | 0.57 | 0.70 | 0.62 | 779.39 | 731.42 | 1881.79 | 3909.86 |
| `VERSION` | 55 | 0.62 | 0.92 | 0.89 | 284.84 | 98.23 | 974.10 | 1107.72 |
| `ZIP_CODE` | 64 | 0.86 | 1.00 | 1.00 | 268.93 | 136.73 | 826.51 | 1663.93 |

#### Collision and compositional metrics

- Collision pairs: **175**
- Strict pair accuracy: **0.03**
- Semantic pair accuracy: **0.43**
- Multi-entity span accuracy: **0.93**
- Multi-entity all-correct accuracy: **0.89**
- Voice-agent group accuracy (strict): **0.34**
- Standalone negative KEEP preservation: **0.94**
- All no-span KEEP preservation: **0.81**

Collision-family metrics:

| Family | Rows | Strict exact | Semantic entities | Mean ms | p95 ms |
|---|---:|---:|---:|---:|---:|
| `date_vs_identifier` | 50 | 0.00 | 1.00 | 76.43 | 80.72 |
| `false_positive_keep` | 50 | 0.94 | — | 41.47 | 126.70 |
| `identifier_vs_time` | 50 | 0.14 | 0.52 | 82.29 | 92.31 |
| `military_time_vs_cardinal` | 50 | 0.06 | 0.06 | 84.80 | 101.06 |
| `money_vs_time` | 50 | 0.00 | 0.50 | 105.53 | 127.54 |
| `ordinal_vs_plain` | 50 | 0.62 | 1.00 | 73.40 | 90.09 |
| `phone_vs_digits` | 50 | 0.50 | 1.00 | 122.04 | 145.33 |
| `url_vs_plain` | 50 | 0.50 | 0.00 | 68.97 | 79.01 |

#### Error categories

`{"false_normalization": 19, "missed_edit": 37, "wrong_rewrite": 836}`

### thutmose

Official NVIDIA itn_en_thutmose_bert artifact in an isolated worker.

- Records: **1500**
- Initialization: **1997.29 ms**
- Warm-up: **487.82 ms** (excluded from per-record latency)
- Per-record output: [`eval/voice_agent_itn/results/20260908T172500Z/thutmose/records.jsonl`](eval/voice_agent_itn/results/20260908T172500Z/thutmose/records.jsonl)
- Runtime: `{"artifact": "/Users/aryamantodkar/Documents/Git Repositories/premove/.artifacts/thutmose-cache/nemo/itn_en_thutmose_bert/bf3a085f3b78525c3b3abe09bd0c62c8/itn_en_thutmose_bert.nemo", "device": "mps", "model_target": "nemo.collections.nlp.models.text_normalization_as_tagging.thutmose_tagger.ThutmoseTaggerModel", "nemo_version": "1.9.0rc0 artifact; isolated weight-compatible loader", "warmup_in_initialization_ms": 487.820625, "worker_python": "/Users/aryamantodkar/Documents/Git Repositories/premove/.venv-thutmose/bin/python", "worker_startup_ms": 2485.089916, "worker_warmup_ms": 487.820625}`

#### Per kind

| Slice | Rows | Strict exact | Semantic entities | All entities | Mean ms | p50 ms | p95 ms | p99 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `CARDINAL` | 53 | 0.74 | 0.91 | 0.91 | 16.52 | 16.37 | 18.30 | 18.57 |
| `DATE` | 123 | 0.00 | 0.63 | 0.63 | 16.38 | 16.43 | 17.93 | 19.76 |
| `DECIMAL` | 38 | 0.82 | 1.00 | 1.00 | 16.32 | 16.30 | 17.71 | 19.01 |
| `DIGIT_SEQUENCE` | 63 | 0.17 | 0.90 | 0.90 | 16.49 | 16.34 | 18.20 | 19.12 |
| `EMAIL` | 48 | 0.00 | 0.00 | 0.00 | 16.41 | 16.29 | 18.52 | 18.82 |
| `FLIGHT_ID` | 38 | 0.00 | 0.50 | 0.50 | 16.48 | 16.39 | 18.36 | 19.70 |
| `IP` | 37 | 0.00 | 0.65 | 0.65 | 16.39 | 16.50 | 17.58 | 18.14 |
| `KEEP` | 100 | 0.92 | — | — | 16.21 | 16.23 | 17.57 | 18.20 |
| `MEASUREMENT` | 47 | 0.74 | 1.00 | 1.00 | 16.52 | 16.50 | 17.98 | 18.86 |
| `MONEY` | 83 | 0.00 | 0.49 | 0.49 | 16.43 | 16.35 | 17.74 | 19.53 |
| `MULTI` | 200 | 0.06 | 0.33 | 0.24 | 16.58 | 16.53 | 18.34 | 19.23 |
| `ORDER_ID` | 48 | 0.00 | 0.75 | 0.75 | 16.40 | 16.33 | 18.07 | 18.56 |
| `ORDINAL` | 53 | 0.58 | 0.83 | 0.83 | 16.40 | 16.49 | 17.97 | 18.56 |
| `PERCENT` | 38 | 0.68 | 1.00 | 1.00 | 16.48 | 16.57 | 18.45 | 18.85 |
| `PHONE` | 93 | 0.00 | 0.71 | 0.71 | 16.48 | 16.53 | 17.80 | 18.79 |
| `REFERENCE_ID` | 138 | 0.02 | 0.62 | 0.62 | 16.45 | 16.44 | 18.09 | 19.14 |
| `TIME` | 153 | 0.18 | 0.96 | 0.96 | 16.48 | 16.53 | 17.79 | 19.06 |
| `URL` | 62 | 0.00 | 0.00 | 0.00 | 16.25 | 16.26 | 17.26 | 17.64 |
| `VERSION` | 38 | 0.29 | 1.00 | 1.00 | 16.74 | 16.59 | 18.49 | 18.70 |
| `ZIP_CODE` | 47 | 0.32 | 0.47 | 0.47 | 16.53 | 16.38 | 18.04 | 19.51 |

#### Per benchmark group

| Slice | Rows | Strict exact | Semantic entities | All entities | Mean ms | p50 ms | p95 ms | p99 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `collision` | 350 | 0.33 | 0.68 | 0.68 | 16.36 | 16.38 | 17.80 | 18.89 |
| `multi` | 200 | 0.06 | 0.33 | 0.24 | 16.58 | 16.53 | 18.34 | 19.23 |
| `negative` | 50 | 0.90 | — | — | 16.15 | 16.05 | 17.47 | 17.76 |
| `normal` | 500 | 0.23 | 0.71 | 0.71 | 16.45 | 16.43 | 18.10 | 18.94 |
| `voice_agent` | 400 | 0.11 | 0.67 | 0.67 | 16.50 | 16.49 | 18.20 | 19.50 |

#### Per domain

| Slice | Rows | Strict exact | Semantic entities | All entities | Mean ms | p50 ms | p95 ms | p99 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `banking` | 75 | 0.20 | 0.61 | 0.69 | 16.39 | 16.34 | 18.20 | 18.98 |
| `customer_support` | 75 | 0.00 | 0.41 | 0.41 | 16.64 | 16.52 | 18.54 | 19.41 |
| `ecommerce` | 75 | 0.17 | 0.56 | 0.63 | 16.73 | 16.57 | 19.19 | 20.61 |
| `general` | 900 | 0.31 | 0.70 | 0.70 | 16.40 | 16.39 | 17.94 | 18.94 |
| `healthcare_admin` | 75 | 0.00 | 0.56 | 0.63 | 16.60 | 16.62 | 18.24 | 18.79 |
| `logistics` | 75 | 0.24 | 0.53 | 0.57 | 16.37 | 16.28 | 17.90 | 19.24 |
| `scheduling` | 75 | 0.00 | 0.41 | 0.41 | 16.54 | 16.50 | 18.23 | 19.28 |
| `technical_support` | 75 | 0.11 | 0.51 | 0.55 | 16.58 | 16.63 | 17.75 | 18.48 |
| `travel` | 75 | 0.00 | 0.34 | 0.33 | 16.39 | 16.34 | 18.01 | 19.05 |

#### Per difficulty

| Slice | Rows | Strict exact | Semantic entities | All entities | Mean ms | p50 ms | p95 ms | p99 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `easy` | 611 | 0.24 | 0.77 | 0.77 | 16.46 | 16.45 | 18.12 | 19.39 |
| `hard` | 573 | 0.20 | 0.44 | 0.46 | 16.41 | 16.42 | 17.90 | 18.95 |
| `medium` | 316 | 0.21 | 0.57 | 0.62 | 16.50 | 16.47 | 18.21 | 19.23 |

#### Per declared entity category

| Slice | Rows | Strict exact | Semantic entities | All entities | Mean ms | p50 ms | p95 ms | p99 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `CARDINAL` | 145 | 0.30 | 0.41 | 0.46 | 16.54 | 16.51 | 18.32 | 19.11 |
| `DATE` | 135 | 0.00 | 0.54 | 0.57 | 16.37 | 16.38 | 17.93 | 19.56 |
| `DECIMAL` | 80 | 0.42 | 0.47 | 0.57 | 16.42 | 16.41 | 17.82 | 18.85 |
| `DIGIT_SEQUENCE` | 76 | 0.14 | 0.68 | 0.79 | 16.51 | 16.38 | 18.01 | 19.09 |
| `EMAIL` | 65 | 0.00 | 0.09 | 0.00 | 16.37 | 16.29 | 18.33 | 18.77 |
| `FLIGHT_ID` | 50 | 0.00 | 0.38 | 0.42 | 16.50 | 16.46 | 18.16 | 19.64 |
| `IP` | 54 | 0.00 | 0.41 | 0.50 | 16.45 | 16.53 | 17.73 | 18.08 |
| `MEASUREMENT` | 64 | 0.58 | 0.64 | 0.80 | 16.39 | 16.45 | 17.66 | 18.75 |
| `MONEY` | 95 | 0.00 | 0.43 | 0.46 | 16.48 | 16.36 | 18.12 | 19.23 |
| `ORDER_ID` | 61 | 0.00 | 0.53 | 0.62 | 16.46 | 16.34 | 18.09 | 18.83 |
| `ORDINAL` | 145 | 0.27 | 0.46 | 0.49 | 16.56 | 16.51 | 18.68 | 19.17 |
| `PERCENT` | 50 | 0.52 | 0.82 | 0.92 | 16.46 | 16.53 | 18.28 | 18.80 |
| `PHONE` | 106 | 0.00 | 0.62 | 0.65 | 16.50 | 16.54 | 17.81 | 19.34 |
| `REFERENCE_ID` | 151 | 0.02 | 0.58 | 0.60 | 16.50 | 16.47 | 18.20 | 19.45 |
| `TIME` | 165 | 0.16 | 0.85 | 0.90 | 16.49 | 16.53 | 17.79 | 18.96 |
| `URL` | 79 | 0.00 | 0.12 | 0.00 | 16.35 | 16.42 | 17.41 | 18.50 |
| `VERSION` | 55 | 0.20 | 0.62 | 0.78 | 16.75 | 16.69 | 18.40 | 18.66 |
| `ZIP_CODE` | 64 | 0.31 | 0.47 | 0.44 | 16.64 | 16.55 | 18.22 | 19.51 |

#### Collision and compositional metrics

- Collision pairs: **175**
- Strict pair accuracy: **0.16**
- Semantic pair accuracy: **0.49**
- Multi-entity span accuracy: **0.33**
- Multi-entity all-correct accuracy: **0.24**
- Voice-agent group accuracy (strict): **0.11**
- Standalone negative KEEP preservation: **0.90**
- All no-span KEEP preservation: **0.92**

Collision-family metrics:

| Family | Rows | Strict exact | Semantic entities | Mean ms | p95 ms |
|---|---:|---:|---:|---:|---:|
| `date_vs_identifier` | 50 | 0.00 | 1.00 | 16.32 | 18.00 |
| `false_positive_keep` | 50 | 0.90 | — | 16.15 | 17.47 |
| `identifier_vs_time` | 50 | 0.14 | 0.50 | 16.32 | 17.52 |
| `military_time_vs_cardinal` | 50 | 0.86 | 0.90 | 16.54 | 17.78 |
| `money_vs_time` | 50 | 0.00 | 0.50 | 16.54 | 17.77 |
| `ordinal_vs_plain` | 50 | 0.62 | 0.68 | 16.34 | 17.56 |
| `phone_vs_digits` | 50 | 0.22 | 0.86 | 16.32 | 17.91 |
| `url_vs_plain` | 50 | 0.50 | 0.00 | 16.15 | 17.42 |

#### Error categories

`{"false_normalization": 8, "missed_edit": 33, "wrong_rewrite": 1127}`

## Latency methodology

Latency is measured with a monotonic high-resolution clock around each individual
record. Initialization and one explicit warm-up call are reported separately. The
per-record `latency_ms` includes the adapter call; for Premove ITN the record also
stores candidate enumeration, token encoding, model forward, and exact decoding
components. Thutmose stores both worker inference time (`backend_latency_ms`) and the
main-process request round trip (`request_latency_ms`). The comparison table uses the
adapter-call `latency_ms` field so all three rows have the same boundary.
For Thutmose, model warm-up happens before the worker emits `ready`; its measured
warm-up interval is subtracted from initialization to avoid double counting.

Percentiles use linear interpolation over the sorted per-record measurements. Throughput
is the number of records divided by the sum of per-record latencies. These measurements
are host-specific and should not be treated as a deployment SLA.

## Reproduction

The temporary evaluation harness was used for this frozen run and intentionally
removed after completion. The retained per-record JSONL files and aggregate
metrics are the audit record for this run. The runtime blocks identify the exact
Premove checkpoint, Thutmose artifact, isolated worker, and host device used.

## Limitations

- This dataset is a balanced synthetic stress suite, not an IID production sample.
- The frozen dataset status remains `pending_independent_review`; this run does not
  change that dataset status or alter its gold labels.
- The current Thutmose Python environment contains NeMo 2.x, while the cached artifact
  targets NeMo 1.9.0rc0. The isolated worker loads the artifact weights and performs
  the documented tagger inference path without passing gold fields to the model.
- No local Docker image was available for this run. Thutmose therefore used the cached
  official `.nemo` artifact in the persistent isolated worker; the exact artifact and
  worker interpreter are recorded in its `metrics.json` runtime block.
