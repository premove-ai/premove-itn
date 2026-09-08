# VoiceAgent ITN benchmark results

Run completed: `2026-09-08T15:17:42.165167Z`  
Dataset: `eval/voice_agent_itn/voice_agent_eval.jsonl`  
Rows: **1500**  

## Executive summary

This is the release latency correction for First Evaluation. Accuracy was already observed.
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
| **premove-itn** | 1500 | 0.41 | 0.90 | 0.88 | 56.49 | 55.73 | 71.40 | 89.69 | 17.70 |
| **thutmose** | 1500 | 0.22 | 0.59 | 0.63 | 15.98 | 15.87 | 17.47 | 18.32 | 62.58 |
| **text-processing-rs** | 1500 | 0.17 | 0.56 | 0.58 | 0.14 | 0.11 | 0.38 | 0.57 | 7054.00 |

Strict exact is the canonical full-sentence match. Semantic entity accuracy is
entity-level and category-aware; it preserves numeric value, currency, unit, digit
order, phone dialing form, and identifier case policy. No-span rows do not enter the
semantic-entity denominator. `all-entities rows` is the fraction of rows for which
every declared entity was semantically correct.
Structured identifiers ignore case and harmless whitespace/separator formatting for
semantic scoring; strict exact accuracy still requires the complete expected sentence.

## Detailed per-backend results

### premove-itn

Production structured-value 20k contextual candidate scorer and decoder.

- Records: **1500**
- Initialization: **15067.53 ms**
- Warm-up: **1114.52 ms** (excluded from per-record latency)
- Per-record output: [records.jsonl](premove-itn/records.jsonl)
- Runtime: `{"checkpoint": "data/models/structured_value_selected_20k/checkpoint.pt", "device": "mps", "model_name": "microsoft/deberta-v3-large"}`

#### Per kind

| Slice | Rows | Strict exact | Semantic entities | All entities | Mean ms | p50 ms | p95 ms | p99 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `CARDINAL` | 53 | 0.53 | 0.53 | 0.53 | 53.63 | 52.99 | 58.80 | 61.11 |
| `DATE` | 123 | 0.00 | 1.00 | 1.00 | 54.76 | 54.10 | 59.15 | 77.64 |
| `DECIMAL` | 38 | 1.00 | 1.00 | 1.00 | 55.33 | 54.82 | 63.07 | 68.86 |
| `DIGIT_SEQUENCE` | 63 | 1.00 | 1.00 | 1.00 | 55.85 | 55.52 | 59.70 | 67.45 |
| `EMAIL` | 48 | 0.96 | 0.96 | 0.96 | 61.17 | 60.26 | 70.57 | 78.39 |
| `FLIGHT_ID` | 38 | 0.00 | 0.95 | 0.95 | 57.33 | 56.87 | 67.41 | 70.77 |
| `IP` | 37 | 1.00 | 1.00 | 1.00 | 62.43 | 62.17 | 66.27 | 70.47 |
| `KEEP` | 100 | 0.81 | — | — | 36.79 | 51.28 | 56.18 | 86.74 |
| `MEASUREMENT` | 47 | 0.30 | 0.98 | 0.98 | 54.36 | 53.49 | 58.47 | 76.07 |
| `MONEY` | 83 | 0.63 | 0.69 | 0.69 | 57.20 | 56.64 | 64.16 | 77.47 |
| `MULTI` | 200 | 0.42 | 0.93 | 0.89 | 67.14 | 63.69 | 89.69 | 112.14 |
| `ORDER_ID` | 48 | 0.00 | 0.94 | 0.94 | 58.88 | 56.56 | 62.11 | 117.86 |
| `ORDINAL` | 53 | 0.92 | 1.00 | 1.00 | 53.17 | 53.52 | 56.71 | 58.66 |
| `PERCENT` | 38 | 0.00 | 1.00 | 1.00 | 53.73 | 53.01 | 59.66 | 67.09 |
| `PHONE` | 93 | 0.00 | 1.00 | 1.00 | 59.04 | 59.33 | 65.73 | 69.63 |
| `REFERENCE_ID` | 138 | 0.01 | 0.80 | 0.80 | 56.36 | 55.38 | 64.55 | 90.19 |
| `TIME` | 153 | 0.06 | 0.85 | 0.85 | 54.83 | 54.48 | 59.20 | 70.78 |
| `URL` | 62 | 0.60 | 0.60 | 0.60 | 60.00 | 61.54 | 73.49 | 77.49 |
| `VERSION` | 38 | 0.55 | 0.84 | 0.84 | 54.32 | 54.33 | 59.66 | 60.42 |
| `ZIP_CODE` | 47 | 1.00 | 1.00 | 1.00 | 54.90 | 54.86 | 59.02 | 62.15 |

#### Per benchmark group

| Slice | Rows | Strict exact | Semantic entities | All entities | Mean ms | p50 ms | p95 ms | p99 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `collision` | 350 | 0.26 | 0.60 | 0.60 | 53.69 | 53.32 | 58.61 | 75.94 |
| `multi` | 200 | 0.42 | 0.93 | 0.89 | 67.14 | 63.69 | 89.69 | 112.14 |
| `negative` | 50 | 0.94 | — | — | 19.86 | 0.37 | 55.67 | 61.60 |
| `normal` | 500 | 0.50 | 0.97 | 0.97 | 57.13 | 56.00 | 65.76 | 83.23 |
| `voice_agent` | 400 | 0.34 | 0.99 | 0.99 | 57.41 | 56.41 | 67.78 | 72.96 |

#### Domain labels across all groups (not voice-agent domain evidence)

| Slice | Rows | Strict exact | Semantic entities | All entities | Mean ms | p50 ms | p95 ms | p99 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `banking` | 75 | 0.45 | 1.00 | 1.00 | 60.97 | 58.29 | 81.04 | 85.89 |
| `customer_support` | 75 | 0.25 | 0.99 | 0.99 | 60.04 | 58.23 | 73.56 | 89.71 |
| `ecommerce` | 75 | 0.47 | 0.99 | 0.99 | 58.65 | 57.49 | 69.17 | 90.90 |
| `general` | 900 | 0.43 | 0.83 | 0.83 | 53.72 | 54.35 | 64.03 | 79.52 |
| `healthcare_admin` | 75 | 0.21 | 0.98 | 0.97 | 59.39 | 57.70 | 70.65 | 86.44 |
| `logistics` | 75 | 0.39 | 0.96 | 0.95 | 59.43 | 56.24 | 80.20 | 101.39 |
| `scheduling` | 75 | 0.28 | 0.95 | 0.93 | 63.07 | 60.32 | 75.79 | 110.82 |
| `technical_support` | 75 | 0.64 | 0.90 | 0.92 | 63.29 | 61.21 | 81.04 | 91.45 |
| `travel` | 75 | 0.23 | 0.92 | 0.92 | 60.36 | 56.67 | 78.28 | 88.35 |

#### Per difficulty

| Slice | Rows | Strict exact | Semantic entities | All entities | Mean ms | p50 ms | p95 ms | p99 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `easy` | 611 | 0.31 | 0.98 | 0.98 | 55.84 | 55.06 | 61.19 | 79.74 |
| `hard` | 573 | 0.45 | 0.78 | 0.74 | 59.10 | 55.78 | 79.49 | 96.06 |
| `medium` | 316 | 0.50 | 0.96 | 0.94 | 53.03 | 58.00 | 67.59 | 72.98 |

#### Rows containing each category (all entities in those rows)

| Slice | Rows | Strict exact | Semantic entities | All entities | Mean ms | p50 ms | p95 ms | p99 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `CARDINAL` | 145 | 0.45 | 0.86 | 0.77 | 63.52 | 60.68 | 88.73 | 112.06 |
| `DATE` | 135 | 0.00 | 1.00 | 1.00 | 55.47 | 54.78 | 63.86 | 76.49 |
| `DECIMAL` | 80 | 0.80 | 0.89 | 0.88 | 62.08 | 59.59 | 83.56 | 95.33 |
| `DIGIT_SEQUENCE` | 76 | 0.96 | 1.00 | 1.00 | 58.00 | 56.21 | 70.74 | 84.75 |
| `EMAIL` | 65 | 0.75 | 0.82 | 0.85 | 64.68 | 61.51 | 80.87 | 86.96 |
| `FLIGHT_ID` | 50 | 0.00 | 0.95 | 0.94 | 59.47 | 57.84 | 71.18 | 84.93 |
| `IP` | 54 | 0.93 | 1.00 | 1.00 | 65.90 | 63.60 | 81.12 | 88.51 |
| `MEASUREMENT` | 64 | 0.34 | 0.98 | 0.97 | 57.13 | 54.21 | 71.67 | 83.49 |
| `MONEY` | 95 | 0.66 | 0.76 | 0.72 | 58.28 | 56.83 | 68.13 | 82.55 |
| `ORDER_ID` | 61 | 0.00 | 0.92 | 0.90 | 61.40 | 57.63 | 87.46 | 115.86 |
| `ORDINAL` | 145 | 0.61 | 0.98 | 0.97 | 59.31 | 56.89 | 75.54 | 91.94 |
| `PERCENT` | 50 | 0.00 | 1.00 | 1.00 | 54.91 | 53.28 | 67.63 | 69.14 |
| `PHONE` | 106 | 0.00 | 0.99 | 0.99 | 60.79 | 59.99 | 78.07 | 89.59 |
| `REFERENCE_ID` | 151 | 0.01 | 0.84 | 0.82 | 57.91 | 55.61 | 74.91 | 118.00 |
| `TIME` | 165 | 0.05 | 0.84 | 0.82 | 55.49 | 54.62 | 64.15 | 73.73 |
| `URL` | 79 | 0.57 | 0.70 | 0.62 | 65.97 | 63.65 | 105.97 | 116.59 |
| `VERSION` | 55 | 0.62 | 0.92 | 0.89 | 58.42 | 56.25 | 74.89 | 87.46 |
| `ZIP_CODE` | 64 | 0.86 | 1.00 | 1.00 | 57.62 | 55.86 | 69.19 | 85.09 |

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
| `date_vs_identifier` | 50 | 0.00 | 1.00 | 51.71 | 55.60 |
| `false_positive_keep` | 50 | 0.94 | — | 19.86 | 55.67 |
| `identifier_vs_time` | 50 | 0.14 | 0.52 | 54.09 | 59.01 |
| `military_time_vs_cardinal` | 50 | 0.06 | 0.06 | 53.40 | 58.50 |
| `money_vs_time` | 50 | 0.00 | 0.50 | 54.84 | 58.02 |
| `ordinal_vs_plain` | 50 | 0.62 | 1.00 | 53.78 | 56.36 |
| `phone_vs_digits` | 50 | 0.50 | 1.00 | 55.92 | 62.34 |
| `url_vs_plain` | 50 | 0.50 | 0.00 | 52.06 | 56.09 |

#### Error categories

`{"false_normalization": 19, "missed_edit": 37, "wrong_rewrite": 836}`

### thutmose

Official NVIDIA itn_en_thutmose_bert artifact in an isolated worker.

- Records: **1500**
- Initialization: **2790.17 ms**
- Warm-up: **278.12 ms** (excluded from per-record latency)
- Per-record output: [records.jsonl](thutmose/records.jsonl)
- Runtime: `{"artifact": "/Users/aryamantodkar/Documents/Git Repositories/premove/.artifacts/thutmose-cache/nemo/itn_en_thutmose_bert/bf3a085f3b78525c3b3abe09bd0c62c8/itn_en_thutmose_bert.nemo", "device": "mps", "model_target": "nemo.collections.nlp.models.text_normalization_as_tagging.thutmose_tagger.ThutmoseTaggerModel", "nemo_version": "1.9.0rc0 artifact; isolated weight-compatible loader", "warmup_in_initialization_ms": 930.17675, "worker_python": "/Users/aryamantodkar/Documents/Git Repositories/premove/.venv-thutmose/bin/python", "worker_startup_ms": 3719.859875, "worker_warmup_ms": 930.17675}`

#### Per kind

| Slice | Rows | Strict exact | Semantic entities | All entities | Mean ms | p50 ms | p95 ms | p99 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `CARDINAL` | 53 | 0.74 | 0.91 | 0.91 | 15.98 | 15.85 | 17.22 | 17.80 |
| `DATE` | 123 | 0.00 | 0.63 | 0.63 | 16.03 | 15.96 | 17.59 | 18.18 |
| `DECIMAL` | 38 | 0.82 | 1.00 | 1.00 | 15.93 | 15.96 | 17.29 | 17.66 |
| `DIGIT_SEQUENCE` | 63 | 0.17 | 0.90 | 0.90 | 15.92 | 15.87 | 17.62 | 18.18 |
| `EMAIL` | 48 | 0.00 | 0.00 | 0.00 | 16.08 | 15.90 | 18.18 | 19.12 |
| `FLIGHT_ID` | 38 | 0.00 | 0.50 | 0.50 | 15.98 | 15.79 | 17.52 | 18.17 |
| `IP` | 37 | 0.00 | 0.65 | 0.65 | 15.86 | 15.74 | 17.25 | 17.98 |
| `KEEP` | 100 | 0.92 | — | — | 15.89 | 15.79 | 17.39 | 17.72 |
| `MEASUREMENT` | 47 | 0.74 | 1.00 | 1.00 | 16.00 | 15.75 | 17.85 | 18.16 |
| `MONEY` | 83 | 0.00 | 0.49 | 0.49 | 16.34 | 16.21 | 17.75 | 21.95 |
| `MULTI` | 200 | 0.06 | 0.33 | 0.24 | 16.12 | 16.04 | 17.64 | 18.61 |
| `ORDER_ID` | 48 | 0.00 | 0.75 | 0.75 | 15.97 | 15.92 | 17.40 | 17.78 |
| `ORDINAL` | 53 | 0.58 | 0.83 | 0.83 | 15.73 | 15.59 | 17.21 | 17.44 |
| `PERCENT` | 38 | 0.68 | 1.00 | 1.00 | 15.91 | 15.80 | 17.41 | 17.73 |
| `PHONE` | 93 | 0.00 | 0.71 | 0.71 | 16.01 | 15.81 | 17.33 | 18.99 |
| `REFERENCE_ID` | 138 | 0.02 | 0.62 | 0.62 | 15.91 | 15.89 | 17.34 | 17.46 |
| `TIME` | 153 | 0.18 | 0.96 | 0.96 | 15.93 | 15.80 | 17.52 | 18.24 |
| `URL` | 62 | 0.00 | 0.00 | 0.00 | 15.65 | 15.62 | 16.99 | 17.28 |
| `VERSION` | 38 | 0.29 | 1.00 | 1.00 | 16.00 | 15.77 | 18.17 | 18.85 |
| `ZIP_CODE` | 47 | 0.32 | 0.47 | 0.47 | 15.98 | 16.09 | 17.34 | 17.41 |

#### Per benchmark group

| Slice | Rows | Strict exact | Semantic entities | All entities | Mean ms | p50 ms | p95 ms | p99 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `collision` | 350 | 0.33 | 0.68 | 0.68 | 15.88 | 15.74 | 17.40 | 18.09 |
| `multi` | 200 | 0.06 | 0.33 | 0.24 | 16.12 | 16.04 | 17.64 | 18.61 |
| `negative` | 50 | 0.90 | — | — | 15.88 | 15.79 | 17.24 | 17.39 |
| `normal` | 500 | 0.23 | 0.71 | 0.71 | 15.96 | 15.86 | 17.52 | 18.32 |
| `voice_agent` | 400 | 0.11 | 0.67 | 0.67 | 16.04 | 15.95 | 17.48 | 18.20 |

#### Domain labels across all groups (not voice-agent domain evidence)

| Slice | Rows | Strict exact | Semantic entities | All entities | Mean ms | p50 ms | p95 ms | p99 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `banking` | 75 | 0.20 | 0.61 | 0.69 | 16.03 | 15.96 | 17.85 | 18.30 |
| `customer_support` | 75 | 0.00 | 0.41 | 0.41 | 16.06 | 15.96 | 17.66 | 18.87 |
| `ecommerce` | 75 | 0.17 | 0.56 | 0.63 | 16.00 | 15.96 | 17.38 | 17.66 |
| `general` | 900 | 0.31 | 0.70 | 0.70 | 15.92 | 15.80 | 17.40 | 18.26 |
| `healthcare_admin` | 75 | 0.00 | 0.56 | 0.63 | 15.89 | 15.83 | 17.06 | 17.66 |
| `logistics` | 75 | 0.24 | 0.53 | 0.57 | 16.22 | 16.11 | 17.76 | 18.44 |
| `scheduling` | 75 | 0.00 | 0.41 | 0.41 | 16.07 | 16.06 | 17.52 | 18.12 |
| `technical_support` | 75 | 0.11 | 0.51 | 0.55 | 16.08 | 15.97 | 17.54 | 18.49 |
| `travel` | 75 | 0.00 | 0.34 | 0.33 | 16.19 | 16.09 | 17.48 | 18.14 |

#### Per difficulty

| Slice | Rows | Strict exact | Semantic entities | All entities | Mean ms | p50 ms | p95 ms | p99 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `easy` | 611 | 0.24 | 0.77 | 0.77 | 16.00 | 15.93 | 17.44 | 18.10 |
| `hard` | 573 | 0.20 | 0.44 | 0.46 | 15.91 | 15.78 | 17.50 | 18.40 |
| `medium` | 316 | 0.21 | 0.57 | 0.62 | 16.05 | 15.95 | 17.50 | 18.74 |

#### Rows containing each category (all entities in those rows)

| Slice | Rows | Strict exact | Semantic entities | All entities | Mean ms | p50 ms | p95 ms | p99 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `CARDINAL` | 145 | 0.30 | 0.41 | 0.46 | 16.08 | 16.00 | 17.52 | 18.70 |
| `DATE` | 135 | 0.00 | 0.54 | 0.57 | 16.05 | 15.97 | 17.68 | 18.52 |
| `DECIMAL` | 80 | 0.42 | 0.47 | 0.57 | 16.03 | 16.08 | 17.32 | 18.32 |
| `DIGIT_SEQUENCE` | 76 | 0.14 | 0.68 | 0.79 | 15.88 | 15.81 | 17.73 | 18.13 |
| `EMAIL` | 65 | 0.00 | 0.09 | 0.00 | 16.04 | 15.91 | 17.97 | 18.93 |
| `FLIGHT_ID` | 50 | 0.00 | 0.38 | 0.42 | 16.05 | 15.96 | 17.39 | 18.12 |
| `IP` | 54 | 0.00 | 0.41 | 0.50 | 15.89 | 15.76 | 17.51 | 18.08 |
| `MEASUREMENT` | 64 | 0.58 | 0.64 | 0.80 | 16.11 | 16.02 | 17.46 | 18.14 |
| `MONEY` | 95 | 0.00 | 0.43 | 0.46 | 16.30 | 15.99 | 17.87 | 21.16 |
| `ORDER_ID` | 61 | 0.00 | 0.53 | 0.62 | 16.08 | 16.10 | 17.48 | 17.99 |
| `ORDINAL` | 145 | 0.27 | 0.46 | 0.49 | 15.94 | 15.88 | 17.48 | 18.26 |
| `PERCENT` | 50 | 0.52 | 0.82 | 0.92 | 15.89 | 15.70 | 17.33 | 17.72 |
| `PHONE` | 106 | 0.00 | 0.62 | 0.65 | 16.00 | 15.81 | 17.30 | 18.36 |
| `REFERENCE_ID` | 151 | 0.02 | 0.58 | 0.60 | 15.94 | 15.92 | 17.34 | 17.45 |
| `TIME` | 165 | 0.16 | 0.85 | 0.90 | 15.96 | 15.85 | 17.58 | 18.21 |
| `URL` | 79 | 0.00 | 0.12 | 0.00 | 15.78 | 15.81 | 17.21 | 17.43 |
| `VERSION` | 55 | 0.20 | 0.62 | 0.78 | 16.12 | 15.98 | 18.18 | 19.08 |
| `ZIP_CODE` | 64 | 0.31 | 0.47 | 0.44 | 16.01 | 16.09 | 17.42 | 18.58 |

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
| `date_vs_identifier` | 50 | 0.00 | 1.00 | 15.77 | 17.07 |
| `false_positive_keep` | 50 | 0.90 | — | 15.88 | 17.24 |
| `identifier_vs_time` | 50 | 0.14 | 0.50 | 15.84 | 17.28 |
| `military_time_vs_cardinal` | 50 | 0.86 | 0.90 | 15.91 | 17.30 |
| `money_vs_time` | 50 | 0.00 | 0.50 | 16.14 | 17.83 |
| `ordinal_vs_plain` | 50 | 0.62 | 0.68 | 15.82 | 17.41 |
| `phone_vs_digits` | 50 | 0.22 | 0.86 | 15.97 | 17.88 |
| `url_vs_plain` | 50 | 0.50 | 0.00 | 15.69 | 17.24 |

#### Error categories

`{"false_normalization": 8, "missed_edit": 33, "wrong_rewrite": 1127}`

### text-processing-rs

Compiled upstream English sentence normalizer exposed by premove_itn.

- Records: **1500**
- Initialization: **0.01 ms**
- Warm-up: **0.58 ms** (excluded from per-record latency)
- Per-record output: [records.jsonl](text-processing-rs/records.jsonl)
- Runtime: `{"runtime": "premove_itn._rust.baseline_normalize_sentence"}`

#### Per kind

| Slice | Rows | Strict exact | Semantic entities | All entities | Mean ms | p50 ms | p95 ms | p99 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `CARDINAL` | 53 | 0.58 | 0.58 | 0.58 | 0.08 | 0.07 | 0.13 | 0.14 |
| `DATE` | 123 | 0.00 | 1.00 | 1.00 | 0.12 | 0.13 | 0.17 | 0.20 |
| `DECIMAL` | 38 | 1.00 | 1.00 | 1.00 | 0.10 | 0.10 | 0.14 | 0.14 |
| `DIGIT_SEQUENCE` | 63 | 0.00 | 0.37 | 0.37 | 0.12 | 0.11 | 0.17 | 0.17 |
| `EMAIL` | 48 | 0.00 | 0.00 | 0.00 | 0.06 | 0.06 | 0.08 | 0.08 |
| `FLIGHT_ID` | 38 | 0.00 | 0.50 | 0.50 | 0.19 | 0.18 | 0.32 | 0.37 |
| `IP` | 37 | 0.19 | 0.19 | 0.19 | 0.13 | 0.11 | 0.31 | 0.41 |
| `KEEP` | 100 | 0.38 | — | — | 0.05 | 0.05 | 0.10 | 0.13 |
| `MEASUREMENT` | 47 | 1.00 | 1.00 | 1.00 | 0.10 | 0.08 | 0.15 | 0.20 |
| `MONEY` | 83 | 0.12 | 0.12 | 0.12 | 0.18 | 0.16 | 0.33 | 0.39 |
| `MULTI` | 200 | 0.01 | 0.30 | 0.14 | 0.35 | 0.34 | 0.60 | 0.72 |
| `ORDER_ID` | 48 | 0.00 | 0.62 | 0.62 | 0.15 | 0.13 | 0.24 | 0.29 |
| `ORDINAL` | 53 | 1.00 | 1.00 | 1.00 | 0.05 | 0.04 | 0.08 | 0.09 |
| `PERCENT` | 38 | 0.00 | 1.00 | 1.00 | 0.06 | 0.06 | 0.09 | 0.10 |
| `PHONE` | 93 | 0.00 | 0.51 | 0.51 | 0.18 | 0.17 | 0.34 | 0.36 |
| `REFERENCE_ID` | 138 | 0.03 | 0.57 | 0.57 | 0.12 | 0.12 | 0.23 | 0.25 |
| `TIME` | 153 | 0.00 | 0.82 | 0.82 | 0.08 | 0.08 | 0.14 | 0.15 |
| `URL` | 62 | 0.02 | 0.45 | 0.45 | 0.09 | 0.08 | 0.24 | 0.28 |
| `VERSION` | 38 | 0.45 | 1.00 | 1.00 | 0.08 | 0.07 | 0.17 | 0.17 |
| `ZIP_CODE` | 47 | 0.00 | 1.00 | 1.00 | 0.11 | 0.11 | 0.15 | 0.17 |

#### Per benchmark group

| Slice | Rows | Strict exact | Semantic entities | All entities | Mean ms | p50 ms | p95 ms | p99 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `collision` | 350 | 0.16 | 0.74 | 0.74 | 0.07 | 0.06 | 0.13 | 0.13 |
| `multi` | 200 | 0.01 | 0.30 | 0.14 | 0.35 | 0.34 | 0.60 | 0.72 |
| `negative` | 50 | 0.72 | — | — | 0.04 | 0.03 | 0.10 | 0.15 |
| `normal` | 500 | 0.22 | 0.57 | 0.57 | 0.11 | 0.09 | 0.24 | 0.32 |
| `voice_agent` | 400 | 0.11 | 0.68 | 0.68 | 0.15 | 0.14 | 0.31 | 0.36 |

#### Domain labels across all groups (not voice-agent domain evidence)

| Slice | Rows | Strict exact | Semantic entities | All entities | Mean ms | p50 ms | p95 ms | p99 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `banking` | 75 | 0.00 | 0.33 | 0.36 | 0.26 | 0.21 | 0.54 | 0.61 |
| `customer_support` | 75 | 0.01 | 0.36 | 0.40 | 0.21 | 0.18 | 0.42 | 0.59 |
| `ecommerce` | 75 | 0.00 | 0.68 | 0.65 | 0.22 | 0.17 | 0.56 | 0.58 |
| `general` | 900 | 0.23 | 0.64 | 0.64 | 0.09 | 0.07 | 0.21 | 0.31 |
| `healthcare_admin` | 75 | 0.13 | 0.53 | 0.48 | 0.25 | 0.17 | 0.55 | 0.66 |
| `logistics` | 75 | 0.13 | 0.60 | 0.65 | 0.19 | 0.14 | 0.49 | 0.54 |
| `scheduling` | 75 | 0.01 | 0.47 | 0.43 | 0.23 | 0.20 | 0.58 | 0.72 |
| `technical_support` | 75 | 0.17 | 0.44 | 0.45 | 0.19 | 0.14 | 0.46 | 0.54 |
| `travel` | 75 | 0.13 | 0.45 | 0.57 | 0.20 | 0.16 | 0.42 | 0.45 |

#### Per difficulty

| Slice | Rows | Strict exact | Semantic entities | All entities | Mean ms | p50 ms | p95 ms | p99 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `easy` | 611 | 0.21 | 0.81 | 0.81 | 0.12 | 0.12 | 0.20 | 0.32 |
| `hard` | 573 | 0.11 | 0.42 | 0.45 | 0.13 | 0.07 | 0.49 | 0.64 |
| `medium` | 316 | 0.17 | 0.39 | 0.29 | 0.20 | 0.18 | 0.41 | 0.49 |

#### Rows containing each category (all entities in those rows)

| Slice | Rows | Strict exact | Semantic entities | All entities | Mean ms | p50 ms | p95 ms | p99 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `CARDINAL` | 145 | 0.21 | 0.32 | 0.23 | 0.27 | 0.27 | 0.59 | 0.66 |
| `DATE` | 135 | 0.00 | 0.87 | 0.92 | 0.15 | 0.13 | 0.43 | 0.53 |
| `DECIMAL` | 80 | 0.50 | 0.51 | 0.56 | 0.23 | 0.22 | 0.50 | 0.59 |
| `DIGIT_SEQUENCE` | 76 | 0.00 | 0.31 | 0.30 | 0.16 | 0.12 | 0.53 | 0.58 |
| `EMAIL` | 65 | 0.00 | 0.00 | 0.00 | 0.12 | 0.07 | 0.34 | 0.43 |
| `FLIGHT_ID` | 50 | 0.00 | 0.50 | 0.42 | 0.22 | 0.19 | 0.43 | 0.49 |
| `IP` | 54 | 0.13 | 0.09 | 0.13 | 0.25 | 0.12 | 0.66 | 0.73 |
| `MEASUREMENT` | 64 | 0.75 | 0.79 | 0.81 | 0.16 | 0.10 | 0.38 | 0.47 |
| `MONEY` | 95 | 0.12 | 0.17 | 0.12 | 0.22 | 0.17 | 0.47 | 0.67 |
| `ORDER_ID` | 61 | 0.00 | 0.48 | 0.56 | 0.19 | 0.15 | 0.42 | 0.57 |
| `ORDINAL` | 145 | 0.37 | 0.53 | 0.50 | 0.22 | 0.20 | 0.53 | 0.71 |
| `PERCENT` | 50 | 0.00 | 0.91 | 0.88 | 0.09 | 0.07 | 0.29 | 0.37 |
| `PHONE` | 106 | 0.00 | 0.46 | 0.46 | 0.20 | 0.18 | 0.38 | 0.50 |
| `REFERENCE_ID` | 151 | 0.03 | 0.48 | 0.52 | 0.15 | 0.14 | 0.33 | 0.50 |
| `TIME` | 165 | 0.00 | 0.77 | 0.78 | 0.10 | 0.08 | 0.29 | 0.41 |
| `URL` | 79 | 0.01 | 0.28 | 0.35 | 0.15 | 0.12 | 0.39 | 0.61 |
| `VERSION` | 55 | 0.31 | 0.69 | 0.71 | 0.20 | 0.08 | 0.51 | 0.60 |
| `ZIP_CODE` | 64 | 0.00 | 0.74 | 0.84 | 0.17 | 0.13 | 0.41 | 0.55 |

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
| `date_vs_identifier` | 50 | 0.00 | 1.00 | 0.06 | 0.06 |
| `false_positive_keep` | 50 | 0.72 | — | 0.04 | 0.10 |
| `identifier_vs_time` | 50 | 0.08 | 0.54 | 0.06 | 0.08 |
| `military_time_vs_cardinal` | 50 | 0.50 | 0.50 | 0.07 | 0.09 |
| `money_vs_time` | 50 | 0.00 | 0.50 | 0.10 | 0.13 |
| `ordinal_vs_plain` | 50 | 0.54 | 1.00 | 0.05 | 0.08 |
| `phone_vs_digits` | 50 | 0.00 | 0.92 | 0.09 | 0.12 |
| `url_vs_plain` | 50 | 0.00 | 1.00 | 0.04 | 0.06 |

#### Error categories

`{"false_normalization": 62, "wrong_rewrite": 1190}`

## Latency methodology

Latency is measured with a monotonic high-resolution clock around each individual
record. Initialization and 15 warm-up calls are reported separately. The
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

```bash
RAYON_NUM_THREADS=8 .venv-benchmark/bin/python benchmarks/run_comparison.py
```

The command writes a new timestamped directory under
`eval/voice_agent_itn/results/`. Use `--output` to select an explicit directory.
The Thutmose environment and artifact can be overridden with
`--thutmose-python` and `--thutmose-artifact`.

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
