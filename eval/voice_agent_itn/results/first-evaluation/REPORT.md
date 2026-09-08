# First Evaluation

Premove leads semantic entity accuracy overall and in the voice-agent group. It remains slower than both comparison backends. The release build substantially reduces the latency previously reported.

## Overall results

| Backend | Entity micro | Entity macro | Strict exact | Mean ms | p50 ms | p95 ms | p99 ms | Max ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| premove-itn | 89.70% | 90.71% | 40.53% | 56.49 | 55.73 | 71.40 | 89.69 | 127.64 |
| thutmose | 59.39% | 57.61% | 22.13% | 15.98 | 15.87 | 17.47 | 18.32 | 27.33 |
| text-processing-rs | 55.79% | 55.05% | 16.53% | 0.14 | 0.11 | 0.38 | 0.57 | 0.74 |

Micro accuracy pools the 1,640 declared entities. Macro accuracy gives each of the 18 entity categories equal weight. KEEP rows have no positive entities and are reported separately. Formatting-only differences count against strict exact, while the archived semantic scorer is unchanged.

## Voice-agent results

These tables use only the 400 `group=voice_agent` rows. Each of eight domains has 50 rows. The generic multi-entity rows do not enter domain claims.

| Backend | Correct entities | Accuracy | Mean ms | p95 ms | p99 ms |
|---|---:|---:|---:|---:|---:|
| premove-itn | 398/400 | 99.50% | 57.41 | 67.78 | 72.96 |
| thutmose | 268/400 | 67.00% | 16.04 | 17.48 | 18.20 |
| text-processing-rs | 273/400 | 68.25% | 0.15 | 0.31 | 0.36 |

| Domain | Premove | Thutmose | text-processing-rs |
|---|---:|---:|---:|
| banking | 50/50 (100%) | 43/50 (86%) | 26/50 (52%) |
| customer_support | 50/50 (100%) | 26/50 (52%) | 29/50 (58%) |
| ecommerce | 49/50 (98%) | 38/50 (76%) | 40/50 (80%) |
| healthcare_admin | 49/50 (98%) | 40/50 (80%) | 36/50 (72%) |
| logistics | 50/50 (100%) | 39/50 (78%) | 48/50 (96%) |
| scheduling | 50/50 (100%) | 26/50 (52%) | 25/50 (50%) |
| technical_support | 50/50 (100%) | 33/50 (66%) | 27/50 (54%) |
| travel | 50/50 (100%) | 23/50 (46%) | 42/50 (84%) |

### Latency by voice-agent domain

| Domain | Backend | Mean ms | p50 ms | p95 ms | p99 ms | Max ms |
|---|---|---:|---:|---:|---:|---:|
| banking | premove-itn | 57.29 | 56.53 | 66.15 | 74.69 | 79.76 |
| banking | thutmose | 15.96 | 15.93 | 17.77 | 18.07 | 18.20 |
| banking | text-processing-rs | 0.19 | 0.16 | 0.34 | 0.39 | 0.39 |
| customer_support | premove-itn | 57.91 | 57.47 | 66.03 | 71.57 | 72.05 |
| customer_support | thutmose | 16.05 | 15.95 | 17.36 | 18.69 | 19.63 |
| customer_support | text-processing-rs | 0.14 | 0.14 | 0.28 | 0.33 | 0.33 |
| ecommerce | premove-itn | 56.26 | 56.00 | 62.87 | 68.74 | 70.15 |
| ecommerce | thutmose | 15.98 | 15.81 | 17.39 | 17.52 | 17.56 |
| ecommerce | text-processing-rs | 0.16 | 0.16 | 0.21 | 0.22 | 0.22 |
| healthcare_admin | premove-itn | 56.42 | 55.41 | 65.84 | 70.38 | 71.91 |
| healthcare_admin | thutmose | 15.85 | 15.80 | 17.12 | 17.41 | 17.44 |
| healthcare_admin | text-processing-rs | 0.17 | 0.14 | 0.29 | 0.35 | 0.36 |
| logistics | premove-itn | 54.81 | 54.85 | 58.60 | 60.36 | 61.48 |
| logistics | thutmose | 16.30 | 16.43 | 17.55 | 18.16 | 18.20 |
| logistics | text-processing-rs | 0.13 | 0.13 | 0.17 | 0.26 | 0.32 |
| scheduling | premove-itn | 59.00 | 58.85 | 66.68 | 71.33 | 72.95 |
| scheduling | thutmose | 15.94 | 15.67 | 17.49 | 18.20 | 18.35 |
| scheduling | text-processing-rs | 0.16 | 0.15 | 0.32 | 0.36 | 0.36 |
| technical_support | premove-itn | 61.58 | 59.18 | 71.72 | 88.89 | 96.45 |
| technical_support | thutmose | 16.04 | 15.93 | 18.00 | 18.74 | 19.23 |
| technical_support | text-processing-rs | 0.12 | 0.12 | 0.19 | 0.21 | 0.21 |
| travel | premove-itn | 55.96 | 55.27 | 66.09 | 72.70 | 74.77 |
| travel | thutmose | 16.22 | 16.18 | 17.42 | 17.74 | 17.92 |
| travel | text-processing-rs | 0.13 | 0.13 | 0.18 | 0.18 | 0.19 |

## Accuracy by entity category

Each count below scores only the named entity, including entities in multi-entity sentences.

| Category | Premove | Thutmose | text-processing-rs |
|---|---:|---:|---:|
| CARDINAL | 118/145 (81.38%) | 86/145 (59.31%) | 33/145 (22.76%) |
| DATE | 135/135 (100.00%) | 77/135 (57.04%) | 127/135 (94.07%) |
| DECIMAL | 76/80 (95.00%) | 54/80 (67.50%) | 53/80 (66.25%) |
| DIGIT_SEQUENCE | 76/76 (100.00%) | 60/76 (78.95%) | 23/76 (30.26%) |
| EMAIL | 56/65 (86.15%) | 0/65 (0.00%) | 0/65 (0.00%) |
| FLIGHT_ID | 47/50 (94.00%) | 21/50 (42.00%) | 24/50 (48.00%) |
| IP | 54/54 (100.00%) | 28/54 (51.85%) | 7/54 (12.96%) |
| MEASUREMENT | 63/64 (98.44%) | 51/64 (79.69%) | 60/64 (93.75%) |
| MONEY | 68/95 (71.58%) | 44/95 (46.32%) | 11/95 (11.58%) |
| ORDER_ID | 55/61 (90.16%) | 39/61 (63.93%) | 34/61 (55.74%) |
| ORDINAL | 145/145 (100.00%) | 89/145 (61.38%) | 103/145 (71.03%) |
| PERCENT | 50/50 (100.00%) | 46/50 (92.00%) | 50/50 (100.00%) |
| PHONE | 105/106 (99.06%) | 69/106 (65.09%) | 49/106 (46.23%) |
| REFERENCE_ID | 124/151 (82.12%) | 90/151 (59.60%) | 79/151 (52.32%) |
| TIME | 136/165 (82.42%) | 149/165 (90.30%) | 133/165 (80.61%) |
| URL | 50/79 (63.29%) | 0/79 (0.00%) | 28/79 (35.44%) |
| VERSION | 49/55 (89.09%) | 43/55 (78.18%) | 47/55 (85.45%) |
| ZIP_CODE | 64/64 (100.00%) | 28/64 (43.75%) | 54/64 (84.38%) |

## Initialization and warm-up

| Backend | Initialization ms | 15 warm-up calls ms |
|---|---:|---:|
| premove-itn | 15067.53 | 1114.52 |
| thutmose | 2790.17 | 278.12 |
| text-processing-rs | 0.01 | 0.58 |

Warm-up records are saved separately for each backend. Thutmose also retains its startup warm-up in runtime metadata. Initialization timing is process/model initialization, not machine cold boot or first download.

## Release latency correction

| Backend | Original mean ms | Release mean ms | Reduction |
|---|---:|---:|---:|
| premove-itn | 240.93 | 56.49 | 76.55% |
| thutmose | 16.45 | 15.98 | 2.86% |
| text-processing-rs | 0.73 | 0.14 | 80.65% |

## Premove timing components

| Component | Mean ms | p95 ms | p99 ms |
|---|---:|---:|---:|
| candidate_graph_ms | 2.21 | 8.23 | 15.68 |
| encode_ms | 2.43 | 4.06 | 5.54 |
| model_ms | 51.35 | 59.61 | 73.60 |
| decode_ms | 0.49 | 0.62 | 1.01 |

## What the result supports

The project succeeds on the measured voice-agent value-normalization objective: 398/400 correct entities versus 268/400 for Thutmose and 273/400 for text-processing-rs. It also leads overall semantic accuracy. It does not win latency, and its remaining collision and KEEP errors prevent a claim of universal ITN superiority.

The original development artifact used the same frozen DeBERTa checkpoint. The expensive component was the debug Rust extension. Release compilation plus batching changes runtime cost. The correction is not evidence of improved model accuracy.

## Equivalence and limitations

- All 1,500 complete candidate tuples match the pre-batch builder exactly.
- All 1,500 Premove predictions match the original accuracy evidence byte for byte. All outputs from both other backends also match.
- The recovered scorer reproduces the archived scoring fields on all 4,500 records.
- This release pass took place after accuracy was observed. It is part of First Evaluation and is not a new blind evaluation.
- No training, model selection, candidate pruning, or benchmark-driven tuning occurred.
- The dataset is a balanced synthetic stress suite. Its scores do not estimate production traffic accuracy.
- Blind human gold adjudication remains pending. Exact and normalized overlap checks passed; token n-gram and embedding contamination checks remain incomplete.
- text-processing-rs is an upstream ablation, not an independent architecture. Thutmose uses a custom weight-compatible loader of the NVIDIA artifact, not the current NeMo API.
- Timing used an Apple Silicon Mac, sequential batch-one requests, MPS completion, and eight Rayon workers. No other benchmark or training workload ran concurrently; ordinary desktop background processes remained active.
- The native build profile was not captured in the original run metadata. Its debug classification follows the later build diagnosis.

## Detailed evidence

[Detailed model, kind, group, collision, and latency tables](DETAILS.md). Domain-label tables there include all groups; use the voice-only table above for domain claims. Their category tables summarize rows containing a category; use the entity-only counts above for category claims.

[Original accuracy evidence](../20260908T172500Z/REPORT.md), [run metadata](run.json), [artifact and graph audit](artifact.json), [prediction audit](prediction-equivalence.json), [reproduction instructions](../../../../benchmarks/README.md).
