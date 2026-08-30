# Checkpoint evaluation registry

This registry preserves the stable comparison results for each major model
checkpoint. Every comparison must show the deterministic Rust normalizer from
`text-processing-rs` as the baseline. Raw metrics remain under
`data/generated/`; they include per-kind counts, ablations, error categories,
samples, source hashes, model revision, and checkpoint metadata.

All model evaluations use DeBERTa revision
`64a8c8eab3e352a784c658aef62be1662607476f`. Evaluation data must not be used
for training.

## Recorded results

| Benchmark | System | Exact | Positive | KEEP | MULTI | Raw metrics |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Golden | Deterministic Rust (`text-processing-rs`) | 65.44% | 67.71% | 60.00% | 50.00% | Both Golden metric files |
| Golden | 1k model | 69.85% | 64.58% | 82.50% | 37.50% | `data/generated/first_run_1000/golden_metrics.json` |
| Golden | 10k model | 70.59% | 64.58% | 85.00% | 37.50% | `data/generated/google_validation_10k/golden_metrics.json` |
| Google validation | Deterministic Rust (`text-processing-rs`) | 61.19% | 56.83% | 87.87% | 41.53% | Both Google validation metric files |
| Google validation | 1k model | 80.81% | 78.04% | 97.71% | 70.63% | `data/generated/google_validation_1k_full/metrics.json` |
| Google validation | 10k model | 88.35% | 86.63% | 98.92% | 81.80% | `data/generated/google_validation_10k/metrics.json` |
| NVIDIA Numb3rs | Deterministic Rust (`text-processing-rs`) | 39.43% | n/a | n/a | n/a | `data/generated/numb3rs/10k/metrics.json` |
| NVIDIA Numb3rs | 1k model | 22.19% | n/a | n/a | n/a | `data/generated/numb3rs/1k/metrics.json` |
| NVIDIA Numb3rs | 10k model | 25.33% | n/a | n/a | n/a | `data/generated/numb3rs/10k/metrics.json` |

## Dataset identities

| Benchmark | Source identity |
| --- | --- |
| Golden | SHA-256 `271ab423ebcb0fc041b3dc0fbb144ef021b8078553a4b9ef69b2e4f9a4a0466a` |
| Google validation | SHA-256 `1da9671c18e4916f3700244d9c100b25b3e8e0113b1336ba338b2f11b63541c0` |
| NVIDIA Numb3rs | revision `192908075e1bd293914cc7b508f4a183ba6ef2b8`; metadata SHA-256 `d6c6020fc8d3ccf8404395814307dbf0aca415305ed6964543e63fb3aa5df377` |

Add each completed run to the table without replacing an earlier result.
Keep benchmark inputs, source revisions, and checkpoint identities unchanged
when comparing checkpoints.
