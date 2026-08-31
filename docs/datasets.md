# Dataset registry

This page is the canonical index of datasets used or prepared by Premove ITN.
Source data and generated artifacts stay outside Git. Revision-pinned source
notes, generated manifests, and SHA-256 hashes provide provenance.

## Current experiment datasets

| Dataset | Current role | Training use | Evaluation use | License or source terms |
| --- | --- | --- | --- | --- |
| Google Text Normalization Dataset 1 | Primary supervised corpus and in-distribution benchmark | Train partition only | Frozen validation, permanent 10k holdout, and later untouched test | Kaggle competition source terms; do not redistribute source artifacts |
| Golden | Curated regression and development benchmark | Never | Every major checkpoint | Repository evaluation artifact |
| NVIDIA Numb3rs | External semiotic-class stress benchmark | Never | Complete 10,131-row test split | CC BY-NC-SA 4.0 |

The full Google experiment uses 711,135 eligible train rows with at most 300
source characters. The 10k checkpoint contains train record IDs 0–9,999.
`scripts/train_full_google.py` continues from it on 701,135 unseen train rows.
It excludes all 39,543 validation rows and all 39,616 test rows. The compiled
Dataset 1 SHA-256 is
`1da9671c18e4916f3700244d9c100b25b3e8e0113b1336ba338b2f11b63541c0`.
The runner uses bounded, ordered process prefetch. A resumed run starts source
preparation at the durable batch cursor, so it does not rebuild or retrain
checkpointed batches. Batches after the last checkpoint are repeated because
their updates are absent from the restored model and optimizer state.
The batch cursor is checkpointed every 1,000 completed batches, including
batches with no candidates. The newest and previous valid checkpoint
generations are retained. A run fingerprint binds a partial checkpoint to the
dataset hash, model revision, selection rules, batch size, and bucket window.
`progress.json` records exact live counters and a measured ETA every 25 batches.
The supervised launcher restarts an exited trainer or one whose progress is
stale for 15 minutes.

Golden remains evaluation-only. Its SHA-256 is
`271ab423ebcb0fc041b3dc0fbb144ef021b8078553a4b9ef69b2e4f9a4a0466a`.
It has been used during development, so it is a development benchmark rather
than a pristine final test set.

Numb3rs is pinned at revision
`192908075e1bd293914cc7b508f4a183ba6ef2b8`. Text evaluation uses `text` as
input and `original_text` as the exact reference. The downloaded metadata
SHA-256 is
`d6c6020fc8d3ccf8404395814307dbf0aca415305ed6964543e63fb3aa5df377`.
It is derived from Google TN, has one test split, and is not a fully independent
linguistic distribution. See
[`research/nvidia-numb3rs-source.md`](research/nvidia-numb3rs-source.md).

## Prepared conversational sources

These sources are compiled for later experiments. They are not part of the
current full-Google training run. Their source text does not provide native
Google-TN-style spoken/written pairs. Compilers therefore create conservative
targets only from annotated spans and accept only complete targets reachable
through `GoldGraph`.

| Dataset | Pinned source | License | Compiled rows | Train / validation / test | Source note |
| --- | --- | --- | ---: | ---: | --- |
| Schema-Guided Dialogue | `e852981ae34990f4358979625854259302feaa78` | CC BY-SA 4.0 | 113,784 | 102,570 / 5,555 / 5,659 | [`research/schema-guided-dialogue-source.md`](research/schema-guided-dialogue-source.md) |
| SLURP real text | `8eb16545762be97ace75334109d73824217311f1` | CC BY 4.0 | 16,431 | 14,769 / 814 / 848 | [`research/slurp-dataset.md`](research/slurp-dataset.md) |
| SpokenWOZ | train/dev `d3aad10f2e5a37e7e1e84375f0db368b5872a044`; test `d6c2d9e53b1005e327db582d327b69311092eb65` | CC BY-NC 4.0 | 74,012 | 66,642 / 3,644 / 3,726 | [`research/spokenwoz-source-readiness.md`](research/spokenwoz-source-readiness.md) |
| Taskmaster-1 spoken Wizard-of-Oz turns | `d92cb6af3005f1dc09c39e75e7daf4a04905e00b` | CC BY 4.0 | 2,422 | 2,192 / 112 / 118 | [`research/taskmaster-1-source.md`](research/taskmaster-1-source.md) |

SpokenWOZ is noncommercial. Do not combine it into a general-use training
artifact without explicitly selecting the noncommercial research mode. SGD is
ShareAlike. Combined artifacts must preserve every effective source
restriction. `scripts/build_training_pool.py` validates licenses, provenance,
deduplication, leakage-safe connected-component splits, model token limits,
and `GoldGraph` reachability.

## Evaluation policy

- Never train on Google validation or test, Golden, or Numb3rs.
- Keep Google test untouched until architecture and hyperparameters are frozen.
- Report deterministic Rust from `text-processing-rs` in every comparison.
- Report exact sentence accuracy, positive, KEEP, MULTI, per-kind results,
  oracle coverage, and error categories when the benchmark supports them.
- Preserve raw metric files by benchmark and checkpoint under
  `data/generated/`. Do not replace earlier results.
- Record the model revision, checkpoint identity, dataset revision or hash,
  selection policy, and per-kind training exposure for every experiment.

See [`evaluations/checkpoint-comparison.md`](evaluations/checkpoint-comparison.md)
for recorded results.

## Artifact locations

- Downloaded sources: `data/external/`
- Compiled corpora: `data/generated/<dataset>/`
- Leakage-safe training-ready corpora: `data/generated/training_ready/<dataset>/`
- Checkpoints and raw metrics: `data/generated/<experiment>/`
- Google Dataset 1: `datasets/google_tn/dataset_1/dataset.jsonl`

The ignored source and generated directories are not package inputs. Do not
commit downloaded corpora, audio, generated datasets, checkpoints, or raw
metrics.
