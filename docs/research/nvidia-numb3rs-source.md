# NVIDIA Numb3rs source decision

## Decision

Use NVIDIA Numb3rs only as an external TN/ITN evaluation benchmark. Pin the
official Hugging Face dataset repository at revision
`192908075e1bd293914cc7b508f4a183ba6ef2b8`. Do not add any Numb3rs row to a
training partition.

The current release contains 10,131 English samples. This corrects the earlier
approximation of 11.6k. The dataset has one `test` split and no train or
validation split. Keep the complete set frozen and report it as one external
benchmark.

## Primary-source findings

- NVIDIA publishes the dataset at
  [`nvidia/Numb3rs`](https://huggingface.co/datasets/nvidia/Numb3rs/tree/192908075e1bd293914cc7b508f4a183ba6ef2b8).
  The pinned dataset card describes paired written and spoken forms with
  synthetic audio. It states that the pairs were derived from the Google Text
  Normalization dataset, that Magpie TTS generated the audio with six voices,
  and that human annotators retained only samples that passed review. See the
  [revision-pinned dataset card](https://huggingface.co/datasets/nvidia/Numb3rs/blob/192908075e1bd293914cc7b508f4a183ba6ef2b8/README.md).
- The same card reports 10,131 samples and 4.89 hours of audio across 12
  categories: `ADDRESS` 885, `CARDINAL` 780, `DATE` 977, `DECIMAL` 928,
  `DIGIT` 771, `FRACTION` 884, `MEASURE` 914, `MONEY` 775, `ORDINAL` 957,
  `PLAIN` 377, `TELEPHONE` 936, and `TIME` 947. It does not include an
  `ELECTRONIC` category.
- The card defines these metadata fields: `file_name`, `name`, `duration`,
  `category`, `original_text`, `text`, and `lang`. `original_text` is the
  written form and `text` is the spoken form. The repository provides
  [`metadata.jsonl`](https://huggingface.co/datasets/nvidia/Numb3rs/blob/192908075e1bd293914cc7b508f4a183ba6ef2b8/metadata.jsonl),
  a full NeMo-format
  [`manifest.jsonl`](https://huggingface.co/datasets/nvidia/Numb3rs/blob/192908075e1bd293914cc7b508f4a183ba6ef2b8/manifest.jsonl),
  and per-category manifests.
- The dataset configuration declares only a `test` split. Its files are the
  category-specific WAV directories. See the pinned
  [dataset-card configuration](https://huggingface.co/datasets/nvidia/Numb3rs/blob/192908075e1bd293914cc7b508f4a183ba6ef2b8/README.md#dataset-creation).
- The repository is licensed under
  [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en).
  Redistribution and adaptations are limited to noncommercial purposes.
  Shared copies require attribution, the license notice and source link, and
  an indication of modifications. Shared adaptations must use the same or a
  compatible license. Therefore, keep downloaded source data outside package
  and Git artifacts; record attribution and revision in generated manifests.

## Revision and integrity data

The Hugging Face repository API identifies
`192908075e1bd293914cc7b508f4a183ba6ef2b8` as the current source revision at
the time of this audit. The pinned repository tree reports these Git blob IDs:

- `metadata.jsonl`: `3bb4f81c03bf7bf3b4ce23e1de68bda061612ec7`
  (2,028,880 bytes)
- `manifest.jsonl`: `94b704da852e6a6b7168f2dcd84fdc5ce7b24439`
  (2,059,273 bytes)

These identifiers are repository object IDs, not published SHA-256 checksums.
The source does not publish a complete dataset checksum. A downloader must
compute and record local SHA-256 values for the selected metadata and audio
files after download.

## Evaluation policy

Use `text` as the ITN input and `original_text` as the exact written reference.
Do not use the synthetic audio for the text-only candidate-selector benchmark.
This avoids a roughly 1.29 GB audio download while preserving the paired ITN
evaluation data.

Map only direct category equivalents when reporting per-kind results:

| Numb3rs category | Premove kind |
| --- | --- |
| `CARDINAL` | `CARDINAL` |
| `DATE` | `DATE` |
| `DECIMAL` | `DECIMAL` |
| `DIGIT` | `DIGIT_SEQUENCE` |
| `MEASURE` | `MEASUREMENT` |
| `MONEY` | `MONEY` |
| `ORDINAL` | `ORDINAL` |
| `TELEPHONE` | `PHONE` |
| `TIME` | `TIME` |

Report `ADDRESS`, `FRACTION`, and `PLAIN` separately until the project defines
an explicit, tested mapping. Numb3rs contains isolated semiotic-class samples,
so it is not a meaningful `MULTI` benchmark. Because it is derived from Google
TN pairs, treat it as a human-verified number and semiotic-class stress test,
not as a fully independent linguistic distribution.
