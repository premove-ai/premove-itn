# VoiceCodeBench and sealed-evaluation audit

Research date: 2026-09-06

## Decision

Add VoiceCodeBench as the next external component evaluation. Its `acoustic`
and `canonical` transcript layers form a direct text-only ITN pair. Pin the
dataset to revision
`3ccea73877a159eb2a8b17304148c325c5fe5061`.

Also add SLUE-VoxPopuli when Hugging Face access is available. Do not add
Taskmaster-2 or Audio2Tool to the text-only ITN scorecard without a separate,
documented transformation. They do not supply direct spoken-text to written-text
ITN targets in the form that Premove needs.

## VoiceCodeBench

### Provenance and license

- Dataset: `besimple-ai/voice-code-bench`
- Pinned revision: `3ccea73877a159eb2a8b17304148c325c5fe5061`
- Split: one `test` split with 300 rows
- License: MIT
- Size: 300 human-recorded English segments, 85 anonymized speakers, 5.587
  hours, and 1,482 audited entities across 26 types

The repository card defines VoiceCodeBench as a test-only benchmark and gives
these counts. The repository's license file contains the MIT grant. Sources:
[dataset card](https://huggingface.co/datasets/besimple-ai/voice-code-bench/blob/3ccea73877a159eb2a8b17304148c325c5fe5061/README.md),
[license](https://huggingface.co/datasets/besimple-ai/voice-code-bench/blob/3ccea73877a159eb2a8b17304148c325c5fe5061/LICENSE).

### Schema

Each metadata row contains:

- recording metadata: `file_name`, `audio_id`, `language`, `duration`
- grouping metadata: `domain`, `scenario`, `difficulty`
- speaker and audio-quality objects
- `transcripts.template`, `transcripts.acoustic`, and
  `transcripts.canonical`
- `entities`, where each entity has `id`, `type`, `role`, `acoustic`, and
  `canonical`
- `entity_types` and `entity_count`

The published metadata has exactly 300 rows. Every row contains three to eight
entities. The 26 entity types include phone numbers and extensions, email and
URLs, dates and times, money, measurements, product codes, reference IDs,
versions, file paths, IP addresses, ports, CLI flags, and spelled sequences.
Source: [pinned metadata](https://huggingface.co/datasets/besimple-ai/voice-code-bench/blob/3ccea73877a159eb2a8b17304148c325c5fe5061/data/metadata.jsonl).

### Premove evaluation contract

VoiceCodeBench supports two useful evaluations:

1. Whole-text ITN: normalize `transcripts.acoustic` and compare with
   `transcripts.canonical`.
2. Entity ITN: normalize each `entities[].acoustic` value and compare with its
   `entities[].canonical` value.

Report at least:

- exact whole-text accuracy
- exact entity accuracy, overall and by entity type
- candidate reachability, overall and by entity type
- accuracy restricted to reachable entities
- all target entities correct per recording

The entity score is the clearer Premove diagnostic because unchanged prose and
punctuation can dominate a whole-text comparison. Keep unsupported entity types
in the reachability denominator, but publish a second result for the documented
Premove-supported subset.

The first implemented protocol is entity ITN. Whole-text evaluation is deferred
because the current exhaustive candidate graph is not bounded for 100–200-word
benchmark transcripts. The initial entity result is 711/1,482 exact-reachable.
Deterministic Rust scores 333/1,482, the 378k model scores 298/1,482, and the
378k + 20k model scores 378/1,482.

The official benchmark protocol sends raw audio to an ASR system and forbids
benchmark-specific post-ASR correction. A run from the supplied acoustic text
through Premove is therefore a **component evaluation**, not an official
VoiceCodeBench ASR result. This distinction must appear in generated reports.
The task and protocol are defined in the
[official dataset card](https://huggingface.co/datasets/besimple-ai/voice-code-bench/blob/3ccea73877a159eb2a8b17304148c325c5fe5061/README.md).

### Sealing rule

The benchmark is public and is not a hidden test. Treat the pinned 300 rows as a
project-sealed regression set:

- do not train on its acoustic or canonical text
- do not hand-author rules from individual failures
- run the Rust baseline and both retained checkpoints on the same revision
- preserve the first result as the pre-adaptation baseline
- after the first run, call it an external holdout or regression benchmark, not
  an unseen benchmark

## SLUE-VoxPopuli

### Provenance and license

- Dataset: `asapp/slue`, configuration `voxpopuli`
- Pinned revision: `67f7da031721a14cc391c7fa7c8d96411282d8a3`
- Splits: 5,000 train, 1,753 validation, and 1,842 test rows
- License: CC0 for the SLUE-VoxPopuli subset and its added NER annotations
- Current access state: Hugging Face marks the repository as gated with
  automatic approval; unauthenticated file downloads return an access error

The official card defines `normalized_text` as the normalized, spoken-form
transcription and `raw_text` as the raw transcription. It also documents the
split sizes and license. Source:
[pinned SLUE card](https://huggingface.co/datasets/asapp/slue/blob/67f7da031721a14cc391c7fa7c8d96411282d8a3/README.md).

### Premove evaluation contract

Use `normalized_text` as input and `raw_text` as expected output. This is a
direct text-only ITN test. Use validation for development. Keep test sealed until
the final checkpoint has been selected. Record the dataset revision and a hash
of the extracted TSV or Parquet file because access and repository contents can
change independently of the evaluator.

SLUE contains full utterances rather than audited ITN spans. Report full-text
exact accuracy and candidate reachability. A derived changed-span metric is
useful only if its extraction algorithm is fixed before test evaluation.

## Taskmaster-2

- Repository: `google-research-datasets/Taskmaster`
- Pinned revision: `d92cb6af3005f1dc09c39e75e7daf4a04905e00b`
- Size: 17,289 dialogs across restaurants, food ordering, movies, hotels,
  flights, music, and sports
- License: CC BY 4.0
- Split: the official release supplies seven domain JSON files, not fixed
  train/validation/test splits

An utterance contains `index`, `speaker`, `text`, and optional annotated
`segments`. A segment contains character offsets, its original text, and
semantic annotation names. User turns are transcriptions of spoken recordings;
assistant turns were written and rendered to users with TTS. Source:
[official TM-2 README](https://github.com/google-research-datasets/Taskmaster/blob/d92cb6af3005f1dc09c39e75e7daf4a04905e00b/TM-2-2020/README.md).

Taskmaster-2 does not provide a separate canonical written realization for each
spoken value. Semantic span labels do not determine punctuation, digit grouping,
currency notation, or date and time format. It can supply contextual sentences
for an audited GoldGraph corpus, but it is not a direct or sealed ITN validation
set.

## Audio2Tool

- Dataset: `RVtech/Audio2Tool`
- Pinned revision: `f1388da9a3189541ab82adac88824a0661670c43`
- License: CC BY-NC 4.0; commercial use requires separate permission
- Scope: 16,843 queries, 36,421 audio files, eight tiers, three assistant
  domains, and 152 tools
- Relevant tiers: tier 2 has 3,160 unique queries and 6,320 speaker-rendered
  rows; tier 6 has 2,146 unique queries and 4,292 speaker-rendered rows

Rows contain `query`, `expected_tool_call`, `extracted_params`, tool schemas,
audio paths, speaker metadata, and tier metadata. Tier 6 additionally contains
the original tool call and correction type. The counts, schema examples,
speaker-source details, and license are in the
[official pinned README](https://huggingface.co/datasets/RVtech/Audio2Tool/blob/f1388da9a3189541ab82adac88824a0661670c43/README.md).

Audio2Tool is not a direct text-only ITN set. Its target is a semantic tool call,
and its parameter values can require intent resolution, defaults, casing, and
schema-specific encoding. Premove alone cannot be fairly scored against
`expected_tool_call`. Use tiers 2 and 6 only in a later end-to-end pipeline where
the same tool-selection system is tested with and without Premove. Keep it out
of the Premove checkpoint-promotion metric.

Its noncommercial license also makes it unsuitable as launch training data
without separate permission. Evaluation and redistribution terms still need to
follow CC BY-NC 4.0.

## Other sets named in the launch plan

| Set | Add as text-only validation now? | Reason |
| --- | --- | --- |
| Existing Google test | Yes | It is already part of the project contract. Run the Rust baseline and both retained checkpoints. |
| Existing conversational test | Yes | It is already part of the project contract. Its data-lineage limits remain separate from evaluation quality. |
| Private Deepgram streaming set | Yes, after collection | It is the only planned test of interim-hypothesis stability. Freeze examples and metrics before the first scored run. |
| LibriTTS | No | It has paired text but adds broad audiobook coverage rather than the structured-value launch gaps. |
| LJSpeech | No | It is a single-speaker audiobook corpus and is lower value than SLUE for this launch. |
| SNuC | No for launch weights; optional research evaluation | Its identifier domain is relevant, but CC BY-NC 4.0 conflicts with commercial training and it requires a separate ingestion contract. |
| CSLU Numbers / Alphadigit | No for the two-day launch | They require LDC licensing and are not immediate public-package fixtures. |

## Final launch scorecard

The smallest defensible scorecard is:

1. Golden, Numb3rs semantic, PolyNorm, Google validation, and conversational
   validation as development regressions.
2. VoiceCodeBench as a pinned external structured-value holdout.
3. SLUE-VoxPopuli validation during development and its test split after model
   selection, once authenticated access is available.
4. Google test and conversational test after model selection.
5. A frozen private Deepgram streaming set for the actual product path.
6. Audio2Tool tiers 2 and 6 only as an end-to-end downstream test, outside the
   text-only ITN promotion score.

Run the Rust baseline on every text-only set. Run both retained checkpoints on
the identical records and evaluator version. Store per-record predictions so a
later Rust change can be separated into new reachability, lost reachability,
selection gains, and selection regressions.
