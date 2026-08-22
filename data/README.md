# Data

The data directories separate human-reviewed evaluation evidence from future
synthetic training data.

```text
data/
├── golden.json        frozen reviewed evaluation set
├── golden.sha256      integrity checksum for the frozen set
├── hard.json          future reviewed contextual contrast cases
├── prefixes.json      future reviewed transcript-revision sequences
├── templates/         future synthetic template families
└── generated/
    └── records.jsonl  versioned synthetic corpus
```

`golden.json` is the frozen, human-reviewed evaluation set. It contains
representative cases and adversarial contrastive pairs as a pretty-printed JSON
array. It must never be used for training. A `pair_id` groups two records whose
context changes the correct span decision or class. Exact input text occurs only
once, so no sentence receives accidental extra evaluation weight.
Normalized-text accuracy alone is insufficient for contrastive pairs because
two different classes can produce the same written value.

The evaluation matrix covers representative positives, contextual
near-misses, class collisions, malformed input, long identifiers, mixed ASR
forms, boundary punctuation, capitalization, already-normalized input, and
adjacent spans without an outside token, same-class spans, or three-span
reconstruction.

Do not change `golden.json` after evaluating a model against it. Add newly
discovered cases to a future benchmark version instead. The remaining reviewed
files and template directories are planned. Add them only in their approved
roadmap phase.

## Reviewed record format

`golden.json` and future reviewed evaluation files use JSON arrays containing
records with this shape:

```json
{
  "id": "digit_001",
  "text": "my order number is four thirty",
  "expected_text": "my order number is 430",
  "spans": [
    {
      "start": 19,
      "end": 30,
      "source": "four thirty",
      "kind": "DIGIT_SEQUENCE",
      "replacement": "430"
    }
  ],
  "tags": ["identifier", "time_collision"]
}
```

Rules:

- `id` is unique across all reviewed data.
- `start` is inclusive and `end` is exclusive.
- `source` equals `text[start:end]` exactly.
- Spans are non-empty, ordered, and non-overlapping.
- `kind` is one of the approved MVP structural classes.
- `expected_text` equals the result of applying replacements right-to-left.
- `tags` describe evaluation slices, not model labels.
- Expected interpretations are reviewed by a human.

## Prefix record format

`prefixes.jsonl` stores all reviewed versions of one evolving transcript:

```json
{
  "id": "stream_digit_001",
  "versions": [
    {
      "text": "my order id is three",
      "expected_text": "my order id is 3",
      "spans": [
        {
          "start": 15,
          "end": 20,
          "source": "three",
          "kind": "DIGIT_SEQUENCE",
          "replacement": "3"
        }
      ]
    }
  ]
}
```

Repeated versions can represent pauses. Earlier words may change between
versions to represent ASR revisions. Each version is evaluated as a complete
current snapshot.

## Synthetic record format

Generated records contain an ID, source text, expected text, labeled spans,
source tokens, BIO labels, template family, split, seed, and provenance. The
versioned corpus uses one JSON object per line in `records.jsonl`.

The Batch 01 cleanup is reproducible with:

```bash
uv run python scripts/repair_dataset.py
```

This is a one-off repair utility for the checked-in Batch 01 artifact, not a
public synthetic-data generation command.

The cleanup regenerates all derived offsets, tokens, BIO labels, and expected
text. It also keeps canonical source-token sequences at or below 48 tokens so
the first 64-token model experiment has room for special tokens and subword
expansion.

Training and validation are split by template family. A family cannot occur in
both splits. The versioned `data/generated/records.jsonl` corpus contains
10,000 records: 9,000 training records and 1,000 validation records. Every
record includes its seed, split, template family, and provenance.

## Separation rules

- Never generate `golden.json`, `hard.json`, or `prefixes.json`.
- Never copy reviewed wording into template files.
- Never train on reviewed benchmark records.
- Keep generated artifacts out of version control unless they are an explicitly
  versioned corpus with recorded provenance and reproducible seeds, such as
  `data/generated/records.jsonl`.
- Never commit private, identifying, or customer transcript data.
- Record provenance and license information for imported public data.
- Do not use an LLM to assign ground-truth labels.
