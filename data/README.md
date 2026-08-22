# Data

The data directories separate human-reviewed evaluation evidence from future
synthetic training data.

```text
data/
├── golden.jsonl       reviewed real-world and representative cases
├── hard.jsonl         reviewed contextual contrast cases
├── prefixes.jsonl     reviewed transcript-revision sequences
├── templates/         future synthetic template families
└── generated/         ignored generated corpora
```

The reviewed files and template directories are planned. Add them only in their
approved roadmap phase.

## Reviewed record format

`golden.jsonl` and `hard.jsonl` use one JSON object per line:

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

Generated records contain source tokens, BIO labels, exact offsets, class,
template family, split, and seed. The shared dataset validator will define the
canonical schema before generation is implemented.

Training and validation are split by template family. A family cannot occur in
both splits.

## Separation rules

- Never generate `golden.jsonl`, `hard.jsonl`, or `prefixes.jsonl`.
- Never copy reviewed wording into template files.
- Never train on reviewed benchmark records.
- Never commit `data/generated/`.
- Never commit private, identifying, or customer transcript data.
- Record provenance and license information for imported public data.
- Do not use an LLM to assign ground-truth labels.
