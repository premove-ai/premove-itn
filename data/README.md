# Data

The data directories separate human-reviewed evaluation evidence from local
raw sources and future derived training data.

```text
data/
├── golden.json        frozen reviewed evaluation set
├── golden.sha256      integrity checksum for the frozen set
├── golden_manifest.json  contract, coverage, and leakage evidence
├── deferred.json      reviewed cases for classes outside Dataset V1
├── deferred.sha256    integrity checksum for deferred reviewed cases
├── hard.json          future reviewed contextual contrast cases
├── prefixes.json      future reviewed transcript-revision sequences
├── templates/         future synthetic template families
├── external/          local downloaded source corpora (ignored)
└── generated/         derived training data (ignored)
```

`golden.json` is the frozen, human-reviewed Dataset V1 evaluation set. It contains
representative cases and adversarial contrastive pairs as a pretty-printed JSON
array. It must never be used for training. A `pair_id` groups two records whose
context changes the correct span decision or class. Exact input text occurs only
once, so no sentence receives accidental extra evaluation weight.

Dataset V1 evaluates the ten classes with positive training supervision:
`CARDINAL`, `DATE`, `DECIMAL`, `DIGIT_SEQUENCE`, `ELECTRONIC`, `MEASUREMENT`,
`MONEY`, `ORDINAL`, `PHONE`, and `TIME`. Telephone extensions are part of the
`PHONE` contract. Network addresses are not.

`deferred.json` preserves reviewed `PUNCTUATION`, `WHITELIST`, and `WORD`
records, together with both sides of every affected contrast pair. These cases
do not contribute to Dataset V1 model metrics because Dataset V1 has no positive
training supervision for those classes. They remain available for a future
dataset version that adds those classes.

`golden_manifest.json` binds Golden V1 to the exact Dataset V1 train and
validation artifacts used for its leakage audit. The `unseen_value` tag means
that at least one spoken span source in the record does not occur in the frozen
training split. It is evaluation metadata, not a model label.
Normalized-text accuracy alone is insufficient for contrastive pairs because
two different classes can produce the same written value.

The Dataset V1 split manifest field `donor_value_groups_split` has narrower
scope than global value disjointness. It counts only enrichment records linked
by generated donor identities or generated spoken-value grouping keys. Google
records are grouped by source sentence, and SGD records are grouped by dialogue.
A value can therefore occur in both train and validation through those sources.
This is expected for normal validation; Golden V1 separately records its
unseen-value slice.

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

`golden.json`, `deferred.json`, and future reviewed evaluation files use JSON
arrays containing
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

## Local raw sources

Downloaded source data is kept under `data/external/` and is ignored by Git.
See [`SOURCES.md`](SOURCES.md) for source URLs, licenses, local layout, and
integrity details.

The English Google Text Normalization source is stored as
`data/external/google_tn/en_with_types_merged.tsv`. It is a byte-for-byte
concatenation of the 100 English `output-*-of-00100` shards, in numeric order,
with the original tab-separated records and `<eos>` boundaries preserved.
No labels, spans, offsets, or replacements are edited in this raw copy.

The Schema-Guided Dialogue source is stored under `data/external/sgd/` in its
original train, dev, and test layout. It supplies natural dialogue context;
it is not itself an ITN record set.

The next derived dataset must be produced by deterministic parsing,
alignment, class mapping, and validation. Do not hand-edit the raw sources or
restore the removed synthetic `records.jsonl` artifact.

## Separation rules

- Never generate `golden.json`, `deferred.json`, `hard.json`, or `prefixes.json`.
- Never copy reviewed wording into template files.
- Never train on reviewed benchmark records.
- Keep downloaded external data and generated artifacts out of version
  control.
- Never commit private, identifying, or customer transcript data.
- Record provenance and license information for imported public data.
- An LLM may draft candidate synthetic examples, but its output is never
  ground truth. Accept a candidate only after deterministic schema and
  realizer validation plus a semantic audit. Reviewed evaluation labels remain
  human-owned.
