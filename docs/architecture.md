# Architecture

`premove-itn` is a context-aware dispatcher for deterministic inverse text
normalization. It learns which source span should be normalized and which
structural class applies. It does not learn how to write the normalized value.

## System contract

The system accepts a complete current transcript and returns normalized text
plus every successful edit's class, source span, source text, replacement, and
model score.

The primary interface is a normalizer. Model, tokenizer, decoder, and parser
controls remain behind that interface.

```text
source text
    |
    v
source tokens with original character offsets
    |
    v
contextual BIO tagger
    |
    v
decoded tagged spans
    |
    v
optional confidence gate
    |
    v
forced class-specific Rust realizer
    |
    v
successful edits applied right-to-left
    |
    v
normalized text plus provenance
```

## Responsibility split

### Learned module

The contextual tagger answers only:

1. Which source-text span is normalizable?
2. Which structural class describes that span?

It emits one BIO label and score per source word. It never produces replacement
text, invokes a deterministic parser, or rewrites the transcript.

### Deterministic module

The Rust realizer receives an already selected class and exact source span. It
routes directly to the matching class-specific parser in `text-processing-rs`.
It does not call the upstream fixed-priority sentence dispatcher for hybrid
realization.

The proof reuses upstream parsers for `CARDINAL`, `TIME`, `DATE`, `MONEY`,
`DECIMAL`, and `PHONE`. `DIGIT_SEQUENCE` uses a small local deterministic parser
because leading zeros, one-by-one digits, and `double` or `triple` repetitions
are not cardinal arithmetic.

The forced interface also routes `ELECTRONIC`, `MEASUREMENT`, `ORDINAL`,
`PUNCTUATION`, `WHITELIST`, and `WORD` to their upstream English ITN parsers.
These deterministic routes do not add the kinds to the contextual MVP label
vocabulary.

The upstream sentence dispatcher remains available only as the baseline that
the hybrid must beat on contextual cases.

### Rewrite module

The rewriter applies only successful realizations. It applies replacements from
right to left so all edits retain their original source coordinates. Returned
edits remain ordered by source position.

## MVP label vocabulary

The proof has seven structural span kinds:

```text
DIGIT_SEQUENCE
CARDINAL
TIME
DATE
MONEY
DECIMAL
PHONE
```

`O` means that a word is outside a normalized span. BIO encoding creates 15
labels: `O` plus `B-` and `I-` forms for each kind.

These are representation classes, not business meanings. `ORDER_ID`,
`BOOKING_ID`, and `ACCOUNT_ID` remain contextual evidence for downstream
business binding.

Deferred classes are `PERCENTAGE`, `ORDINAL`, `FRACTION`, `MEASUREMENT`,
`ELECTRONIC`, and `ALPHANUMERIC_SEQUENCE`. They require benchmark evidence.

## Safety invariants

Every implementation must preserve these invariants:

1. The learned module never constructs normalized text.
2. A prediction alone cannot cause an edit. The selected deterministic realizer
   must accept the complete span.
3. A rejected or unsupported realization leaves its source text unchanged.
4. Text outside successful edits remains byte-for-byte identical to the input.
5. Public spans use `[start, end)` character offsets into the original input.
6. Successful edits cannot overlap.
7. Uncertain input can be preserved instead of guessed.
8. The benchmark is frozen before synthetic training data is created.
9. Training and validation are split by template family.

Without the first four invariants, a class error can become a destructive value
rewrite. Without original offsets, downstream provenance cannot identify the
evidence that produced a normalized value.

## Coordinates and tokenization

The pipeline uses three coordinate systems:

1. Original-text character offsets used by public results.
2. Source word tokens used by the classifier contract.
3. Model subword tokens used internally by the encoder.

Source tokenization records the exact range for every word. The model tokenizer
receives those words with `is_split_into_words=True`. Its `word_ids()` mapping
associates model subwords with source words. Model subword offsets never become
public provenance.

## BIO decoding

The first proof uses deterministic repair, not a CRF or Viterbi decoder.

An `I-X` label starts a new `B-X` span when it follows `O` or a `B-`/`I-` label
of a different kind. A span's initial score is the minimum score of its member
words, which prevents averaging away one weak decision.

Raw argmax predictions are evaluated before a confidence threshold is added.
This exposes actual classifier accuracy rather than hiding errors through
abstention.

## Confidence and abstention

Confidence gating is optional during the proof. When enabled, a span below the
configured threshold is preserved unchanged.

Softmax scores are not treated as calibrated probabilities. Temperature scaling
and final threshold tuning are deferred until raw classifier and complete hybrid
accuracy pass the POC gate. Production policy prioritizes edit precision over
maximum coverage.

## Python and Rust seam

Python owns:

- source tokenization and provenance;
- model training and inference;
- BIO decoding and confidence policy;
- edit orchestration and result types;
- datasets, evaluation, and benchmarks.

Rust owns:

- forced class-specific realization;
- the local `DIGIT_SEQUENCE` parser;
- the upstream sentence baseline;
- written-to-spoken normalization for synthetic-data tooling.

PyO3 exposes Rust in process as `premove_itn._rust`. There is no subprocess,
JSON-lines protocol, worker queue, or duplicate normalization service.

## Failure behavior

The system fails closed:

```text
unknown structural class
    -> explicit interface error

known class but parser rejects the span
    -> no edit

prediction below an enabled threshold
    -> no edit

malformed or overlapping spans
    -> reject the affected edit or result
```

Ordinary text and unsupported cases remain unchanged.

## Streaming model

Streaming stays outside the normalizer. Each revised ASR hypothesis is a new
complete input:

```text
transcript snapshot v17 -> normalize -> derived result v17
transcript snapshot v18 -> normalize -> derived result v18
```

The proof recomputes the bidirectional encoder over each snapshot. It does not
cache attention state or make pauses close a value span. Prefix behavior is
measured with a dedicated reviewed replay benchmark.

Any later Premove integration must use latest-only scheduling instead of a FIFO
backlog of obsolete transcript revisions.

## Evaluation boundary

The proof compares three systems on the same reviewed inputs:

1. The current `text-processing-rs` sentence dispatcher.
2. Classifier-only exact span and class prediction.
3. The complete `premove-itn` hybrid.

The primary metric is exact target replacement. Whole-sentence exact match,
hard-context accuracy, per-class accuracy, destructive edits, and unchanged
coverage are reported separately.

The project continues only if the hybrid substantially beats the deterministic
baseline on the hard contextual set. Strong isolated grammar performance alone
does not prove the hypothesis.

## Deferred architecture

The proof does not include:

- ONNX export or INT8 quantization;
- calibrated confidence thresholds;
- an all-Rust model runtime;
- true incremental Transformer state;
- Premove pipeline integration;
- changes to Typed, MiniLM, cross-encoders, GLiNER, or binding;
- WASM, Swift, Core ML, or release packaging;
- multilingual normalization;
- a public synthetic-data command;
- additional structural classes.

These items require a separate decision after the accuracy and latency gates
pass.
