# Premove ITN

Premove ITN turns spoken value expressions into deterministic written forms
without changing unrelated language.

## Language

**Dataset Record**:
A source sentence and expected sentence accepted for contextual-scoring work.
_Avoid_: Labelled sentence, BIO record

**Quarantined Sentence**:
A source sentence excluded from training with its provenance and one or more
explicit rejection reasons.
_Avoid_: Deleted sentence, bad row

**Annotation Mismatch**:
A Google TN written value that is not semantically equivalent to any current
Rust realization of its spoken value.
_Avoid_: Oracle failure, malformed sentence

**Oracle Miss**:
A structurally valid source and expected sentence for which no complete
candidate-and-`KEEP` derivation exists.
_Avoid_: Incorrect sentence, malformed sentence
