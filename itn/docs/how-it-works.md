---
title: How Premove ITN works
description: How deterministic candidates, contextual scoring, and exact decoding work together.
---

Premove ITN separates inverse text normalization into three steps:

```text
Generate → Score → Decode
```

## Generate

Deterministic Rust realizers enumerate structurally valid written forms for
spans in the transcript. The model cannot invent an output that the candidate
layer did not generate.

## Score

DeBERTa encodes the complete sentence once. It gives each candidate a score
based on the source span, proposed replacement, candidate kinds, and sentence
context.

```text
"the room code is one oh five"

             ┌── 01:05
one oh five ─┤
             └── 105       ← selected in the retained room-code case
```

Both replacements are generated for this span. The retained v0.1.0 prediction
for `the room code is one oh five` is `the room code is 105`. This example does
not imply that every ambiguity is resolved correctly; see the
[benchmark limitations](/itn/benchmarks).

## Decode

Candidates can overlap. Exact dynamic programming selects the highest-scoring
compatible set of edits across the complete sentence. Leaving source text
unchanged is always available through a zero-score `KEEP` path.

The rules decide what can be written. The model decides what fits the context.
The decoder decides which edits can coexist.

## What this design does not do

The scorer cannot generate arbitrary new strings. An unsupported format needs
a Rust candidate before the model can select it. The decoder can also keep the
source text when no candidate wins. This avoids forcing a rewrite for every
number-like phrase, but it means a supported phrase is not guaranteed to
change in every context.

Read the [implementation architecture](/itn/docs/internals/architecture) for tensor,
alignment, scoring, and decoding details. See [supported forms](/itn/docs/supported-forms)
for the current public boundary.
