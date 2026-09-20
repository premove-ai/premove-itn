---
title: How Premove ITN works
description: How deterministic candidates, contextual scoring, and exact decoding work together.
---

Premove ITN separates inverse text normalization into three steps:

```text
Generate → Score → Decode
```

The v0.3.0 public API adds a deterministic fourth boundary after decoding:

```text
Generate → Score → Decode → Enrich temporal spans
```

The first three stages produce the readable normalized text. Enrichment uses
only explicit caller context and cannot add candidates, change a decoder edit,
or trigger another model pass.

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

Both replacements are generated for this span. The frozen v0.2.0 model
prediction for `the room code is one oh five` is `the room code is 105`. This does
not imply that every ambiguity is resolved correctly; see the
[benchmark limitations](/itn/benchmarks).

## Decode

Candidates can overlap. Exact dynamic programming selects the highest-scoring
compatible set of edits across the complete sentence. Leaving source text
unchanged is always available through a zero-score `KEEP` path.

The rules decide what can be written. The model decides what fits the context.
The decoder decides which edits can coexist. The temporal resolver then adds a
canonical value only when the selected span and caller context make it
deterministic.

## Enrich temporal values

`normalize_structured()` retains the decoder's exact source and normalized
offsets, then applies deterministic contextual rules to compatible `DATE`
spans and unchanged temporal text. For example:

```text
call me tomorrow
        ↓  reference_datetime=2026-09-20
call me 2026-09-21   (resolved_text)
```

The readable `text` remains `call me tomorrow`. The structured span carries
`resolved_value="2026-09-21"`. Without a reference datetime, the span remains
annotated but unresolved. `normalize()`, `normalize_resolved()`, and
`normalize_structured()` share the same internal result and model inference.

## What this design does not do

The scorer cannot generate arbitrary new strings. An unsupported format needs
a Rust candidate before the model can select it. The decoder can also keep the
source text when no candidate wins. This avoids forcing a rewrite for every
number-like phrase, but it means a supported phrase is not guaranteed to
change in every context.

Read the [implementation architecture](/itn/docs/internals/architecture) for tensor,
alignment, scoring, and decoding details. See [supported forms](/itn/docs/supported-forms)
for the current public boundary.
