# Architecture

Premove ITN separates three jobs that are often combined in inverse text
normalization:

1. Rust generates valid written forms.
2. DeBERTa uses sentence context to score those alternatives.
3. An exact decoder chooses the highest-scoring compatible set of edits.

The model does not generate arbitrary normalized text. It chooses between
deterministic candidates.

Both the Python API and CLI use the same `PremoveITN.normalize()` path:

```text
ASR transcript
      ↓
candidate generation
      ↓
Rust realizers
      ↓
one contextual DeBERTa pass
      ↓
candidate scores
      ↓
exact interval decoding
      ↓
normalized transcript
```

## Generate valid candidates

Python tokenizes the transcript with `\w+|[^\w\s]`, which preserves
punctuation as separate tokens. It considers every non-empty contiguous token
span as a possible normalization region.

Repeated span text is deduplicated, then all unique spans are sent to Rust in
one batched call. Rust evaluates the 13 supported realizer kinds and returns
every valid written form for each span. If multiple kinds produce the same
replacement for the same source span, Premove ITN stores one candidate with
multiple kind labels.

Each candidate contains:

```text
Candidate(
    token_start,
    token_end,
    char_start,
    char_end,
    text,
    replacement,
    kinds,
)
```

Token and character spans are half-open. Character offsets preserve exact
provenance back to the original transcript. Exact no-op replacements are not
included, and candidates are returned in stable span-and-replacement order.

Rust is the only source of realization rules. Python constructs the candidate
graph and scores it; it does not implement a second normalizer. See
[Rust candidate coverage](rust-candidate-coverage.md) for the complete grammar.

## Align candidates to model tokens

The pinned fast tokenizer encodes the complete transcript with special tokens,
an attention mask, and character offsets. Each candidate's character span is
mapped to the corresponding half-open DeBERTa token span. A candidate must be
fully covered by encoder tokens. Inputs longer than 512 encoder tokens are
rejected instead of being silently truncated.

Candidate replacements are tokenized separately without special tokens. The
encoded sentence, aligned source spans, and replacement token IDs form one
`EncodedCandidates` value. Batching pads sentences and replacements while
flattening candidate metadata in stable input order.

## Score candidates with sentence context

The complete transcript is encoded by DeBERTa once. For each candidate,
Premove ITN builds three feature groups:

```text
source context
    [first contextual token;
     last contextual token;
     mean contextual token]

replacement
    [first replacement token;
     last replacement token;
     mean replacement token]

candidate kinds
    projection of a 13-way multi-hot vector
```

The source features come from the contextual sentence encoding. Span means are
computed from prefix sums, so pooling does not loop over every token in every
candidate.

The replacement does not require a second DeBERTa forward pass. Its features
come directly from DeBERTa's input embedding table. First, last, and mean
pooling preserves replacement order and distinguishes alternatives with the
same source span and kind labels.

The combined feature vector has size `6 × hidden_size + 32`. It passes through
a 256-unit GELU MLP with dropout and produces one scalar candidate score. For
example, `seven eighty eight` can produce the same-kind alternatives `95` and
`788`; replacement features let the sentence context score them differently.

## Decode the best compatible edits

Candidate scores are not applied independently. Two high-scoring replacements
can overlap, and selecting one changes which other edits remain possible.
Premove ITN therefore solves the complete sentence as a character-interval graph.

Candidates are weighted edges over source character intervals. Leaving a
character unchanged is a zero-score `KEEP` edge. The decoder uses dynamic
programming to compute the maximum-score complete path, then follows
predecessor pointers backwards to recover the selected candidates.

This guarantees that:

- overlapping edits cannot both be selected;
- several non-overlapping edits can be selected in one sentence;
- a negative-scoring candidate loses to leaving the source unchanged.

The final renderer validates every selected source span, applies replacements
in source order, and preserves untouched text exactly. It also handles the
supported implicit spacing transition required for punctuation-like output,
such as `five percent` → `5%`.

The result is a global sentence-level decision, not a collection of independent
span classifications.

## Runtime and model boundary

`PremoveITN.from_pretrained()` resolves a local artifact or the pinned Hugging
Face snapshot, loads the model and tokenizer once, and keeps them resident for
reuse. Before inference, the loader verifies release metadata, architecture
version, ordered candidate-kind list, pinned DeBERTa revision, and model
SHA-256. See [Inference artifact release](inference-artifact.md) for the full
artifact contract.

The CLI constructs the same `PremoveITN` object as the Python API and reuses it
for every input line:

```text
Python API ─┐
            ├─→ PremoveITN.normalize() → candidates → scorer → decoder
CLI ────────┘
```

## Implementation map

| Component | Source |
| --- | --- |
| Public runtime | [`src/premove_itn/contextual.py`](../src/premove_itn/contextual.py) |
| Candidate graph | [`src/premove_itn/candidates.py`](../src/premove_itn/candidates.py) |
| Model inputs | [`src/premove_itn/model_inputs.py`](../src/premove_itn/model_inputs.py) |
| Candidate scorer | [`src/premove_itn/candidate_scorer.py`](../src/premove_itn/candidate_scorer.py) |
| Exact path algorithm | [`src/premove_itn/structured_loss.py`](../src/premove_itn/structured_loss.py) |
| Final rendering | [`src/premove_itn/decoder.py`](../src/premove_itn/decoder.py) |
| Rust realizers | [`rust/src/lib.rs`](../rust/src/lib.rs) |
