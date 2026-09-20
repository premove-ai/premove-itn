---
title: Premove ITN architecture
description: Follow Premove ITN from Rust candidate generation through token alignment, contextual scoring, and exact decoding.
---

Premove ITN divides normalization into three learned/selection decisions and
one deterministic contextual enrichment step:

1. **Generate:** Rust decides which written forms are valid for each source span.
2. **Score:** a trained DeBERTa-based model uses the full sentence to score
   each proposed edit.
3. **Decode:** an exact interval algorithm selects a compatible set of edits
   and renders the output.

After decoding, the v0.3.0 runtime performs a fourth deterministic step:

4. **Enrich:** caller-supplied context resolves compatible temporal spans and
   produces the resolved text view.

This separation limits the learned model to a bounded choice. It cannot invent
a value that the deterministic layer did not generate. It also lets us locate
failures: missing candidates belong to the realizer layer, wrong rankings to
the scorer, and incompatible selections or rendering to the decoder.

The CLI calls the same `PremoveITN.normalize()` implementation exposed by the
Python API. The structured Python methods reuse that operation and retain its
result views:

```text
ASR transcript
  → token spans → batched Rust realizers → candidate edits
  → encoder-token alignment + replacement tokenization
  → one contextual DeBERTa pass + candidate feature head
  → scalar candidate scores → exact interval decoding
  → validated replacement rendering → normalized transcript
  → optional contextual temporal enrichment → structured result views
```

## 1. Generate the candidate graph

Python finds word and punctuation tokens with `\w+|[^\w\s]` and enumerates
every non-empty contiguous token span. For `T` tokens, this is
`T × (T + 1) / 2` possible spans before realization. A span's `text` is an
exact slice of the original string from the first token's start to the last
token's end, so it retains internal spaces or tabs. Its token indices and
character offsets are both half-open.

Repeated span strings are sent to Rust only once per transcript. One
`realize_candidate_batch()` call evaluates the 13 `SpanKind` realizers for each
unique string; Rust parallelizes independent strings. It removes exact no-op
outputs, groups identical replacements across kinds into one kind bitmask,
and returns replacements in deterministic order. Python expands each result
back to its original source occurrences and sorts candidates by token span
and replacement. There is no learned proposal step or top-k pruning here.

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

The same source span can have competing replacements. For example,
`one oh five` produces `105` and `01:05`; `105` can carry several kind labels
but remains one candidate. The candidate's character interval is the
selection topology, while its `replacement` is the exact string later written
by the renderer.

Rust owns the realization rules. `SPAN_KINDS` in Python supplies the ordered
kind vocabulary used by batching and scoring; Python does not implement a
second realizer. The
[Rust candidate coverage](/itn/docs/internals/rust-candidate-coverage) page
explains each kind and the complete-span guards.

## 2. Align candidates and score them in context

The pipeline uses three coordinate systems for different purposes:

| Coordinates | Used for |
| --- | --- |
| Source token indices | Enumerating candidate spans. |
| Source character offsets | Gold alignment, non-overlap constraints, and final rendering. |
| DeBERTa token indices | Pooling contextual embeddings for each candidate. |

The pinned fast tokenizer encodes the **whole source sentence** with special
tokens, an attention mask, and character offsets. `align_candidate_tokens()`
maps each candidate's source characters to the overlapping half-open encoder
token span and rejects an interval not fully covered by encoder tokens. The
sentence is encoded without truncation; a result above 512 encoder tokens
raises an error rather than silently losing an edit.

Candidate replacements are tokenized separately without special tokens. The
encoded sentence, aligned source spans, and replacement token IDs form one
`EncodedCandidates` value. `collate_candidate_batch()` pads sentence and
replacement IDs, masks padding, and flattens candidates in stable sentence
order. `candidate_offsets` records each sentence's slice of the flattened
score vector. The collator rejects empty replacement tokenizations and two
different written forms that become indistinguishable through identical
encoder-token span, kind features, and replacement token IDs. Without that guard,
the scorer would be forced to assign the same features to different outputs.

### Candidate features

The DeBERTa encoder runs once for the padded sentence batch. For each
candidate, the scorer concatenates three feature groups:

```text
source:       [first; last; mean] of contextual encoder states
kinds:        linear projection of the 13-way multi-hot kind vector to 32 values
replacement:  [first; last; mean] of replacement input-token embeddings
```

The source features carry information from the entire sentence because their
token states come from the contextual encoder. Prefix sums compute each span
mean without repeatedly summing every token in every candidate. The kind
vector preserves all realizer derivations for a merged replacement.

Replacement features use the encoder's **input embedding table**, not a
second contextual encoder pass. First and last embeddings retain order that a
mean alone would lose. They let the head distinguish outputs such as `95` and
`788` for the same span and kind labels. Replacement padding is masked before
mean pooling.

With encoder hidden size `H`, the concatenated feature width is `6H + 32`.
A 256-unit GELU head, dropout `0.1` during training, and a final linear layer
produce one scalar per candidate. These are path weights, not calibrated
per-candidate probabilities. In inference the loaded model is in evaluation
mode, so dropout is disabled.

## 3. Select and render a complete path

The scorer can give two overlapping edits high scores. Applying both would
corrupt the source span, so the decoder chooses a **complete source path**
instead of accepting independent candidate thresholds. Each candidate is a
weighted edge from `char_start` to `char_end`. A zero-score `KEEP` edge advances
one source character and makes every source position reachable.

`max_path_indices()` computes the best cumulative score at each character
boundary and stores predecessor pointers. Backtracking yields the selected
non-overlapping candidate indices in source order. This is exact max-sum
dynamic programming over the generated intervals, not beam search. A
negative-score edit cannot improve a path because zero-score `KEEP` edges can
cover that same interval.

The renderer then checks every selected interval against the original source,
copies gaps exactly, and writes replacements in order. It rejects overlapping,
out-of-range, or stale source spans. A narrow spacing rule can remove one
source separator before an attaching symbol, as in
`five percent` → `5%`. It does not rewrite unrelated whitespace.

For the source graph, dynamic programming takes `O(n + C)` time for `n` source
characters and `C` candidates after scoring. Candidate enumeration starts
with `O(T²)` token spans and model cost grows with sequence length and
candidate count. The exact decoder is therefore not a shortcut around
candidate coverage or model inference cost.

Training uses the same source intervals but marginalizes all paths that can
reconstruct the expected sentence. The
[structured prediction](/itn/docs/internals/structured-prediction) page derives
the gold graph, loss, and max-path algorithm.

## Runtime and model boundary

`PremoveITN.from_pretrained()` resolves a local artifact or the pinned Hugging
Face commit. It verifies release provenance and the model-file SHA-256, then
loads the exact scorer state and fast tokenizer in evaluation mode. It keeps
both resident for reuse; the default release cannot silently drift to a new
Hub commit. See [Inference artifact release](/itn/docs/internals/inference-artifact)
for the artifact contract.

`normalize()` returns empty or whitespace-only strings unchanged and returns
the source unchanged if Rust generates no edits. Otherwise it encodes, scores,
decodes, and returns the readable `text` view. The v0.3.0 structured APIs keep
the decoder's exact span offsets and add a deterministic `resolved_text` view.
`NormalizationContext` is caller supplied. A contextual resolver may annotate
unchanged temporal text when Rust generates no candidate, but it never builds
another candidate graph, calls the model again, or changes an authoritative
decoder edit. The CLI constructs the same `PremoveITN` object and reuses it
for every input line:

```text
Python API ─┐
            ├─→ PremoveITN.normalize() → candidates → scorer → decoder → result
CLI ────────┘
```

## 4. Enrich temporal spans

The temporal layer consumes the single decoded `NormalizationResult`. It can
enrich a selected `DATE` span or add a compatible annotation for unchanged
expressions such as `tomorrow`. It uses fixed caller facts such as
`reference_datetime`, `timezone`, `locale`, and `date_order`; it does not read
the host clock or infer application state.

The resolver supports a deliberately narrow grammar: relative dates and
bounded day/week offsets, calendar-week weekdays, named and
weekday-qualified dates, and numeric dates. Canonical resolved calendar values
are ISO `YYYY-MM-DD`. Missing context, invalid dates, ambiguous numeric forms,
and contradictory weekdays remain unresolved. Existing `resolved_value`
metadata is authoritative.

All three public methods share one internal operation:

```text
normalize()            → result.text
normalize_resolved()   → result.resolved_text
normalize_structured() → result
```

No method performs a second candidate build, model call, decode, or alignment
pass.

## Implementation map

| Component | Source |
| --- | --- |
| Public runtime | [`src/premove_itn/contextual.py`](https://github.com/premove-ai/premove-itn/blob/main/src/premove_itn/contextual.py) |
| Candidate graph | [`src/premove_itn/candidates.py`](https://github.com/premove-ai/premove-itn/blob/main/src/premove_itn/candidates.py) |
| Model inputs | [`src/premove_itn/model_inputs.py`](https://github.com/premove-ai/premove-itn/blob/main/src/premove_itn/model_inputs.py) |
| Candidate scorer | [`src/premove_itn/candidate_scorer.py`](https://github.com/premove-ai/premove-itn/blob/main/src/premove_itn/candidate_scorer.py) |
| Exact path algorithm | [`src/premove_itn/structured_loss.py`](https://github.com/premove-ai/premove-itn/blob/main/src/premove_itn/structured_loss.py) |
| Training batch and loss | [`src/premove_itn/training_batch.py`](https://github.com/premove-ai/premove-itn/blob/main/src/premove_itn/training_batch.py) |
| Final rendering | [`src/premove_itn/decoder.py`](https://github.com/premove-ai/premove-itn/blob/main/src/premove_itn/decoder.py) |
| Artifact verification | [`src/premove_itn/inference_artifact.py`](https://github.com/premove-ai/premove-itn/blob/main/src/premove_itn/inference_artifact.py) |
| Context and temporal enrichment | [`src/premove_itn/context.py`](https://github.com/premove-ai/premove-itn/blob/main/src/premove_itn/context.py), [`src/premove_itn/temporal.py`](https://github.com/premove-ai/premove-itn/blob/main/src/premove_itn/temporal.py) |
| Structured result views | [`src/premove_itn/results.py`](https://github.com/premove-ai/premove-itn/blob/main/src/premove_itn/results.py) |
| Rust realizers | [`rust/src/lib.rs`](https://github.com/premove-ai/premove-itn/blob/main/rust/src/lib.rs) |

The focused tests cover
[candidate enumeration](https://github.com/premove-ai/premove-itn/blob/main/tests/test_candidates.py),
[scorer features](https://github.com/premove-ai/premove-itn/blob/main/tests/test_candidate_scorer.py),
[exact gold alignment](https://github.com/premove-ai/premove-itn/blob/main/tests/test_gold_graph.py),
[path partitions](https://github.com/premove-ai/premove-itn/blob/main/tests/test_structured_loss.py),
and [final rendering](https://github.com/premove-ai/premove-itn/blob/main/tests/test_decoder.py).
