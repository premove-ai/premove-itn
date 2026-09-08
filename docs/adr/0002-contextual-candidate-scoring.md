# ADR 0002: Add contextual candidate scoring above the deterministic runtime

## Status

Accepted

## Context

The deterministic runtime can enumerate Rust realizations and recover every
derivation of an expected sentence, but it cannot use sentence context to rank
competing candidates. The first model experiment must test whether contextual
candidate scoring learns this decision before work starts on more datasets or
production optimization.

## Decision

Add contextual candidate scoring as an optional layer above the deterministic
runtime. The runtime remains the single source of truth for candidates,
`SpanKind` values, complete-path legality, and expected-output compatibility.
Model code must consume those interfaces instead of reproducing realization,
`KEEP`, spacing, or gold-graph rules.

The first controlled experiment fine-tunes a revision-pinned
`microsoft/deberta-v3-large` encoder with a small candidate-scoring head. It
encodes each source sentence once, aligns candidate character spans to encoder
tokens, and assigns one scalar score to each complete `Candidate`. Training
marginalizes all expected-output-compatible paths against all legal complete
paths with dynamic programming. Inference uses exact maximum-score decoding
over the same path definition.

PyTorch, Transformers, and model-training dependencies remain optional. They
must not become dependencies of the deterministic `premove_itn` runtime API.
Dataset 1 train, validation, and internal-test assignments remain fixed for the
experiment.

## Consequences

- The first experiment measures the candidate-scoring formulation with an
  accuracy-first encoder; it does not select a production encoder.
- Encoder comparisons, distillation, quantization, ONNX export, and latency
  optimization remain out of scope until the formulation learns.
- Training and decoding must share one path implementation so their legality
  and spacing behavior cannot diverge.
- Model results apply to Dataset 1 coverage; absent kinds and external ITN
  behavior require later evaluation.
