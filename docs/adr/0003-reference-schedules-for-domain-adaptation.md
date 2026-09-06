# ADR 0003: Use reference schedules for domain adaptation

## Status

Accepted

## Context

The canonical conversational pool is 93.2% `KEEP`, while the useful adaptation
signal is concentrated in positive records and `KEEP` records for which Rust
offers a plausible edit. Training the complete pool or applying an additional
positive loss weight would obscure the exposure actually responsible for a
checkpoint's behavior.

## Decision

Domain-adaptation experiments use deterministic Training Schedules that refer
to immutable canonical records by dataset and record ID. A schedule controls
bucket proportions and order, never copies source text, changes partitions, or
includes validation or test records. Sampling controls class exposure; scheduled
examples receive the normal structured loss without an additional class weight.

The first conversational pilot starts from the selected 378k Google weights
with fresh AdamW state at `5e-6`. It schedules candidate-bearing conversational
`KEEP`, every conversational positive once, and kind-stratified Google-positive
replay in a `2:1:0.5` ratio. The three streams are deterministically interleaved.

## Consequences

- The canonical pool remains the sole source of record content and provenance.
- Schedule manifests and hashes make exposure and ordering independently
  auditable.
- Source or dataset order cannot create a late-training distribution shift.
- Rare-kind coverage still requires new source data rather than repeated rows.
