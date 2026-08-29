# ADR 0001: Keep corpus preparation outside the runtime API

## Status

Accepted

## Context

Contextual scoring needs full-sentence `(text, expected_text)` records with
recoverable gold derivations. Google TN is much larger than the first training
pool and contains source classes that the current Rust realizers do not support.

## Decision

Google TN corpus preparation is an offline, provenance-preserving pipeline in
`scripts/`; the runtime package continues to own only deterministic realization,
candidate enumeration, and gold-graph recovery. Dataset artifacts contain
`text` and `expected_text`, never BIO labels or manually selected kinds, because
the current architecture derives all structured supervision from `GoldGraph`.

The compiler rejects a complete sentence when its CSV structure is invalid, a
source class is unsupported, a context annotation is invalid, Google and Rust
disagree on a semantic value, or the sentence fails the conservative English
quality checks. It records each rejection reason and bounded provenance
examples. Each selected record also passes a constructive candidate/realizer
derivation check; the audit command samples those records through `GoldGraph`.

The compiler assigns each unique full sentence to train, validation, or test by
a stable seeded SHA-256 hash. It then keeps at most 100,000 train sentences per
contained kind, with proportional validation and test caps. A multi-kind
sentence satisfies each applicable quota but appears once in the final dataset.

## Consequences

- Runtime realization behavior remains the single source of truth.
- Dataset builds are reproducible and do not depend on shard order.
- Train, validation, and test records cannot leak across partitions.
- Rare kinds keep every valid unique sentence instead of duplicating records.
- A manifest records source hashes, quality failures, quota results, and artifact
  hashes.
