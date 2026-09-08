# Roadmap

The repository currently owns only deterministic realization.

## Current baseline

- Keep the `text-processing-rs` dependency pinned.
- Expose explicit kind realization and the upstream sentence-level functions.
- Maintain local Rust behavior only where there is a demonstrated gap.
- Keep focused tests for each local rule and supported kind.

## Next architecture gate

Build a candidate-oracle experiment before adding a model or dataset pipeline:

1. [x] Enumerate and deduplicate every valid kind and interval.
2. [x] Add implicit character-level `KEEP` edges.
3. [x] Search for a non-overlapping path that recreates the expected output.
4. [x] Measure exact target reachability: 136/136 Golden rows.

Do not add contextual training code until this experiment shows that the Rust
candidate lattice has a sufficient ceiling. If it does, design the encoder,
span scorer, and global decoder as new modules. Do not restore BIO labeling.

## Gold supervision

- [x] Build the source-target alignment state graph.
- [x] Retain all candidate transitions on at least one complete derivation.
- [x] Keep unchanged-text transitions implicit.
- [x] Recover a gold graph for all 136 Golden rows.
- [ ] Prepare the first clean contextual-training subset.
