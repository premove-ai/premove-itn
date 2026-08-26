# Roadmap

Each phase has an exit condition. Do not start the next phase until the current
result is reviewed.

## Phase 1: Freeze architecture and package boundary

- Finalize the learned and deterministic responsibility split.
- Use the `premove-itn` distribution and `premove_itn` Python import.
- Configure a mixed Python/Rust package with Maturin and PyO3.
- Pin `text-processing-rs` v0.3.0.
- Expose forced class realization, sentence baseline, and TN through one
  in-process Rust extension.
- Implement only the missing strict `DIGIT_SEQUENCE` realizer locally.

Exit condition: Python can call every forced MVP realizer and the baseline in
process, with focused Rust and Python tests.

## Phase 2: Define public contracts

- Define all 13 configured span kinds and derive the 27 BIO labels from one
  source.
- Define immutable word, span, prediction, edit, and result types.
- Implement source tokenization with exact original-text offsets.
- Remove the premature broad schema after its replacement is tested.

Exit condition: labels, offsets, and public result invariants are stable.

## Phase 3: Freeze the judge

- Create 75–100 de-identified, human-reviewed golden cases.
- Create 40–60 hard contextual contrast cases.
- Create a separate reviewed streaming-prefix replay set.
- Include real failures, negative/O cases, multiple spans, long identifiers,
  leading zeros, and ID/time/date/phone collisions.
- Run and save the plain `text-processing-rs` sentence baseline.

Exit condition: every reviewed record passes structural validation, expected
interpretations are approved, and the baseline report is saved.

## Phase 4: Build synthetic training data

- Define named train and validation template families for every MVP class and
  negative/O examples.
- Keep every template family in exactly one split.
- Use upstream TN for compatible spoken forms.
- Use a custom digit speaker for `DIGIT_SEQUENCE`.
- Force generated spoken spans back through the selected realizer and reject
  failures.
- Generate about 20,000–40,000 train and 3,000–5,000 validation records first.

Exit condition: all records pass BIO, offset, split, provenance, and deterministic
round-trip validation.

## Phase 5: Prove contextual classification

- Fine-tune the pinned Microsoft DeBERTa-v3 Large token-classification encoder.
- Use a maximum sequence length of 72 for the first experiment. This is the
  smallest limit that covers every frozen Dataset V1 record with the pinned
  DeBERTa-v3 tokenizer.
- Map subwords back to source words with `word_ids()`.
- Evaluate raw argmax span-and-class predictions before abstention.
- Inspect the central `four thirty` contrasts before the full benchmark.
- Make a second data pass only from observed confusion families.

Exit condition: the classifier reaches about 90% or better on the POC hard set
with strong digit-sequence behavior. Stop if it remains near 80–85% after one
serious error-directed correction.

## Phase 6: Prove end-to-end normalization

- Repair malformed BIO sequences deterministically.
- Route decoded spans to forced Rust realizers.
- Reject parser failures and overlaps.
- Apply successful edits right-to-left.
- Return normalized text and exact original-source provenance.
- Evaluate classifier-only, deterministic baseline, and complete hybrid results.

POC continuation gate:

```text
exact target values      >= 90%
hard contextual set      >= 90%
digit sequences          >= 95%
false destructive edits  <= 2%
```

The hybrid must substantially beat plain `text-processing-rs` on the hard
contextual set. A small improvement does not justify continuing.

## Phase 7: Prove streaming-prefix behavior and latency

- Replay every complete current transcript snapshot without cached model state.
- Measure class stability and exact growing values.
- Warm up before latency measurement.
- Report tagger-only and full-normalizer p50, mean, p90, p95, and maximum.

POC latency gate:

```text
p50 <= 30 ms
p95 <= 60 ms
```

## Phase 8: Production hardening

Only after the POC passes:

- calibrate confidence and tune abstention for precision;
- export ONNX and quantize to INT8;
- benchmark macOS ARM64 and Linux CPU;
- package model, tokenizer, labels, and Rust extension;
- test latest-only scheduling with real Premove transcript revisions.

Production target:

```text
critical-value exact accuracy  >= 95%
high-confidence precision      >= 99%
destructive edit rate          < 0.5%
hard contextual collisions     >= 95%
p95 local inference            <= 50 ms
```

Premove integration, an all-Rust runtime, additional classes, WASM, Core ML,
and release publishing require separate decisions after these gates pass.
