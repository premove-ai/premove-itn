# Roadmap

## Phase 1: Freeze the judge

- Create 300–500 de-identified, human-reviewed cases.
- Include a held-out `context_contrast` split for ambiguous forms.
- Record exact spans, classes, and canonical values.
- Keep benchmark templates out of the generator.

Exit condition: the benchmark format and review process are stable.

## Phase 2: Generate training data

- Expand deterministic generators to 100,000–300,000 examples.
- Add spoken, numeric, mixed, punctuation, and ASR-style variants.
- Include at least as many ambiguity pairs and hard negatives as easy cases.
- Validate BIO transitions and original-text offsets.

Exit condition: every record passes schema and invariant checks.

## Phase 3: Prove contextual classification

- Fine-tune one small pretrained encoder for token classification.
- Map model subwords back to word tokens and original character offsets.
- Use constrained BIO decoding.
- Evaluate exact span-and-class accuracy and false-positive conversions.

Kill gate: stop near 80–85% accuracy. Continue at about 95% or better with
strong ID/time/date contrast results.

## Phase 4: Prove end-to-end normalization

- Route accepted classes to deterministic class-specific parsers.
- Add `DIGIT_SEQUENCE` and `ALPHANUMERIC` realizers only where needed.
- Preserve all text outside predicted spans byte-for-byte.
- Calibrate confidence on held-out data and add abstention.

Exit condition: at least 95% exact final values with very high precision.

## Phase 5: Prove deployment

- Export the tagger to ONNX and quantize it to INT8.
- Benchmark macOS ARM64 and Linux CPU.
- Target p50 below 20 ms and initial p95 below 50 ms.
- Replay complete, revised STT snapshots using latest-pending scheduling.

Only after these gates pass should this experiment become a standalone product
or gain native Rust, Core ML, browser, or true incremental inference work.
