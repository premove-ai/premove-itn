# Contributing

`premove-itn` tests whether context can select the correct deterministic ITN
parser. Keep every contribution focused on that hypothesis.

## Set up the repository

```bash
git clone https://github.com/premove-ai/premove-itn.git
cd premove-itn
uv sync --all-packages --all-groups
uv run pytest
uv run --package premove-itn-data pytest tools/data_pipeline/tests
uv run --package premove-itn-training pytest tools/training/tests
```

## Before opening a pull request

Run every command supported by the current revision:

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest
uv run --package premove-itn-data pytest tools/data_pipeline/tests
uv run --package premove-itn-training pytest tools/training/tests
uv build --package premove-itn
uv build --package premove-itn-data
uv build --package premove-itn-training
```

When Rust is present, also run:

```bash
cargo test --manifest-path rust/Cargo.toml
```

A change must include focused tests when it changes labels, offsets, dataset
validation, deterministic realization, decoding, rewriting, or evaluation.

## Architecture invariants

- The learned module predicts spans and structural classes only.
- A deterministic realizer produces every normalized value.
- Both classification and deterministic parsing must succeed before an edit.
- Text outside successful edits stays byte-for-byte unchanged.
- Public offsets refer to the original input text.
- Successful edits do not overlap.
- Uncertain input can be preserved instead of guessed.
- Streaming logic stays outside the normalizer.
- Both private tool projects may depend on `premove-itn`; the runtime must not
  depend on or ship either private tool project.

Read [docs/architecture.md](docs/architecture.md) before changing a public
interface or moving a seam.

## Data contributions

Never submit private transcripts, credentials, personal data, or customer data.
Human-reviewed cases must be de-identified.

Reviewed benchmark rules:

- Record exact source offsets, classes, and replacements.
- Keep IDs unique across reviewed files.
- Review expected interpretations manually.
- Keep `golden.json`, `hard.json`, and `prefixes.json` independent from
  synthetic template families.
- Do not modify a frozen evaluation file after observing model results. Add
  new cases to a future benchmark version.

Synthetic-data rules:

- Record the random seed and template family.
- Keep provenance-connected records in one split; do not split random examples
  independently.
- Add contrast families that reuse similar spoken values under different
  contexts.
- Verify generated spoken forms with the forced deterministic realizer.
- Do not use an LLM to assign ground-truth labels.
- Do not commit generated corpora or model artifacts.

## Scope discipline

During the proof, do not add:

- a worker process or JSON-lines runtime protocol;
- a second implementation of an upstream deterministic parser;
- true incremental Transformer state;
- ONNX, Core ML, WASM, or release packaging before the POC gate;
- Premove pipeline changes before the standalone benchmark passes;
- a public synthetic-data command;
- speculative normalization classes.

Use a focused branch. Explain why the change exists, the invariant it protects,
what would fail without it, and how the test demonstrates the behavior.
