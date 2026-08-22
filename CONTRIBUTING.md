# Contributing

This repository tests one narrow hypothesis: context can select the correct ITN
class, while deterministic code remains responsible for the exact rewrite.

## Set up the project

```bash
git clone https://github.com/premove-ai/premove-itn.git
cd premove-itn
uv sync --all-groups
uv run pytest
```

Use a focused branch. Keep generated datasets, trained weights, and local model
exports out of Git.

## Before opening a pull request

```bash
uv run ruff format .
uv run ruff check .
uv run pytest
uv build
```

A change must include tests when it changes data generation, token labels,
character offsets, decoding, realization, or rewriting.

## Project invariants

- The learned component predicts spans and semantic classes only.
- A deterministic realizer produces every normalized value.
- Text outside an accepted span stays byte-for-byte unchanged.
- Character offsets refer to the original input text.
- Generated examples are reproducible from a recorded seed.
- Exact templates in the held-out benchmark must not enter training data.
- Ambiguous low-confidence input must be eligible for abstention.

## Data contributions

Do not submit private transcripts, credentials, personal data, or customer data.
Human-reviewed benchmark cases must be de-identified. Add provenance and license
information for imported public datasets.

When adding a template family, add a contrasting family that can produce the
same spoken form with a different class. Easy numeric examples alone do not test
the project hypothesis.

## Scope

Do not add streaming state, attention caches, a Python worker protocol, or a new
normalization engine during the initial proof. The first model operates on each
complete transcript snapshot. Runtime work starts only after the accuracy gate
in [ROADMAP.md](ROADMAP.md) passes.
