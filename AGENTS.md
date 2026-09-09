# Premove ITN

Premove ITN is a contextual inverse text normalization system for English
voice-agent transcripts. Rust generates valid written candidates, a frozen
DeBERTa scorer ranks them in sentence context, and an exact decoder selects
compatible edits.

## Boundaries

- `rust/` owns deterministic realization rules and candidate generation.
- `src/premove_itn/` owns the Python API, model loading, scoring, and decoding.
- `eval/voice_agent_itn/` is frozen release evidence. Do not tune against it.
- The public model artifact is `premove-ai/premove-itn` on Hugging Face.
- Python must not duplicate a Rust realizer or maintain a second kind list.

## Change rules

- Read the relevant implementation, direct callers, and tests before editing.
- Make the smallest coherent change and preserve unrelated work.
- Keep `PremoveITN` as the single implementation behind the Python API and CLI.
- Do not change model, candidate, decoder, or normalization behavior in a
  cleanup-only change.
- A behavior-preserving inference refactor requires exact regression evidence;
  reject it if any frozen prediction changes.
- Do not add training data, customer transcripts, credentials, model weights,
  or generated benchmark output to Git.
- Do not move a published release tag or modify frozen release evidence.

## Git

- Treat `main` as the stable release branch.
- Use one short-lived branch and one focused pull request per change.
- Do not push directly to `main` or force-push shared branches.
- Merge only after required checks pass.

## Verification

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest
cargo test --manifest-path rust/Cargo.toml
uv build
uv run python scripts/check_local_links.py
uv run python scripts/check_frozen_boundaries.py
```
