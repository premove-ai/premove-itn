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

## Model routing

Use repository subagents when they isolate substantial context or distinct
reasoning work. Do not delegate work that the coordinator can complete as
reliably with less overhead. Prefer deterministic tools, including CodeGraph for
indexed code, when they answer the question directly.

- Use `explorer` for bounded code discovery and evidence gathering. Its output
  is evidence, not design authority.
- Use `implementer` for meaningful production changes when the design and
  invariants are settled. The coordinator may complete small mechanical changes
  directly when delegation would add overhead.
- Use `reviewer` after meaningful behavior changes or when semantic correctness
  cannot be established mechanically.
- Use `architect` before changing candidate or GoldGraph semantics, the
  Rust/Python ownership boundary, scoring or model inputs, structured loss or
  decoder legality, training or evaluation policy, public context/result
  meaning, or model/package/release identity.

If an implementer finds an architectural ambiguity, conflicting invariant, or
required contract change, stop that implementation path and escalate to the
coordinator. If two evidence-driven fixes fail and the cause remains unclear,
return the evidence for coordinator debugging. Delegate independent tasks only;
coordinate changes to shared files.

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

<!-- CODEGRAPH_START -->
## CodeGraph

In repositories indexed by CodeGraph (a `.codegraph/` directory exists at the repo root), reach for it BEFORE grep/find or reading files when you need to understand or locate code:

- **MCP tool** (when available): `codegraph_explore` answers most code questions in one call — the relevant symbols' verbatim source plus the call paths between them, including dynamic-dispatch hops grep can't follow. Name a file or symbol in the query to read its current line-numbered source. If it's listed but deferred, load it by name via tool search.
- **Shell** (always works): `codegraph explore "<symbol names or question>"` prints the same output.

If there is no `.codegraph/` directory, skip CodeGraph entirely — indexing is the user's decision.
<!-- CODEGRAPH_END -->
