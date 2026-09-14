# Contributing

Premove ITN has one production path: Rust candidate generation, contextual
scoring, and exact decoding behind the `PremoveITN` Python API and CLI. Keep
changes focused and preserve this boundary.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/premove_itn/` | Python runtime and public API |
| `rust/` | Deterministic realization rules |
| `benchmarks/` | Optional comparison and artifact-verification tools |
| `eval/` | Frozen benchmark and retained results |
| `examples/` | Minimal Python and stdin examples |
| `scripts/` | Release checks and development utilities |
| `tests/` | Python regression tests |
| `itn/` | Published ITN documentation, learning content, and implementation internals |
| `docs/` | Retained model and evaluation records outside the site navigation |

## Documentation routes

Mintlify reads `docs.json` from the repository root. This site serves the
Premove overview at `https://docs.premove.dev/` and the ITN documentation at
`https://docs.premove.dev/itn/`. The top navigation has Premove and Premove ITN
tabs. Configure `docs.premove.dev` as a custom domain in Mintlify and set its
DNS record at the domain provider. `tests/test_docs_routes.py` checks the
intended public routes.

Mintlify generates `sitemap.xml`, `robots.txt`, `llms.txt`, and `llms-full.txt`.
The sitemap and LLM indexes use the navigable pages by default. Do not add
hand-maintained copies. Every navigable page must have a distinct frontmatter
title and description so the generated search and LLM indexes carry useful
metadata. Keep the model card outside the navigation; it is maintained for the
Hugging Face artifact, not as a duplicate website page.

After deployment, verify:

- The overview at `https://docs.premove.dev/` and all intended `/itn/` pages
  return 200, while unknown pages return 404.
- Canonical URLs and sitemap locations use `https://docs.premove.dev/`, not a
  Mintlify preview domain. Check the deployed HTML, not local preview HTML.
- `https://docs.premove.dev/sitemap.xml`, `/llms.txt`, and
  `/llms-full.txt` return the expected public pages. Submit the sitemap
  in Search Console after verifying the domain.
- `https://docs.premove.dev/robots.txt` allows crawlers and references the
  sitemap.
- The host does not block AI crawlers or Markdown page access. Use
  `mint score` against the deployed docs URL to find remaining GEO gaps.

## Pull requests

Create a short-lived branch from `main` and open one focused pull request.
Explain the problem, the protected invariant, and the validation evidence. Do
not push directly to `main`. Merge only after the required checks pass.

Bug reports and tests must use synthetic or de-identified text. Do not include
private transcripts, customer data, credentials, model weights, generated
training corpora, or new outputs derived from the frozen evaluation dataset.

## Validation

Run:

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest
cargo test --manifest-path rust/Cargo.toml
uv build
python scripts/inspect_release_artifact.py dist/*.whl dist/*.tar.gz
uv run python scripts/check_local_links.py
uv run python scripts/check_frozen_boundaries.py
```

Pull requests run the same core checks in GitHub Actions. Before a release,
maintainers manually run the supported platform matrix and frozen prediction
equivalence gate. The release workflow repeats its required release checks.

Add focused tests for changes to a candidate kind, parser route, decoder, model
loader, or public output. Update
[`itn/docs/internals/rust-candidate-coverage.md`](itn/docs/internals/rust-candidate-coverage.md) and the
README when supported forms change.

## Invariants

- Rust owns deterministic realization and candidate semantics.
- Python does not duplicate a Rust parser or kind definition.
- The model chooses among valid candidates; it does not generate arbitrary
  written text.
- Exact decoding selects compatible, non-overlapping edits.
- Frozen evaluation artifacts are evidence, not tuning input.
- Cleanup and refactoring must not change released normalization behavior.
