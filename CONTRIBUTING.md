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
| `docs/` | User documentation, learning content, and implementation internals |

## Documentation routes

Mintlify reads `docs.json` from the repository root. The planned public routes
use `/itn/` as the Mintlify deployment base path on `premove.dev`. Set that
base path in the Mintlify domain dashboard and proxy `/itn` and `/itn/*` from
the website host. The repository paths alone do not configure the public
domain. `tests/test_docs_routes.py` checks the intended twelve public routes.

Mintlify generates `sitemap.xml`, `robots.txt`, `llms.txt`, and `llms-full.txt`.
The sitemap and LLM indexes use the navigable pages by default. Do not add
hand-maintained copies. Every navigable page must have a distinct frontmatter
title and description so the generated search and LLM indexes carry useful
metadata. Keep the model card outside the navigation; it is maintained for the
Hugging Face artifact, not as a duplicate website page.

Before publishing, confirm that the chosen Mintlify plan supports a custom
`/itn` subpath. Mintlify's reverse-proxy setup for custom subpaths is an
Enterprise feature. Configure its domain and proxy according to the current
Mintlify instructions. After deployment, verify:

- All intended `https://premove.dev/itn/` pages return 200, while unknown pages
  return 404. Check the redirect from `/itn` to `/itn/`.
- Canonical URLs and sitemap locations use `https://premove.dev/itn/`, not a
  Mintlify preview domain. Check the deployed HTML, not local preview HTML.
- `https://premove.dev/itn/sitemap.xml`, `/itn/llms.txt`, and
  `/itn/llms-full.txt` return the expected public pages. Submit the sitemap
  in Search Console after verifying the domain.
- The website-owned `https://premove.dev/robots.txt` allows the docs and
  references the ITN sitemap. A `robots.txt` under `/itn/` does not control
  crawlers for the host. Keep other website sitemap entries intact.
- The host and proxy do not block AI crawlers or Markdown page access. Use
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
[`docs/internals/rust-candidate-coverage.md`](docs/internals/rust-candidate-coverage.md) and the
README when supported forms change.

## Invariants

- Rust owns deterministic realization and candidate semantics.
- Python does not duplicate a Rust parser or kind definition.
- The model chooses among valid candidates; it does not generate arbitrary
  written text.
- Exact decoding selects compatible, non-overlapping edits.
- Frozen evaluation artifacts are evidence, not tuning input.
- Cleanup and refactoring must not change released normalization behavior.
