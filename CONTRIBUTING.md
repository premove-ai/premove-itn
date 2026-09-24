# Contributing

Premove ITN has one production path: Rust candidate generation, contextual
scoring, and exact decoding behind the `PremoveITN` Python API and CLI. Keep
changes focused and preserve this boundary.

## Agent development setup

Development requires Git, the `uv` version pinned in `uv.toml`, and a Rust
toolchain for Rust and package validation. The agent harness also expects Codex
and GitHub CLI when those integrations are used. Install `uv`
with [Astral's official installer](https://docs.astral.sh/uv/getting-started/installation/)
if it is not already available. Python is not an external prerequisite because
`uv` provisions the required interpreter. After cloning, run:

```bash
uv run --locked python scripts/agent/bootstrap.py
```

The bootstrap command installs or verifies the pinned development tools and
creates machine-local generated state such as the CodeGraph index. Generated
harness state is not committed and is not part of the Premove ITN package.
When Codex first opens the clone, accept its normal repository-trust prompt if
shown. The project-local `.codex/config.toml` is active only for a trusted
repository. Review and trust the project RTK hook if Codex prompts for hook
trust. Bootstrap does not grant either form of trust automatically.

Use the repository-owned workflow commands for routine agent work:

```bash
uv run --locked python scripts/agent/start_change.py --base main --branch <branch>
```

`start_change.py` requires a clean working tree and creates the requested branch
directly from the fetched remote base. It does not commit, push, or create a
pull request. Run bootstrap again to verify or repair the installed harness.

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

During implementation, run focused mechanical checks and name the semantic
tests that cover the change:

```bash
uv run --locked python scripts/agent/check.py focused --pytest <test>
```

Prefer the smallest validation that establishes the change locally. Do not run
the complete local gate only to duplicate checks that pull-request CI runs from
a clean checkout.

Run the complete local gate when a change crosses two or more of the Python,
Rust, and packaging boundaries; modifies the validation harness or CI; prepares
a release; cannot use CI; or lacks sufficient focused evidence:

```bash
uv run --locked python scripts/agent/check.py full
```

GitHub Actions is the authoritative completion gate for pull requests. If CI
fails, reproduce the failure with the smallest relevant local gate before
pushing a fix. Before a release, maintainers manually run the supported
platform matrix and frozen prediction equivalence gate. The release workflow
repeats its required release checks.

## Package releases

The Python package and trained model have separate release identities. A
runtime-only release can reuse the frozen model artifact; document that
relationship in the changelog and README instead of creating a duplicate Hub
model tag. For a package release, update `pyproject.toml`, `rust/Cargo.toml`,
`uv.lock`, the CLI/version tests, and the version-specific references in
`.github/workflows/release.yml`. Run the full validation suite, build and
inspect the artifacts, then create the matching `v<package-version>` tag. The
tag-triggered workflow builds the supported wheels, verifies inference, stages
the GitHub release, publishes PyPI, and runs public-install checks.

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
