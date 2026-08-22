## Summary

<!-- Explain what changed and why. -->

## Protected invariant

<!-- State the invariant this change protects and what would fail without it. -->

## Validation

<!-- List the exact commands and relevant results. -->

- [ ] `uv run ruff format --check .`
- [ ] `uv run ruff check .`
- [ ] `uv run pytest`
- [ ] `uv build`

## Data and privacy

- [ ] This change does not add private, identifying, or customer transcript data.
- [ ] New generated data is reproducible from a recorded seed.
- [ ] New training templates do not duplicate held-out benchmark templates.
- [ ] Not applicable; this change does not affect data.

## Scope

- [ ] The change is focused and does not add unneeded runtime or model complexity.
- [ ] Behavior changes include focused tests.
- [ ] Documentation is updated when the public contract changes.
