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

- [ ] This change adds no private or customer data, model weights, training data, or new outputs derived from the frozen evaluation dataset.

## Scope

- [ ] The change is focused and does not add unneeded runtime or model complexity.
- [ ] Behavior changes include focused tests.
- [ ] Documentation is updated when the public contract changes.
