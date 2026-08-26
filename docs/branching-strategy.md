# Branching strategy

This repository uses short-lived topic branches and one integration branch per
MVP slice. `main` is the validated MVP branch. It is not the place to assemble
unfinished work.

## Branch roles

| Branch | Purpose | Created from | Merges into |
| --- | --- | --- | --- |
| `main` | Stable, validated MVP baseline | — | — |
| `epic/<epic-name>` | Integrates one coherent MVP slice | `main` | `main` |
| `<type>/<epic-name>/<short-name>` | Implements one focused change | the active epic | the active epic |

Use a short lowercase slug. Valid topic types include `feat`, `fix`, `test`,
`docs`, `refactor`, and `chore`. Add the issue number when the repository uses
issue tracking, for example `feat/42-digit-sequence`.

Examples:

```text
epic/deterministic-mvp
feat/deterministic-mvp/digit-sequence
test/deterministic-mvp/parser-routing
docs/deterministic-mvp/public-contract
```

Older branches do not need to be renamed. Apply this naming scheme to new
work.

## Normal flow

1. Define the MVP slice and its acceptance evidence in an issue or project
   note.
2. Create one epic branch from the latest `main`:

   ```bash
   git fetch origin --prune
   git switch main
   git pull --ff-only origin main
   git switch -c epic/<epic-name>
   ```

3. Create a short-lived topic branch from that epic:

   ```bash
   git switch -c <type>/<epic-name>/<short-name>
   ```

4. Keep the topic branch narrow. Add focused tests with behavior changes. Keep
   unrelated cleanup out of the branch.
5. Open the topic pull request against the epic branch. Review and squash
   merge it into the epic. Delete the topic branch after the merge.
6. Keep integrating topic branches until the epic meets its MVP acceptance
   criteria.
7. Run the full repository validation. Open one epic pull request against
   `main` with the acceptance evidence and validation results.
8. Merge the epic pull request with a merge commit. This preserves the MVP
   integration boundary. Delete the epic branch after the merge.

The merge direction is therefore:

```text
topic branch  --squash-->  epic branch  --validated merge commit-->  main
```

Do not open normal topic pull requests directly against `main`. Do not merge
an incomplete epic into `main`. Do not force-push a shared epic branch.

## Merge gates

Every topic pull request must have:

- a focused scope and a clear reason for the change;
- relevant tests or an explicit reason that tests are not needed; and
- review of the protected invariant and the changed behavior.

An epic pull request may target `main` only when:

- the MVP acceptance criteria are complete and recorded;
- the full validation commands pass;
- the public contract and documentation are current; and
- the pull request explains what would fail without the change.

Use the repository validation commands from `CONTRIBUTING.md`:

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest
cargo test --manifest-path rust/Cargo.toml
uv build
```

## Keeping `main` protected

Configure the remote repository to protect `main` with these rules:

- require pull requests;
- require the repository validation checks to pass;
- block force-pushes and branch deletion; and
- require at least one review for the epic-to-`main` pull request.

Protect active epic branches with pull requests as well when more than one
person is contributing. Branch protection is a remote GitHub setting. This
document defines the workflow, but does not configure that setting.
