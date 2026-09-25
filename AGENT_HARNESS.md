# Agent harness

This guide summarizes the repository's Codex setup. The detailed rules live in
[AGENTS.md](AGENTS.md); setup and bootstrap steps live in
[CONTRIBUTING.md](CONTRIBUTING.md).

## Agents and when to use them

The coordinator is `gpt-6-luna` at `xhigh`. It routes work, combines results,
and handles small mechanical, configuration, Git, GitHub, and workflow tasks.
The four named specialists are configured in `.codex/agents/`:

| Agent | Model | Use it for |
| --- | --- | --- |
| `explorer` | `gpt-6-luna`, `high` | A bounded, unresolved investigation. It gathers evidence and does not edit. |
| `implementer` | `gpt-6-sol`, `medium` | Meaningful code or workflow logic when the design is settled. |
| `reviewer` | `gpt-6-sol`, `high` | Independent review when the user asks, a repository rule requires it, protected behavior changes, or tests cannot establish correctness. |
| `architect` | `gpt-6-astra`, `medium` | An unresolved high-impact design decision or a concrete invariant conflict. |

## How `AGENTS.md` routes work

An explicit plan is accepted when the user asks to execute it or clearly says
its decisions are settled. Requests to critique or compare a plan stay in
discovery mode. In execution mode, the coordinator verifies only the facts
needed to apply the plan. It escalates only for an invariant conflict, a
material mismatch with the repository, a consequential missing decision, or
validation that contradicts the plan. Specialists answer unresolved questions;
they do not reopen settled choices.

The configured `reviewer` agent handles independent review when routing rules
call for it.

## Code navigation and output

The bootstrap command installs or verifies the pinned CodeGraph and RTK tools:

```bash
uv run --locked python scripts/agent/bootstrap.py
```

`scripts/agent/start_change.py` starts a branch from a fetched remote base and
requires a clean working tree.

CodeGraph is configured as the repository MCP server. Use it first for
structural questions in the indexed codebase; it returns symbols and call
paths.

The `.codex/hooks.json` Bash hook sends supported commands through the
repository's `scripts/agent/rtk_hook.py` adapter. RTK shortens supported shell
output and preserves a way to recover the full output. When a compact result
omits needed detail, use `rtk recall <hash>`. Use `rtk proxy <command>` to
rerun a command with raw output. Set `RTK_DISABLED=1` only when investigating
RTK itself.

## Validation

`scripts/agent/check.py` provides focused and full validation profiles. Focused
validation checks changed files and runs the selected semantic tests:

```bash
uv run --locked python scripts/agent/check.py focused --pytest <test>
```

Use the full profile for cross-cutting changes or when focused evidence is not
enough:

```bash
uv run --locked python scripts/agent/check.py full
```
