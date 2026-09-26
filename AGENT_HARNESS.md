# Agent harness

This guide summarizes the repository's Codex setup. The detailed rules live in
[AGENTS.md](AGENTS.md); setup and bootstrap steps live in
[CONTRIBUTING.md](CONTRIBUTING.md).

## Agents and when to use them

The root agent is `gpt-6-sol` at `medium`. It owns repository inspection,
implementation, validation, Git, GitHub, and final decisions. Three named
specialists are configured in `.codex/agents/`:

| Agent | Model | Use it for |
| --- | --- | --- |
| `explorer` | `gpt-6-luna`, `high` | Independent investigation that can run in parallel, or a large investigation that benefits from context isolation. |
| `reviewer` | `gpt-6-sol`, `high` | Independent review when the user asks or a high-risk semantic change needs review beyond deterministic evidence. |
| `architect` | `gpt-6-astra`, `medium` | An unresolved design decision or concrete invariant conflict. |

The session allows at most two concurrent subagents.

## Plan authority and routing

An explicit implementation plan is settled when the user asks to apply it.
The root checks only the facts needed to apply it. The root escalates only for
an invariant conflict, a material mismatch with the repository, a missing
consequential decision, or validation that contradicts the plan.

Use a specialist only when separate context has a clear benefit: parallel work,
context isolation for a large investigation, independent high-risk review, or
an unresolved architecture decision. Handle simple, single-file, sequential,
and shared-context work in the root. Do not inspect source just to prepare a
child task. Either investigate directly or delegate the bounded question
before inspecting the same sources.

## Subagent reports

Keep specialist reports to 500 words or less:

- Explorer: finding, file and line evidence, impact.
- Reviewer: actionable findings with severity, file and line, and why each
  matters. Say `No findings.` when there are none.
- Architect: decision, reason, affected invariants, required tests.

Run independent review alongside CI or pull request administration when
possible.

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
