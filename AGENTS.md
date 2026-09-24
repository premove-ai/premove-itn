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

- For semantic implementation changes, read the relevant implementation,
  direct callers, and tests before editing. For small mechanical or
  configuration changes, inspect only the target state and evidence needed to
  apply and validate the requested change.
- Make the smallest coherent change and preserve unrelated work.
- Keep `PremoveITN` as the single implementation behind the Python API and CLI.
- Do not change model, candidate, decoder, or normalization behavior in a
  cleanup-only change.
- A behavior-preserving inference refactor requires exact regression evidence;
  reject it if any frozen prediction changes.
- Do not add training data, customer transcripts, credentials, model weights,
  or generated benchmark output to Git.
- Do not move a published release tag or modify frozen release evidence.

## Plan authority

Treat an explicit user-supplied implementation plan as the accepted design
when the user asks to execute, apply, or implement it, or otherwise clearly
indicates that its decisions are settled. Do not treat a supplied plan as
accepted when the user asks to review, evaluate, compare, critique, validate,
or improve the plan itself. Those requests remain discovery work.

In execution mode, verify the repository facts needed to apply the plan,
execute it, and run the smallest validation that establishes the result. Do
not reopen settled decisions or repeat investigation that the plan has already
answered.

In execution mode, the coordinator performs small literal or mechanical
configuration, Git, GitHub, and workflow changes directly. Delegate meaningful
semantic code or workflow-logic work that requires software-engineering
judgment to `implementer` with the accepted plan and invariants. Do not route
to `explorer` or `architect`, or request pre-implementation review, merely
because a settled plan touches their domain.

Spawn `reviewer` when the user explicitly requests semantic review, a
repository rule requires independent review for the affected boundary, the
implementation materially changes protected runtime or public behavior, or
focused mechanical evidence cannot establish semantic correctness. Do not
spawn `reviewer` merely because `implementer` was used, the diff is large, the
change is semantic, a pull request will be opened, or independent review would
be generally useful.

Escalate execution only on concrete evidence of one of these conditions:

1. Invariant conflict: the plan violates an explicit repository invariant.
2. Reality mismatch: code, API, or configuration materially differs from the
   plan's assumptions.
3. Insufficient specification: a consequential choice remains undecided.
4. Validation contradiction: focused tests or observed behavior contradict the
   plan's stated result.

Report the specific evidence and route only the unresolved question. A plan
does not override repository invariants, observed code, or failing tests.
Enter discovery mode when the user asks Codex to determine what, why, how,
which design, whether a change is safe, or what caused a failure.

## Model routing

The coordinator owns routing, synthesis, and final decisions. It must delegate
work to the configured named specialist when the task meets a role trigger
below.

Use the model and reasoning pins in `.codex/config.toml` and
`.codex/agents/*.toml`. Use only the four configured roles. Do not use untyped
or default subagents, or request per-run model or reasoning overrides.

- Use CodeGraph directly for structural questions that it can answer without
  substantial source investigation.
- Spawn `explorer` for a bounded, unresolved investigation that requires
  tracing behavior, comparing implementations, or inspecting multiple source
  or test files. Its output is evidence, not design authority.
- Spawn `implementer` for meaningful semantic code or workflow-logic work that
  requires software-engineering judgment when the design and invariants are
  settled.
- Spawn `reviewer` only for the explicit positive triggers in plan authority.
  Review realization of the accepted plan.
- Spawn `architect` for an unresolved decision or concrete invariant conflict
  involving candidate or GoldGraph semantics, the Rust/Python ownership
  boundary, scoring or model inputs, structured loss or decoder legality,
  training or evaluation policy, public context or result meaning, or model,
  package, or release identity.

When work satisfies a named role's trigger, delegation to that role is
mandatory. The coordinator handles the direct execution work defined above
and structural lookups. Do not force every task through the full agent pipeline.

Apply role triggers to specialist work required by the current task. Do not
spawn a specialist only because an existing branch, diff, or artifact contains
substantial prior work. The coordinator handles deterministic and administrative
operations such as status checks, branch comparisons, pushes, and opening or
integrating already-reviewed pull requests. Spawn a specialist only if that work
uncovers a new need for investigation, implementation, semantic review, or an
architectural decision.

Delegate independent tasks only. Coordinate changes to shared files.
If two evidence-driven fixes fail and the cause remains unclear, stop the
implementation path and return the evidence to the coordinator.

## Git

- Treat `main` as the stable release branch.
- Use one short-lived branch and one focused pull request per change.
- Do not push directly to `main` or force-push shared branches.
- Merge only after required checks pass.

## Verification

During implementation, use the focused gate with the semantic tests relevant to
the change:

```bash
uv run --locked python scripts/agent/check.py focused --pytest <test>
```

Prefer the smallest validation that establishes the change locally. Do not run
the complete local gate only to duplicate checks that pull-request CI runs from
a clean checkout.

Use the complete local gate when the change is cross-cutting, modifies the
validation harness or CI, prepares a release, CI is unavailable, or focused
evidence is insufficient:

```bash
uv run --locked python scripts/agent/check.py full
```

GitHub Actions is the authoritative completion gate for pull requests.
Do not repeat an already-passing local gate only to open or integrate an
unchanged, already-reviewed branch. Confirm the branch state and expected diff,
then rely on the target pull request's authoritative CI unless prior validation
is stale, incomplete, or the integration introduces new changes.

## Command output

RTK may compact supported shell output. When compact output omits information
needed to continue:

1. Use the printed `rtk recall <hash>` handle.
2. Use `rtk proxy <command>` if raw re-execution is required.
3. Use `RTK_DISABLED=1 <command>` only when RTK behavior itself is under
   investigation.

<!-- CODEGRAPH_START -->
## CodeGraph

In repositories indexed by CodeGraph (a `.codegraph/` directory exists at the repo root), reach for it BEFORE grep/find or reading files when you need to understand or locate code:

- **MCP tool** (when available): `codegraph_explore` answers most code questions in one call — the relevant symbols' verbatim source plus the call paths between them, including dynamic-dispatch hops grep can't follow. Name a file or symbol in the query to read its current line-numbered source. If it's listed but deferred, load it by name via tool search.
- **Shell** (always works): `codegraph explore "<symbol names or question>"` prints the same output.

If there is no `.codegraph/` directory, skip CodeGraph entirely — indexing is the user's decision.
<!-- CODEGRAPH_END -->
