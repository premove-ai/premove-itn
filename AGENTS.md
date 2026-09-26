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

- Treat an explicit user-approved plan as settled when asked to implement, apply, or execute it.
- Review, critique, compare, or evaluate a plan in discovery mode. Do not treat it as settled.
- Do not rediscover or redesign settled work.

Escalate execution only for:

1. An explicit invariant conflict.
2. A material mismatch between the plan and repository reality.
3. A consequential unresolved choice.
4. Validation that contradicts the plan.

Escalate only the unresolved question. Keep the rest of the plan settled.

## Model routing

The root agent owns investigation, implementation, validation, Git, GitHub, and final decisions. Use CodeGraph for structural navigation.

Spawn a configured named subagent only when separate context provides a clear benefit:

- Independent work can run in parallel with root work.
- A large investigation should be isolated from root context.
- An independent semantic review is required.
- An unresolved architecture decision requires stronger reasoning.

Do not spawn for simple edits, sequential investigation or implementation, or work that depends on the root's current context. Do not inspect source merely to prepare a subagent task; either handle the work directly or delegate the bounded question immediately.

- Independent investigation → `explorer`.
- Independent semantic review → `reviewer`.
- Unresolved architecture or invariant decision → `architect`.

Use `reviewer` only when the user requests review or protected semantic behavior changes and deterministic evidence is insufficient. When review is needed, overlap it with CI or independent PR work when possible. An accepted plan alone does not trigger `architect`.

For Premove ITN, unresolved architecture or invariant decisions may involve candidate or GoldGraph semantics, the Rust/Python ownership boundary, scoring or model inputs, structured loss or decoder legality, training or evaluation policy, public context or result meaning, or model, package, or release identity.

Do not delegate a dependent step that the root can do directly. Keep subagent tasks bounded and returned context minimal. Pass the bounded task explicitly and use `fork_turns="none"` by default. Inherit only the minimum recent turns required by the task.

Do not use default or untyped subagents or override configured models or reasoning levels.

If two evidence-driven fixes fail and the cause remains unclear, stop and return the evidence.

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
