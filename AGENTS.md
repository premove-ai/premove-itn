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

Treat an explicit implementation plan as settled when the user asks to apply
it. Do not rediscover or redesign settled decisions. Escalate only when there
is concrete evidence of one of these conditions:

1. Invariant conflict: the plan violates an explicit repository invariant.
2. Reality mismatch: code, API, or configuration materially differs from the
   plan's assumptions.
3. Insufficient specification: a consequential choice remains undecided.
4. Validation contradiction: focused evidence contradicts the plan's result.

Report the evidence and resolve only the conflicting or undecided part. A plan
does not override repository invariants, observed code, or failing checks.

## Model routing

The root agent owns repository inspection, implementation, validation, Git,
GitHub, and final decisions. It handles work directly when steps depend on
shared context or must happen in sequence.

Use the model and reasoning pins in `.codex/config.toml` and
`.codex/agents/*.toml`. Use only the configured specialist roles. Do not use
untyped or default subagents, or request per-run model or reasoning overrides.

- Use CodeGraph directly for structural questions that it can answer without
  substantial source investigation.
- Use `explorer` only for independent work that can run in parallel or for a
  large investigation whose isolated context saves root context.
- Use `reviewer` when the user asks for independent review, or when a
  high-risk semantic change cannot be established by deterministic evidence.
- Use `architect` only for an unresolved design decision or invariant conflict
  involving candidate semantics, Rust/Python ownership, scoring or model
  inputs, decoder legality, evaluation policy, public result meaning, or model,
  package, or release identity.

Do not spawn a specialist for a simple edit, a single-file task, a sequential
step, or work that depends on the root's current context. Do not delegate work
that the root must wait for when the root can do it directly. Do not inspect
source merely to prepare a child task. Either handle the investigation
directly or delegate its bounded question before inspecting the same sources.

Keep specialist tasks bounded and reports compact. Explorer reports a finding,
file and line evidence, and impact in at most 500 words. Reviewer reports only
actionable findings with severity, file and line, and why each matters; if
there are none, report `No findings.` Architect reports the decision, reason,
affected invariants, and required tests. When independent review is required,
run it alongside CI or pull request administration when possible.

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
