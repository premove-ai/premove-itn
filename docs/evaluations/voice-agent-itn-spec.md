# VoiceAgent ITN frozen benchmark

## Status

First Evaluation has a retained release-artifact run under
`eval/voice_agent_itn/results/first-evaluation/`. It was run after exploratory work. It must
not be described as a blind run. The pending human adjudication
and incomplete similarity contamination checks remain limitations.

The main VoiceAgent ITN dataset contains 1,500 rows. It is frozen after an
independent audit of the complete JSONL found no
dataset blocker. The frozen SHA-256 is
`782b14d0291e4e176e4ba22d0d756019e906a5ea4c7d267d9703e80b082cbd61`.

Content freeze and launch readiness are separate states. Blind human gold
adjudication, contamination review, backend revision freeze, scorer freeze,
and adapter-isolation verification remain mandatory before the first run. Do
not alter this dataset to satisfy a backend after any output is observed. A confirmed
future gold defect requires a new benchmark version.

This is a balanced synthetic English ITN stress benchmark. It is not an IID
sample of production traffic. Its normal, voice-agent, multi-entity, and
collision groups intentionally use controlled sentence shells.

## Backend-neutral contract

The backend-neutral semantic category and `premove_span_kind` are different
fields. Every non-null Premove kind must be a current public `SpanKind` member.
Each semantic category maps to at most one Premove kind. URL maps to
`ELECTRONIC`. PERCENT remains a required benchmark category and has a null
Premove mapping because Premove has no public PERCENT kind. This dataset does
not add or change a Premove realization rule.

The headline inference adapter receives only `text`. It must not receive gold
spans, categories, Premove kinds, domains, difficulty, or semantic values. An
oracle span-kind realizer result can be reported only as a separate diagnostic.
Premove versus text-processing-rs is an upstream ablation comparison, not a
comparison of fully independent systems.

## Semantic equivalence

Semantic scoring is category-aware. It does not compare raw strings. It
preserves numeric value, currency, unit, digit order, leading zero, entity
boundary, and structured dialing form. PHONE values contain `digits` and an
`international` flag. Identifier values ignore case when speech does not encode
case. Strict sentence exact match still requires the canonical uppercase style.
Decimal padding, harmless phone separators, time padding, AM/PM punctuation,
money trailing zeros, date capitalization, and measurement spacing can differ
without changing semantic correctness.

The money-versus-time collision explicitly says dollars. Room-code collisions
use REFERENCE_ID, not ORDER_ID. The old single-payload date-versus-modal family
is replaced by 25 date-versus-reference contrasts with 25 distinct shared
spoken payloads.

Inputs can contain already-written context beside an unresolved spoken entity.
This models partially normalized ASR output. Only declared spans are scored as
positive entities.

## Frozen reporting rules

The primary result is macro semantic entity accuracy. Also report micro
semantic entity accuracy, strict sentence exact match, family-level collision
pair accuracy, multi-entity span and all-entities-correct accuracy, standalone
KEEP false-normalization rate, all-KEEP preservation accuracy, and
feature-based difficulty diagnostics. Do not treat synthetic rows or pairs as
IID trials. Use clustered analysis by ambiguous surface or collision family if
uncertainty is reported.

Voice-agent macro-domain accuracy uses only the 400 rows where
`group=voice_agent`. Multi rows are compositional evidence, not voice-domain
evidence. Standalone KEEP uses 50 negative rows. All-KEEP preservation uses all
100 no-span rows. The 50 collision KEEP rows remain part of pair accuracy and must not
be triple-counted in a composite.

The Standard ITN Core and Voice-Agent Structured Extension memberships must be
fixed evaluator constants. Backend support must not determine membership.

## Gates before the first blind run

Independently label all rows without showing proposed targets. Adjudicate every
disagreement and review collision, multi-entity, and KEEP cases. Check exact,
normalized, token n-gram, and embedding overlap against all training,
development, and earlier evaluation sources. Then freeze the exact dataset and
all backend commits before any system receives an input.
