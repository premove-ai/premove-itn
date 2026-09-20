---
title: Premove ITN supported forms
sidebarTitle: Supported forms
description: Structured values and English spoken forms supported by Premove ITN.
---

Premove ITN targets English structured values commonly used by voice agents.
The table shows valid written candidates. The contextual scorer can leave a
phrase unchanged or select another candidate when the sentence supports it.

| Form | Spoken | Written |
| --- | --- | --- |
| Numbers | `one hundred twenty` | `120` |
| Digit sequences | `zero eight two zero six three` | `082063` |
| Times | `four thirty` | `04:30` |
| Dates | `march fifth twenty twenty four` | `march 5 2024` |
| Money | `twenty dollars` | `$20` |
| Decimals | `one point five` | `1.5` |
| Measurements | `two hundred meters` | `200 m` |
| Ordinals | `the eighth` | `8th` |
| Phone numbers | `eight one four two three one four` | `814-2314` |
| Email addresses and URLs | `support at example dot com` | `support@example.com` |
| Versions and identifiers | `v three dot one dot nine` | `v3.1.9` |
| Punctuation | `comma` | `,` |
| Approved abbreviations | `doctor` | `dr.` |

The deterministic Rust layer exposes 13 candidate kinds: `DIGIT_SEQUENCE`,
`CARDINAL`, `TIME`, `DATE`, `MONEY`, `DECIMAL`, `PHONE`, `ELECTRONIC`,
`MEASUREMENT`, `ORDINAL`, `PUNCTUATION`, `WHITELIST`, and `WORD`.

## Contextual temporal resolution

The v0.3.0 Python layer can enrich selected `DATE` candidates and unchanged
temporal text after decoding. It does not add context-dependent candidates to
the Rust graph and it never performs a second model call.

| Expression family | Examples | Required context |
| --- | --- | --- |
| Relative dates | `today`, `tomorrow`, `yesterday`, `day after tomorrow` | `reference_datetime` |
| Bounded offsets | `in two days`, `one week from today`, `a week ago` | `reference_datetime` |
| Calendar weekdays | `next Monday`, `this Friday`, `last Sunday` | `reference_datetime` |
| Named dates | `September 30`, `30th September 2026` | Reference year only when the year is missing |
| Weekday-qualified dates | `Thursday, September 30` | Reference year when missing; weekday must agree |
| Numeric dates | `03/04/2026`, `24.09`, `2026-09-30` | `DateOrder` only when ambiguous; reference year when missing |

Resolved calendar dates use ISO `YYYY-MM-DD`. Missing or contradictory context
leaves the expression unresolved instead of guessing. Numeric separators are
part of the surface form, not separate context fields. `DateOrder` describes
the field order only and supports `DMY`, `DYM`, `MDY`, `MYD`, `YDM`, and `YMD`.

Provide context explicitly through `NormalizationContext`; the runtime does
not discover the current clock, timezone, or locale.

## Current boundaries

Premove ITN does not provide first-class normalization for non-English speech,
street addresses, free-form rewriting, or arbitrary application-specific
formats. Text remains unchanged when the decoder does not select a candidate.
The model cannot create a written form that the Rust candidate layer did not
offer. If an application requires a stricter output schema, validate the
result before passing it to a tool.

For the exact accepted grammar and edge cases, see
[Rust candidate coverage](/itn/docs/internals/rust-candidate-coverage).
