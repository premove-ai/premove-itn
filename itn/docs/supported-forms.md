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

## Current boundaries

Premove ITN does not provide first-class normalization for non-English speech,
street addresses, free-form rewriting, or arbitrary application-specific
formats. Text remains unchanged when the decoder does not select a candidate.
The model cannot create a written form that the Rust candidate layer did not
offer. If an application requires a stricter output schema, validate the
result before passing it to a tool.

For the exact accepted grammar and edge cases, see
[Rust candidate coverage](/itn/docs/internals/rust-candidate-coverage).
