---
title: Why voice agents need inverse text normalization
description: How ITN turns voice transcripts into reliable tool and API arguments.
---

Voice agents often receive spoken-form transcripts while tools and APIs expect
structured written values.

```text
ASR transcript:
"my order id is seven eight three two nine"

API argument:
order_id="78329"
```

## Where ITN fits

```text
speech → ASR → ITN → entity or tool argument → API
```

ITN converts the transcript before an agent extracts values or calls a tool.
This reduces avoidable formatting failures at the API boundary.

For example, an order lookup may use an exact string key. If the transcript
contains `seven eight three two nine` but the lookup receives the spoken words
instead of `78329`, the tool may not find the order. The same issue applies to
phone digits, confirmation codes, and amounts.

## Values that voice agents exchange

Common values include order IDs, phone numbers, dates, times, amounts, room
codes, email addresses, URLs, and alphanumeric identifiers. See the complete
[supported forms](/docs/supported-forms).

## Ambiguity affects tool calls

`two thirty` can be a time, a price, or an identifier. A format that is valid
in one sentence can be wrong in another. If the wrong interpretation reaches a
tool call, lookup and transaction failures follow.

Premove ITN generates valid alternatives, scores them in sentence context, and
uses exact decoding to select compatible edits. The frozen benchmark measures
this behavior on synthetic voice-agent stress cases. It is not a production-
traffic accuracy claim.

## Put ITN at the right boundary

Normalize transcript text before converting it to a tool argument. Then apply
the tool's own validation rules. ITN can reduce formatting ambiguity, but it
does not confirm that the ASR transcript is correct or that the user authorized
a transaction.

Continue with [Getting started](/docs/getting-started), review
[supported forms](/docs/supported-forms), or inspect the
[benchmark method and limits](/benchmarks).
