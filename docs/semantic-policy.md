# Model V1 semantic policy

Policy version: 1  
Frozen: 2026-08-26, before the first Golden V1 model evaluation.

This document is the source of truth for normalization eligibility and Model V1
class/span semantics. It defines what Premove considers normalizable. It does
not define tokenizer behavior, numeric label IDs, realization algorithms, or
business-tool bindings.

## Core rule

Premove normalizes an expression when it denotes a literal numeric, ordinal,
temporal, monetary, measurement, telephone, identifier, or electronic value.
It leaves number-like language unchanged when the language is lexicalized,
idiomatic, or does not denote an independent value.

Tool relevance does not determine normalization. A literal value is normalized
even when no current downstream tool consumes it. Entity extraction, tool
selection, and argument binding decide whether the normalized value is useful.

For example, all of these contain normalizable literal values:

```text
Alexander was twenty three.       -> Alexander was 23.
Season five starts tomorrow.      -> Season 5 starts tomorrow.
Objective one is complete.        -> Objective 1 is complete.
Liverpool finished second.        -> Liverpool finished 2nd.
She placed sixth.                 -> She placed 6th.
He is in eleventh grade.          -> He is in 11th grade.
```

Their relevance to a particular tool call is intentionally outside ITN.

## Normalization eligibility

Normalize literal values in these uses:

- quantities and counts, such as `three tickets`;
- ages, such as `twenty three years old`;
- numbered entities, including seasons, series, objectives, platforms, gates,
  routes, rooms, books, chapters, versions, models, and levels;
- rankings and ordinal positions, such as `finished second` or `first prize`;
- grades and generations, such as `eleventh grade` or `fourth generation`;
- calendar dates, years, decades, and historical date expressions;
- clock times and explicit meridiem expressions;
- monetary amounts;
- decimal values;
- measurements with units;
- telephone numbers and extensions;
- codes, references, PINs, OTPs, account suffixes, and other digit sequences;
- email addresses, URLs, and other supported electronic values.

Machine-readable value expression takes priority over editorial prose style.
For example, a literal generation or edition can use `4th` or `2nd` even when
published prose might prefer the words `fourth` or `second`.

## Lexical and idiomatic exclusions

Do not normalize a number word that is part of a fixed lexical or idiomatic
construction and does not denote an independent value. Examples include:

```text
one another
one of a kind
no one
one way or another
at first
first aid
first and foremost
second nature
second thoughts
```

`O` means that a source word is not part of an expression eligible for ITN. It
does not mean that a literal value is irrelevant to the current application.

Context decides whether a word is literal or lexical:

```text
She finished second.   -> `second` is ORDINAL.
Give me a second.      -> `a second` is lexical and remains O.

He won first prize.    -> `first` is ORDINAL.
First aid is nearby.   -> `first aid` is lexical and remains O.
```

## Class semantics

### `CARDINAL`

Use `CARDINAL` for a literal numeric value expressed as arithmetic quantity or
number. This includes counts, ages, and the numeric component of a numbered
entity when the digits are read as a number.

```text
three tickets       -> 3 tickets
twenty three years  -> 23 years
season five         -> season 5
objective one       -> objective 1
top ten finish      -> top 10 finish
```

### `ORDINAL`

Use `ORDINAL` when the expression denotes rank, order, level, grade,
generation, edition, or another ordinal position.

```text
finished second      -> finished 2nd
first prize          -> 1st prize
eleventh grade       -> 11th grade
fourth generation    -> 4th generation
second edition       -> 2nd edition
```

Lexical uses such as `at first`, `first aid`, and `second thoughts` remain
`O`.

### `DECIMAL`

Use `DECIMAL` for a literal decimal value when no enclosing class owns the
value and its semantic unit.

```text
one point five -> 1.5
```

A decimal inside an amount or measurement belongs to `MONEY` or
`MEASUREMENT`, respectively.

### `MEASUREMENT`

Use `MEASUREMENT` for a value together with its measurement unit. The span owns
the complete realizable number-plus-unit expression.

```text
forty meters                  -> one MEASUREMENT span
eighteen point five kilometers -> one MEASUREMENT span
two hundred kilometers per hour -> one MEASUREMENT span
```

Do not label only the number as `CARDINAL` or `DECIMAL` when the adjacent unit
is part of the same realizable measurement.

### `MONEY`

Use `MONEY` for the complete realizable amount and currency expression.

```text
twenty dollars                    -> one MONEY span
two dollars and thirty cents      -> one MONEY span
```

Do not split the numeric component into `CARDINAL` or `DECIMAL` when `MONEY`
owns the complete amount.

### `DATE`

Use `DATE` for the complete realizable calendar expression, including all
spoken components that jointly identify one date.

```text
january twenty second twenty sixteen -> one DATE span
```

Do not split one calendar expression into separate `CARDINAL`, `ORDINAL`, or
`DATE` spans. Distinct coordinated dates remain distinct values unless a
single deterministic realizer explicitly supports the complete coordination.

### `TIME`

Use `TIME` for a clock value and its explicit meridiem or supported timezone
component.

```text
two thirty  -> TIME
seven a m   -> TIME
```

The same spoken digits in a room, code, quantity, or identifier context are
not `TIME`.

### `PHONE`

Use `PHONE` when the context denotes a telephone number or extension. Explicit
signals include `call`, `phone`, `telephone`, `dial`, `mobile`, `line`, and
`extension` when they govern the number.

### `DIGIT_SEQUENCE`

Use `DIGIT_SEQUENCE` for a literal identifier whose digits must preserve order
and leading zeros. Examples include verification codes, PINs, OTPs, booking
codes, reference numbers, account suffixes, and serial-like identifiers.

```text
verification code four seven nine
                  |-------------|
                   DIGIT_SEQUENCE
```

Context words such as `verification code` identify the class but are not part
of the span.

### `ELECTRONIC`

Use `ELECTRONIC` for the complete supported email address, URL, domain, or
other electronic expression. Spoken separators such as `at`, `dot`, and
`slash` belong to the span when the deterministic electronic realizer consumes
them.

## Span ownership

A span is the smallest complete source expression that one selected
deterministic realizer can consume without borrowing words from another span.

These rules are mandatory:

1. A semantic owner includes all intrinsic value components it needs.
2. `MEASUREMENT` owns its number and unit.
3. `MONEY` owns its amount and currency.
4. `DATE` owns all components of one calendar expression.
5. `TIME` owns supported meridiem or timezone components.
6. `ELECTRONIC` owns spoken separators consumed by its realizer.
7. `PHONE` and `DIGIT_SEQUENCE` own the spoken number, not contextual cue
   words such as `call`, `PIN`, or `reference`.
8. Model V1 spans do not overlap.
9. Context words outside a value remain `O`.

When a larger class owns an expression, do not emit nested classes. For
example, `eighteen point five kilometers` is one `MEASUREMENT`, not a nested
`DECIMAL` plus a unit.

## Ambiguity procedure

Apply these questions in order:

1. Does the expression denote an independent literal value in this sentence?
2. Is it instead a fixed lexical or idiomatic construction?
3. Which one Model V1 class owns the complete realizable expression?
4. What is the smallest complete span accepted by that class's deterministic
   realizer?

Do not consider available tools, tool arguments, or the current application's
interest in the value.

If the expression remains genuinely unresolved, preserve it as `O` and record
the case for the next policy version. Do not guess a class merely because the
source contains a number word.

## Versioning and frozen artifacts

This policy was frozen after Dataset V1 validation exposed differences between
Google written/spoken conventions and Premove's semantic ontology, and before
the first Golden V1 model evaluation.

Dataset V1 remains immutable. A Dataset V1 label that conflicts with this
policy remains part of the historical V1 experiment and is recorded as a
policy disagreement. Confirmed corrections and new supervision belong to
Dataset V2.

Golden V1 also remains immutable. If a frozen Golden annotation conflicts with
this policy, report the frozen metric and a separate policy adjudication. Never
edit Golden after observing model predictions.

A future change to eligibility or class/span ownership requires a new policy
version and an explicit benchmark audit.
