---
title: What is inverse text normalization?
description: Learn how ITN turns spoken-form ASR transcripts into structured written text.
---

Inverse text normalization converts spoken-form text produced by speech
recognition into the written forms that people and software expect.

```text
twenty dollars              → $20
four thirty                 → 04:30
support at example dot com  → support@example.com
```

## ITN and text normalization

Text normalization prepares written text for speech synthesis. It can turn
`$20` into `twenty dollars`. Inverse text normalization handles the opposite
direction after speech recognition.

```text
speech → ASR → spoken-form transcript → ITN → written transcript
```

## Why rules are useful

Deterministic rules are good at validating structure. They can ensure that a
date is possible, preserve leading zeroes in an identifier, or render a phone
number consistently.

## Why context matters

The same words can represent different values:

```text
one oh five → 01:05 (time candidate) or 105 (identifier candidate)
```

A rule can generate both interpretations. Sentence context is needed to choose
the intended one. In the retained room-code example, Premove ITN outputs
`the room code is 105`. The [benchmark](/itn/benchmarks) also records ambiguity
errors, so this is not a guarantee that every contrast is resolved correctly.

## Contextual ITN

Premove ITN combines deterministic candidate generation with contextual
ranking. Rust generates valid written forms, DeBERTa scores them in the full
sentence, and an exact decoder selects compatible edits.

## Why it matters after ASR

ASR transcribes what was said. ITN decides how a spoken value should be written.
For a person, `zero eight two zero six three` is readable. For a database key,
preserving the leading zero and producing `082063` can be necessary for a
successful lookup. The correct formatting also depends on what the sentence
means. A time, an amount, and a room number can share spoken words.

ITN does not repair transcription errors or infer a value that the speaker did
not say. It also does not replace application-side validation. A voice agent
should still check values before a consequential API action.

Read [why voice agents need ITN](/itn/docs/learn/itn-for-voice-agents),
[how Premove ITN works](/itn/docs/how-it-works), or
[install the package](/itn/docs/getting-started).
