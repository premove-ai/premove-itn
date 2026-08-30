# Taskmaster-1 source decision

## Decision

Use only `USER` utterances from `TM-1-2019/woz-dialogs.json`. These are the
spoken turns in Taskmaster-1. Do not use `self-dialogs.json`, and do not treat
typed `ASSISTANT` turns as spoken data.

Pin the source repository at commit
`d92cb6af3005f1dc09c39e75e7daf4a04905e00b`. The SHA-256 of the selected
`woz-dialogs.json` file is
`cd3bc4e968487315d412c044d30af2bf0a4b33c3ef8b74c589f1e1fa832bf72f`.

## Primary-source findings

- The [official Taskmaster-1 README](https://github.com/google-research-datasets/Taskmaster/blob/d92cb6af3005f1dc09c39e75e7daf4a04905e00b/TM-1-2019/README.md)
  reports 13,215 dialogues: 5,507 spoken Wizard-of-Oz dialogues and 7,708
  written self-dialogues. It says that `USER` turns in the Wizard-of-Oz file
  are transcribed speech and `ASSISTANT` turns are written.
- The same README defines utterance `text`, character-indexed `segments`, and
  segment `annotations`. These fields let the compiler restrict weak
  canonicalization to annotated semantic spans.
- The same README licenses the work under
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Generated data
  must retain source attribution.
- The [official ontology](https://github.com/google-research-datasets/Taskmaster/blob/d92cb6af3005f1dc09c39e75e7daf4a04905e00b/TM-1-2019/ontology.json)
  identifies numeric, date, time, price, and duration slot families used by
  the conservative mapping.

## Supervision boundary

Taskmaster-1 does not provide paired spoken and written ITN text. The compiler
therefore does not invent free-form written targets. It only canonicalizes a
slot-annotated span when the existing Rust candidate graph provides one
unambiguous, non-overlapping realization of the slot's mapped `SpanKind`.
Every emitted full-sentence target must then pass `build_gold_graph`. Ambiguous,
invalid, unchanged, and unreachable turns are quarantined or excluded.
