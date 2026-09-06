# Post-P0 candidate coverage

Evaluation date: 2026-09-06

## Conclusion

The four launch-scope Rust stages added 181 exactly reachable VoiceCodeBench
entities without losing any previously reachable entity. Candidate reachability
rose from 711/1,482 to 892/1,482. The largest intended changes reached PHONE
59/60 and version 29/30.

This is a candidate-graph audit. It does not score contextual checkpoint
selection and does not open a sealed test set. VoiceCodeBench remains an
external entity-level component diagnostic, not an official raw-audio score.

## Method

The audit uses VoiceCodeBench revision
`3ccea73877a159eb2a8b17304148c325c5fe5061` and compares each current
`target_is_reachable` result with the preserved pre-P0 reachability cache. The
same 1,482 ordered entity IDs occur in both runs. Current reachability was
computed directly instead of through the cache because the old cache did not
include the Rust implementation revision.

PolyNorm uses the pinned en-US set at revision
`f3c67e047bea6b7c40bc2466c0fdaad51d8ce67d`. Its exact candidate reachability
rose from 161/540 to 162/540.

## VoiceCodeBench result

| Category | Before | After | Change |
| --- | ---: | ---: | ---: |
| All entities | 711/1,482 | 892/1,482 | +181 |
| PHONE | 23/60 | 59/60 | +36 |
| Version | 0/30 | 29/30 | +29 |
| Account or record number | 52/65 | 63/65 | +11 |
| Product code | 77/90 | 89/90 | +12 |
| Reference ID | 119/150 | 144/150 | +25 |
| CLI flag | 0/44 | 17/44 | +17 |
| Environment variable | 0/35 | 28/35 | +28 |
| Code symbol | 0/35 | 16/35 | +16 |
| Command | 1/50 | 8/50 | +7 |

No category lost an exactly reachable entity.

## Remaining launch-category misses

The single PHONE miss is `seven oh seven dash five five dash zero one three
four`, whose target also contains only nine digits. It is not evidence for
another general PHONE grammar change.

The single version miss is `API dash 2027 dash 12 dash 24`. This is a named API
revision, not the `v`-prefixed multipart form added in the P0 stage.

Most remaining CLI and code misses require compound hyphenated flags, full
command spans, camel/Pascal case, corrections such as `not dash p`, or
benchmark-specific URL path capitalization. These are outside the narrow P0
examples and should not be added as one undifferentiated grammar expansion.

The ELECTRONIC `O` change has no exact before/after gain in this benchmark. Its
value is the focused deterministic regression: it prevents `S U P P O R T`
from becoming `SUPP0RT` while retaining numeric `O` behavior.

## Cache correction

VoiceCodeBench reachability depends on Rust realization behavior. The evaluator
now includes the SHA-256 of `rust/src/lib.rs` in the cache key. A Rust change
therefore invalidates old graph results instead of silently reusing them.
