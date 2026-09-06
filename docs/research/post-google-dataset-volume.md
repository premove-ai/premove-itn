# Post-Google training-data volume

## Conclusion

The four prepared conversational sources contain **206,649 compiled rows** in
their standalone training-ready artifacts. Their recorded splits contain
**186,173 train**, **10,125 validation**, and **10,351 test** rows. Only the
train rows are additional training data. None of these rows is part of the
active full-Google run.

Of the 206,649 compiled rows, **13,932 contain a non-KEEP ITN edit** and
**192,717 are KEEP contexts**. The sources do not provide native
Google-TN-style spoken/written pairs. Their compilers create weak targets only
when current Rust realizers can construct the annotated edit. See the
[dataset registry](../datasets.md#prepared-conversational-sources) and each
compiler's quality gates:
[SGD](../../scripts/build_sgd_dataset.py),
[SLURP](../../scripts/build_slurp_dataset.py),
[SpokenWOZ](../../scripts/build_spokenwoz_dataset.py), and
[Taskmaster-1](../../scripts/build_taskmaster_1_dataset.py).

## Exact current counts

| Source | Raw source volume in scope | Compiled rows | Non-KEEP rows | KEEP rows | Train / validation / test |
| --- | ---: | ---: | ---: | ---: | ---: |
| Schema-Guided Dialogue | 463,284 turns | 113,784 | 2,672 | 111,112 | 102,570 / 5,555 / 5,659 |
| SLURP real text | 16,521 rows | 16,431 | 920 | 15,511 | 14,769 / 814 / 848 |
| SpokenWOZ | 202,950 turns; 101,475 user turns | 74,012 | 7,918 | 66,094 | 66,642 / 3,644 / 3,726 |
| Taskmaster-1 spoken Wizard-of-Oz | 5,507 dialogues; 60,347 user utterances | 2,422 | 2,422 | 0 | 2,192 / 112 / 118 |
| **Standalone total** | Not additive across different source units | **206,649** | **13,932** | **192,717** | **186,173 / 10,125 / 10,351** |

The SGD, SLURP, and SpokenWOZ compiled and split counts come from their current
local manifests under `data/generated/` and
`data/generated/training_ready/`. The Taskmaster-1 count and all four split
counts are also recorded in the canonical
[dataset registry](../datasets.md#prepared-conversational-sources).

### Schema-Guided Dialogue

The pinned source has 463,284 turns. Its compiler found 345,477 eligible turn
occurrences before deduplication, rejected 117,807 turns, removed 125,016
duplicate occurrences, and applied per-kind caps to produce 113,784 rows. The
current artifact contains 2,672 non-KEEP rows: 1,882 TIME, 467 DECIMAL, 318
DIGIT_SEQUENCE, and 5 CARDINAL. The exact build evidence is
`data/generated/sgd/manifest.json`; the source and supervision limits are in
the [SGD source note](schema-guided-dialogue-source.md#full-build-result).

### SLURP

The three real textual source files contain 11,514 train, 2,033 development,
and 2,974 test rows, or 16,521 rows total. The compiler rejects 90 occurrences
and emits 16,431 unique rows. The current artifact contains 920 non-KEEP rows
when each output row is counted once. The source manifest's `distribution`
counts contained kind occurrences, so it sums above the row count for MULTI
records; `data/generated/training_ready/slurp/manifest.json` gives the
row-level primary-kind distribution. The separate 19,711-row synthetic file
is excluded by policy. See the [SLURP source note](slurp-dataset.md#decision).

### SpokenWOZ

The current compiler sees 202,950 turns and excludes 101,475 non-user turns.
After quality filtering and pair deduplication, it emits 74,012 rows. Of these,
7,918 have a non-KEEP primary kind. The exact current evidence is
`data/generated/spokenwoz/manifest.json` and
`data/generated/training_ready/spokenwoz/manifest.json`.

The [SpokenWOZ readiness note](spokenwoz-source-readiness.md#complete-build-result)
reports an older 74,979-row build. Do not use that number for the current
artifact. The old `data/generated/training-pool-partial/manifest.json` also
uses that older SpokenWOZ build and an older 16,435-row SLURP build.

### Taskmaster-1

The official Wizard-of-Oz source contains 5,507 dialogues and 60,347 USER
utterances. The compiler emits only utterances with an unambiguous annotated
ITN edit, so all 2,422 compiled rows are non-KEEP. The current local
`data/generated/training_ready/` directory does not contain the Taskmaster-1
artifact. Its exact compiled and split counts remain recorded in the
[dataset registry](../datasets.md#prepared-conversational-sources), and its
source boundary is defined in the
[Taskmaster-1 source note](taskmaster-1-source.md#supervision-boundary).

## Deduplication and licensing caveats

The 206,649 total is the sum of standalone corpora. It does not claim that all
rows remain unique in one combined pool. The three current local
training-ready artifacts for SGD, SLURP, and SpokenWOZ contain 204,227 row
occurrences but only **204,177 unique `(text, expected_text)` pairs**: 50
occurrences are cross-source duplicates. The exact four-source unique total
cannot be recomputed from the current workspace because the Taskmaster-1
artifact is absent. Run the
[training-pool builder](../../scripts/build_training_pool.py) after restoring
or rebuilding it; that builder also reassigns leakage-safe splits.

License terms also limit what "usable" means. SpokenWOZ's 74,012 rows are
CC BY-NC 4.0 and are restricted to noncommercial research. SGD's 113,784 rows
are CC BY-SA 4.0 and carry ShareAlike requirements. SLURP and Taskmaster-1 are
CC BY 4.0. The combined builder requires explicit noncommercial mode when a
restrictive source is present. See the
[dataset registry](../datasets.md#prepared-conversational-sources) for the
recorded licenses.

## Reproduction checks

The counts above were checked on 2026-08-31 with line counts over the pinned
source and generated JSONL files, row-level `kind` counts, manifest fields, and
sorted unique base64 encodings of `(text, expected_text)`. No training file,
checkpoint, or active-run state was changed.
