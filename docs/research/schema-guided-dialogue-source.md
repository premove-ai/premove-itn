# Schema-Guided Dialogue source and ITN suitability

## Decision

Use the official Google Research Schema-Guided Dialogue (SGD) repository at
commit `e852981ae34990f4358979625854259302feaa78`. Keep the source clone and
compiled artifacts outside version control. Compile only turn-local slot
canonicalizations that the current Rust realizers can reproduce. Do not treat
all SGD canonical values as ITN supervision.

## Primary source

- Repository: [google-research-datasets/dstc8-schema-guided-dialogue](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue/tree/e852981ae34990f4358979625854259302feaa78)
- Pinned revision: `e852981ae34990f4358979625854259302feaa78`
- Local source: `data/external/schema-guided-dialogue/`
- Reproduction: `git clone https://github.com/google-research-datasets/dstc8-schema-guided-dialogue.git data/external/schema-guided-dialogue`
  followed by `git -C data/external/schema-guided-dialogue checkout e852981ae34990f4358979625854259302feaa78`.

The official README describes more than 20,000 annotated, multi-domain,
task-oriented conversations covering 20 domains. It says the dialogues were
generated with a dialogue simulator and paid crowd-workers. This is a text
dialogue corpus, not an audio corpus or a speech-transcript corpus. Its value to
this project is therefore task-oriented context plus narrow canonical-value
supervision, not direct spoken/written TN pairs. See the
[pinned official README](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue/blob/e852981ae34990f4358979625854259302feaa78/README.md).

## License

The official repository releases SGD and SGD-X under Creative Commons
Attribution-ShareAlike 4.0. The license permits sharing and adaptation subject
to attribution, change indication, a license reference, and ShareAlike for
adapted material. Compiled dataset artifacts must not be included in the
Apache-2.0 package without a separate distribution and licensing decision. See
the [pinned official license](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue/blob/e852981ae34990f4358979625854259302feaa78/LICENSE.txt)
and the repository's [license statement](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue/blob/e852981ae34990f4358979625854259302feaa78/README.md#license).

## Schema and stable partitions

The source provides `train/`, `dev/`, and `test/`. Each split contains
`schema.json` plus `dialogues_*.json` shards. A dialogue has `dialogue_id`,
`services`, and ordered `turns`. Each turn has a `speaker`, an `utterance`, and
service `frames`. A frame's non-categorical `slots` use half-open character
offsets into the utterance. Frame `actions` pair surface `values` with
same-length `canonical_values`. These facts and field meanings are defined in
the [official dialogue representation](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue/blob/e852981ae34990f4358979625854259302feaa78/README.md#dialogue-representation).

The compiler maps official `train` to `train`, `dev` to `validation`, and
`test` to `test`. It globally deduplicates `(text, expected_text)` and gives an
overlap to the earliest partition. This preserves the benchmark boundary while
preventing exact-pair leakage.

## Canonicalization boundary

SGD's `canonical_values` are service values. They include ITN-like time and
number forms, but also ISO dates dependent on dialogue context, entity aliases,
case fixes, abbreviations, and other database cleanup. The official README
explicitly defines them as values in canonicalized form “as used by the
service”; it does not claim they are written-form ITN labels.

The offline compiler therefore requires all of the following for a changed
span:

1. The slot offsets are structurally valid.
2. The spanned utterance text exactly equals one action surface value.
3. That surface value has one unambiguous canonical value in the frame.
4. A current Rust `SpanKind` realization is non-identity and exactly or
   semantically agrees with the service canonical value.
5. Replacements do not overlap and the complete result reaches `GoldGraph`.
6. The source and target pass the same conservative English quality filter as
   Google TN Dataset 1.

If any changed annotated span fails, the complete turn is quarantined. This
prevents unsupported canonicalizations from becoming false `KEEP` examples.

## Full build result

The pinned revision contains 463,284 turns. The build at the time of this note
produced 113,784 selected records after conservative quarantine, global
deduplication, and per-kind caps:

| Kind | Records |
| --- | ---: |
| KEEP | 111,112 |
| TIME | 1,882 |
| DECIMAL | 467 |
| DIGIT_SEQUENCE | 318 |
| CARDINAL | 5 |

The official partitions produced 101,983 train, 5,768 validation, and 6,033
test records. The compiler quarantined 117,807 turns; the dominant reason was
unsupported service canonicalization. The manifest records all input shard and
artifact SHA-256 hashes, rejection counts, provenance, quota results, and exact
distribution.

The result is intentionally sparse outside the five listed kinds. Zero yield
for another kind means SGD did not provide an independently verifiable example
under the current deterministic realizers. It must not be filled with guessed
labels.

## Commands

```bash
uv run python scripts/build_sgd_dataset.py \
  data/external/schema-guided-dialogue data/generated/sgd
uv run python scripts/audit_sgd_dataset.py \
  data/generated/sgd --sample-per-kind 100
```

Both directories are covered by the repository's existing `data/external/`
and `data/generated/` ignore rules.
