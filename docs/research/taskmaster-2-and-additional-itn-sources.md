# Taskmaster-2 and additional ITN sources

## Decision

Audit **Taskmaster-2** and **LibriTTS** next. Taskmaster-2 adds natural spoken
dialogue context around DATE, TIME, MONEY, CARDINAL, ORDINAL, and MEASUREMENT
forms. LibriTTS supplies direct written/normalized-text pairs that can be
reversed for ITN. Neither source is likely to solve PHONE or ELECTRONIC
coverage, so those kinds still need a separate source or controlled synthetic
data.

Do not add MASSIVE en-US because it is a localization of SLURP and maps each
record back to its original SLURP ID. Do not use STOP in a compiled dataset;
its license prohibits derivative works and incorporation into another dataset.
Keep TOPv2 and ATIS out of the current commercial-ready mixture because their
licenses are restrictive.

## Taskmaster-2

Taskmaster-2 is the best immediate dialogue candidate. Pin the official
repository at commit `d92cb6af3005f1dc09c39e75e7daf4a04905e00b`.

- The [official README](https://github.com/google-research-datasets/Taskmaster/blob/d92cb6af3005f1dc09c39e75e7daf4a04905e00b/TM-2-2020/README.md)
  reports 17,289 spoken two-person Wizard-of-Oz dialogues in restaurants,
  food ordering, movies, hotels, flights, music, and sports. `USER` turns are
  speech transcripts. `ASSISTANT` turns were written and played through TTS;
  they are not parallel written targets for the user speech.
- User utterances contain character-indexed semantic spans. Two workers
  annotated each conversation, and both annotations are retained. The
  [official ontologies](https://github.com/google-research-datasets/Taskmaster/tree/d92cb6af3005f1dc09c39e75e7daf4a04905e00b/TM-2-2020/ontology)
  include dates, times, price and fare fields, ticket and guest counts,
  ratings, durations, scores, luggage, stops, and flight numbers.
- A local source audit found 17,304 JSON dialogues, 157,516 `USER` turns,
  75,696 annotated `USER` turns, and 108,480 distinct annotated spans. The
  source-file count differs by 15 from the README. A compiler must record this
  discrepancy and derive counts from the pinned files.
- The source has no official train/dev/test partition. Split by complete
  dialogue and preferably by `instruction_id`, then deduplicate against all
  existing sources and evaluations. An utterance-level random split would leak
  repeated scenarios and entities.
- The README warns that transcripts can retain errors and that transcribers
  sometimes inserted, deleted, or replaced language. Treat spans as weak
  semantic boundaries, not as exact speech evidence.
- The repository releases Taskmaster-2 under
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Taskmaster-2 has no native spoken/written ITN pair. Compile only a complete,
non-overlapping annotated span for which the current Rust graph yields one
unambiguous realization. Require the final sentence to pass `build_gold_graph`.
Use unedited natural `USER` turns as KEEP only after excluding unresolved
constructible spans. This can improve contextual KEEP supervision without
inventing canonical labels.

Expected value by weak kind is high for DATE and TIME; moderate for MONEY,
CARDINAL, ORDINAL, and MEASUREMENT; low for DIGIT_SEQUENCE; and negligible for
PHONE and ELECTRONIC. These are source-level expectations. Record counts must
come from a compiler audit, not ontology names alone.

## LibriTTS

LibriTTS is the strongest new direct-pair candidate.

- The [official OpenSLR release](https://www.openslr.org/60/) provides about
  585 hours of read English speech under CC BY 4.0. It explicitly includes
  both original and normalized text and preserves clean/other train, dev, and
  test subsets.
- The [official paper](https://www.isca-archive.org/interspeech_2019/zen19_interspeech.pdf)
  says the normalized text was produced with Google's text-normalization
  engine. The release contains 2,456 speakers. The published split metadata
  totals 369,086 utterances.
- Reverse `text_normalized -> text_original` to form a native ITN pair, but
  retain only examples that the current Rust candidate graph can construct.
  The source may include punctuation, capitalization, abbreviation, and book
  editing changes outside this project's realization boundary.
- Keep the official speaker/chapter partitions. Deduplicate against Google TN,
  Golden, and all dialogue sources. Book passages and neighboring sentences
  can otherwise create near-duplicate leakage.

LibriTTS should be audited first for CARDINAL, ORDINAL, DATE, MONEY,
MEASUREMENT, and ambiguous KEEP. Its read-book domain is a distribution shift
from conversational requests. PHONE, ELECTRONIC, and command-style TIME will
probably remain sparse.

## Other candidates

| Source | Size and supervision | Decision |
| --- | --- | --- |
| [MASSIVE](https://github.com/alexa/massive/tree/f966f21846043aabef9b0f974fa7970027f43738) | More than one million localized utterances across 52 languages, 60 intents, and 55 slot types; official train/dev/test field; CC BY 4.0. The official README says it was created by localizing SLURP and that each ID maps to the original SLURP record. | Reject en-US as duplicate SLURP data. Other languages do not serve the current English model. |
| [STOP](https://github.com/facebookresearch/spoken_task_oriented_parsing) | More than 200,000 recordings from over 800 speakers, based on 125,000 TOPv2 text/parse pairs across eight domains. | Reject for compilation. The [official license](https://github.com/facebookresearch/spoken_task_oriented_parsing/blob/main/LICENSE) prohibits derivative works and incorporation into another dataset, and permits only limited audio redistribution. |
| [TOPv2](https://aclanthology.org/2020.emnlp-main.413/) | About 180,000 crowdsourced text queries across alarm, event, messaging, music, navigation, reminder, timer, and weather, with semantic parses and official domain splits. No speech transcript/written pair. | Useful mainly for TIME and contextual KEEP research, but its CC BY-NC 4.0 release is not suitable for the commercial-ready mixture. |
| [ATIS](https://catalog.ldc.upenn.edu/LDC2019T04) | 5,871 microphone-speech utterances with 4,978 train and 893 test records; entity annotations include dates. No native ITN pair. | Reject for the open mixture. The official LDC release requires an LDC user agreement and access fee. It is also small, old, and flight-domain narrow. |
| [Taskmaster-3](https://github.com/google-research-datasets/Taskmaster/blob/d92cb6af3005f1dc09c39e75e7daf4a04905e00b/TM-3-2020/README.md) | 23,789 movie-ticket dialogues with time, date, ticket-count, price, duration, and rating spans; CC BY 4.0. | Lower priority. It is entirely written self-dialogue, so it adds weak slot context but no spoken evidence or native ITN pairs. |
| [PolyAI EVI](https://github.com/PolyAI-LDN/evi-paper) | 5,506 real spoken human-to-machine dialogues across British English, Polish, and French; English has 1,407 dialogues, 12,663 turns, and 1,081 speakers. ASR hypotheses link to synthetic profiles containing names, dates of birth, and postcodes; CC BY 4.0. | Reserve as an independent DATE and alphanumeric DIGIT_SEQUENCE evaluation. The [official release](https://huggingface.co/datasets/PolyAI/evi) has only a `test` split, and profile linking is not token-aligned ITN supervision. |
| [PolyAI task-specific datasets](https://github.com/PolyAI-LDN/task-specific-datasets) | Restaurant8k and smaller DSTC8-derived sets have span slot annotations under CC BY 4.0, but no canonical ITN values. | Audit Restaurant8k only after Taskmaster-2. Exclude or exactly deduplicate the DSTC8 subsets because they can overlap the prepared SGD source. |

## Recommended order

1. Build a read-only Taskmaster-2 audit. Report conservative yield by kind,
   duplicate-annotation agreement, exclusion reason, and proposed grouped
   splits before publishing records.
2. Download only LibriTTS metadata/text first. Measure direct reversible pair
   yield by kind before downloading about 80 GB of audio, which this text model
   does not need.
3. Add neither source to training until its held-out partition is frozen and
   checked for exact and normalized overlap with Golden and current datasets.
4. Freeze EVI as evaluation-only unless there is an explicit decision to
   forfeit that independent benchmark.
5. After these audits, search specifically for permissively licensed PHONE and
   ELECTRONIC pairs. The sources above do not close those two gaps.
