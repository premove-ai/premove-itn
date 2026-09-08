# Production model record

## What is selected

The production model is the final structured-value 20k checkpoint based on
`microsoft/deberta-v3-large`. It is the only retained checkpoint.

The model saw 418,000 training examples in three stages:

| Stage | Examples |
| --- | ---: |
| Google text normalization | 378,000 |
| Conversational adaptation | 20,000 |
| Structured-value adaptation | 20,000 |
| **Total** | **418,000** |

The frozen VoiceAgent ITN dataset was not used for training or model selection.

## Training sources

The base stage used the Google Text Normalization training partition.

The conversational stage combined Google positive replay with conversational
examples derived from:

- SLURP real textual annotations
- Schema-Guided Dialogue
- SpokenWOZ
- Taskmaster-1 spoken data

The final structured-value stage contained:

| Training bucket | Examples |
| --- | ---: |
| Candidate-bearing KEEP | 4,000 |
| Conversational positive replay | 2,850 |
| Google positive replay | 3,000 |
| Targeted electronic values | 3,000 |
| Targeted hard KEEP | 150 |
| Targeted identifiers | 4,000 |
| Targeted numeric IDs | 750 |
| Targeted phone values | 2,250 |
| **Total** | **20,000** |

## Training kind distribution

The table counts every example seen by the selected checkpoint.

| Kind | Google stage | Conversational stage | Structured stage | Total |
| --- | ---: | ---: | ---: | ---: |
| CARDINAL | 32,472 | 853 | 876 | 34,201 |
| DATE | 78,643 | 315 | 398 | 79,356 |
| DECIMAL | 12,033 | 465 | 701 | 13,199 |
| DIGIT_SEQUENCE | 3,258 | 398 | 674 | 4,330 |
| ELECTRONIC | 0 | 12 | 3,030 | 3,042 |
| KEEP | 100,000 | 11,429 | 4,150 | 115,579 |
| MEASUREMENT | 16,647 | 300 | 349 | 17,296 |
| MONEY | 23,688 | 258 | 271 | 24,217 |
| MULTI | 82,490 | 3,259 | 2,078 | 87,827 |
| ORDINAL | 21,484 | 258 | 271 | 22,013 |
| PHONE | 2,761 | 260 | 1,994 | 5,015 |
| PUNCTUATION | 624 | 258 | 272 | 1,154 |
| TIME | 3,894 | 1,879 | 876 | 6,649 |
| WHITELIST | 6 | 17 | 17 | 40 |
| WORD | 0 | 39 | 4,043 | 4,082 |
| **Total** | **378,000** | **20,000** | **20,000** | **418,000** |

## Training configuration

The Google stage used AdamW, learning rate `2e-5`, weight decay `0.01`, batch
size `8`, local length bucketing, and a deterministic suffix shuffle.

Both adaptation stages used AdamW, learning rate `5e-6`, weight decay `0.01`,
batch size `8`, one epoch, and a fresh optimizer state. The structured stage
used microbatch size `2`. Validation and test rows were not used for training.

## Why this checkpoint was selected

The final checkpoint was selected because it gave the best product-relevant
balance for structured values and conversational normalization while retaining
strong general ITN accuracy. It improved electronic values, identifiers,
numeric IDs, phone values, conversational validation, and the targeted
validation set.

The final recorded development results were:

| Evaluation | Correct | Total | Accuracy |
| --- | ---: | ---: | ---: |
| Targeted structured validation | 1,465 | 1,485 | 98.65% |
| Google validation | 37,698 | 39,543 | 95.33% |
| Conversational validation | 9,832 | 10,121 | 97.14% |
| NVIDIA Numb3rs semantic | 7,637 | 10,131 | 75.38% |

The Numb3rs result was below its predeclared regression floor of 76.39%. The
checkpoint was accepted as a documented promotion exception because its gains
on the product-relevant structured and conversational tasks were larger. This
tradeoff must remain visible. The development results must not be presented as
the final blind VoiceAgent benchmark result.

## Benchmark separation and current limits

Before deleting the training rows, all 418,000 exposed training inputs were
compared with all 1,500 frozen VoiceAgent inputs. There were no exact matches
and no matches after lowercasing and removing punctuation. Token n-gram and
embedding-similarity contamination checks were not completed.

The VoiceAgent dataset passed a full independent mechanical and semantic audit,
but blind human adjudication remains pending. First Evaluation has run on the
frozen dataset. Its original raw outputs are preserved under
`eval/voice_agent_itn/results/20260908T172500Z/`. The original native Rust
extension was a development build, so its latency is not the production
comparison. A release measurement and exact output equivalence audit are
recorded under `eval/voice_agent_itn/results/first-evaluation/`.

The bulk training corpora and historical raw evaluation outputs were deleted.
The repository can load and verify the selected checkpoint, but it cannot
reproduce the full training run without reacquiring and rebuilding the source
datasets. Dataset licenses and provider terms must be checked before any source
corpus is reacquired or redistributed.
