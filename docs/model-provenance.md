# Production model provenance

## Frozen checkpoint

- Path: `data/models/structured_value_selected_20k/checkpoint.pt`
- SHA-256: `9021fa11a028faefb31ef67878170cbe29ed25e68a9a78999f37b120c2ad00d5`
- Size: 5,227,643,767 bytes
- Model: `microsoft/deberta-v3-large`
- Model revision: `64a8c8eab3e352a784c658aef62be1662607476f`
- Total training exposure: 418,000 examples
- Final run fingerprint: `96c110801753166276df98ca9d65406f161009b8e981c03ca8cca94a9fb489fe`
- Repository HEAD recorded at finalization:
  `c4605d68539d9432311e5893c9652b319d745618`

The training worktree contained uncommitted changes. Therefore this hash must
not be presented as a complete source snapshot of the training run. The frozen
run fingerprint and schedule hashes are the authoritative run identities.

The checkpoint is the only retained model checkpoint. Loading uses the model
name and immutable revision above. `data/models/production.json` is the runtime
pointer.

## Training stages

| Stage | Exposures | Source schedule SHA-256 |
| --- | ---: | --- |
| Google TN base | 378,000 | Training record IDs: `80cec20da44e0adb024e6ae55b59317daafdac5daf7ad915248371b95510d26f` |
| Conversational adaptation | 20,000 | `764001955fe4c823cb7f3f953b24083b48daab705040bcac86a4388c77f31f89` |
| Structured-value adaptation | 20,000 | `cb86ab87b1dcb6ece4f79235b968f68f08e39ddb999a48898a825ecc22c2efd4` |

The final adaptation used AdamW, learning rate `5e-6`, weight decay `0.01`,
batch size `8`, microbatch size `2`, one epoch, a fresh optimizer state, and no
validation or test rows in training.

## Kind distribution by stage

| Kind | Google 378k | Conversational 20k | Structured 20k | Total |
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

Counts were reconstructed from the exact Google training sequence and the
first 20,000 records of each immutable adaptation schedule before bulk data
deletion.

## Frozen benchmark separation

The VoiceAgent ITN dataset is evaluation-only. It was not used for training or
model selection. Before bulk training data deletion, all 418,000 exposed input
texts were compared with all 1,500 frozen benchmark inputs. Exact overlap was
zero. Lowercase alphanumeric-normalized overlap was also zero. This check did
not include token n-gram or embedding-similarity analysis.

## Deleted artifacts

The repository cleanup removed generated and external training corpora,
training schedules, intermediate and predecessor checkpoints, cached datasets,
checkpoint evaluation outputs, logs, and exploratory training/evaluation
documents. These artifacts are not required to load the frozen checkpoint.
