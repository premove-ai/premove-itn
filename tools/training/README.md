# Premove ITN model training

This private project trains Model V1 as a 21-label contextual BIO tagger. The
baseline is `microsoft/deberta-v3-large` at the pinned revision
`64a8c8eab3e352a784c658aef62be1662607476f`.

The training module verifies the frozen split manifest and both artifact
checksums before it loads records. It aligns labels through the tokenizer's
`word_ids()` mapping, supervises only each source word's first subword, and
fails if any record would be truncated. The DeBERTa tokenizer audit selected
`max_length=72`: train has a maximum of 72 subwords and validation has a maximum
of 63. The pinned fast tokenizer produces zero unknown subwords across both
partitions.

Audit the frozen data and tokenizer without loading model weights:

```bash
uv run --package premove-itn-training premove-train-model-v1 \
  data/generated/dataset_v1/split/manifest.json \
  models/contextual-bio-v1 \
  --audit-only
```

Run the first baseline:

```bash
uv run --package premove-itn-training premove-train-model-v1 \
  data/generated/dataset_v1/split/manifest.json \
  models/contextual-bio-v1
```

The best checkpoint is selected by strict validation span micro-F1. The run
also reports per-kind span metrics, exact word-level BIO accuracy,
context-only exact accuracy, and multi-span exact accuracy. `run.json` records
dataset hashes, revisions, labels, hyperparameters, the git state, and final
metrics. Model outputs remain ignored by Git.

`data/golden.json` is not loaded by this project. It remains frozen for final
hard evaluation after model selection.
