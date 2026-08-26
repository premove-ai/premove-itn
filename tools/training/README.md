# Premove ITN model training

This private project trains Model V1 as a 21-label contextual BIO tagger. It
consumes the runtime-owned, versioned `ModelV1LabelContract`, so training and
future inference share one numeric label meaning. The baseline is
`microsoft/deberta-v3-small` at the pinned revision
`a36c739020e01763fe789b4b85e2df55d6180012`.

The training module verifies the frozen split manifest and both artifact
checksums before it loads records. It aligns labels through the tokenizer's
`word_ids()` mapping, supervises only each source word's first subword, and
fails if any record would be truncated. The DeBERTa tokenizer audit selected
`max_length=72`: train has a maximum of 72 subwords and validation has a maximum
of 63. The pinned fast tokenizer produces zero unknown subwords across both
partitions.

The default run uses BF16 mixed precision and length-grouped batches. It
evaluates and saves every 250 optimizer steps, with early stopping after three
non-improving validation checks. Cache clearing is disabled initially because
it reduces throughput; enable it only if a Small run shows MPS memory growth.

Audit the frozen data and tokenizer without loading model weights:

```bash
uv run --package premove-itn-training premove-train-model-v1 \
  data/generated/dataset_v1/split/manifest.json \
  models/contextual-bio-v1 \
  --audit-only
```

Run the first Small baseline:

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
