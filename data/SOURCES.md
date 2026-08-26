# Dataset sources

This file is the source of truth for external datasets used by the dataset
compiler. Raw datasets are not committed to this repository.

## Google Text Normalization

- Upstream: [rwsproat/text-normalization-data](https://github.com/rwsproat/text-normalization-data)
- Paper: [RNN approaches to text normalization: A challenge](https://arxiv.org/abs/1611.00068)
- License: Creative Commons Attribution-ShareAlike 4.0, as distributed with
  the dataset
- Language used: English
- Expected local file:
  `data/external/google_tn/en_with_types_merged.tsv`
- Local byte size: `20,946,258,561`

The local TSV is a byte-for-byte concatenation of the 100 English
`output-*-of-00100` shards in numeric order. The merge does not parse, rewrite,
or normalize any record. Normal records keep the upstream three-column TSV
format. Sentence boundaries remain exact `<eos>\t<eos>` rows.

The project parser maps only supported source classes. It quarantines known
unsupported classes and fails closed on unknown classes. Derived records must
also pass the deterministic production realizer before they can become
training data.

## Schema-Guided Dialogue

- Upstream: [google-research-datasets/dstc8-schema-guided-dialogue](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue)
- Paper: [Towards Scalable Multi-domain Conversational Agents: The Schema-Guided Dialogue Dataset](https://arxiv.org/abs/1909.05855)
- License: Creative Commons Attribution-ShareAlike 4.0, as distributed in the
  local dataset
- Expected local directory: `data/external/sgd/`

The original train, dev, and test layout is retained. This dataset supplies
natural dialogue context. It does not supply ITN labels or ground-truth
normalization records.

## Repository policy

- `data/golden.json` is evaluation-only and must never be used for training.
- `data/deferred.json` is reviewed future-class evaluation data and must never
  be used for training.
- `data/external/` and `data/generated/` remain ignored by Git.
- Raw source records remain untouched.
- Derived labels are deterministic. An LLM must not assign ground-truth
  labels.
- Any distributable derived artifact requires a separate license and
  provenance review.
