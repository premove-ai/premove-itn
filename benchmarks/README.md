# Reproduce First Evaluation

This directory contains optional comparison tools for Premove ITN,
text-processing-rs, and Thutmose. They support the published First Evaluation.
The normal library installation does not require benchmark dependencies.

## Protocol

Use the frozen 1,500-row VoiceAgent dataset and the selected production
checkpoint. The adapters receive only input text. Scoring runs after inference.
Run each backend sequentially, with one request at a time. Close each model
before loading the next backend. Avoid other compute workloads during timing.

The runner uses 15 warm-up calls per backend. It saves their individual latency
and excludes them from measured records. Thutmose also performs its historical
startup warm-up, recorded in runtime metadata. The main latency boundary covers
the complete adapter call, including Thutmose worker communication. Neural
inference completes on MPS before timing ends. Percentiles use linear
interpolation. Reciprocal mean latency is sequential throughput, not a load test.

## Release environment

Build and install a release wheel into a separate environment. Do not use an
editable installation for the published measurement.

```sh
uv run maturin build --release --manifest-path rust/Cargo.toml --out dist
uv venv .venv-benchmark --python 3.11
uv pip install --python .venv-benchmark/bin/python dist/<wheel-name>.whl \
  torch==2.13.0 'transformers[sentencepiece]==5.16.1'
RAYON_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false \
  .venv-benchmark/bin/python benchmarks/run_comparison.py \
  --output /path/to/new-results \
  --thutmose-python /path/to/thutmose-environment/bin/python \
  --thutmose-artifact /path/to/itn_en_thutmose_bert.nemo
```

Replace placeholder paths before running. The selected Premove checkpoint must
be available at the path in `data/models/production.json`. Model weights and
caches are not committed. The runner refuses a Rust extension whose profile is
not `release`, or which has debug assertions enabled. It records native build
metadata and the actual imported extension path.
Before loading any backend, it also verifies the dataset SHA-256 from the frozen
benchmark manifest, the checkpoint SHA-256 from `data/models/production.json`,
the model name and revision, and the imported extension's bytes. Those values
are written to `run.json`; a mismatch aborts the run before inference starts.

For a released inference-only model folder, pass
`--premove-artifact path/to/premove-itn-v0.1.0`. The backend then
uses `premove_itn.inference_artifact.load_inference_artifact`, which verifies the model digest and
loads only the frozen state dict. The checkpoint remains required for the
benchmark provenance gate.

Thutmose uses the NVIDIA `itn_en_thutmose_bert` artifact in a persistent isolated
worker. This is a weight-compatible BERT loader for the older NeMo artifact.
Create its environment with Python 3.12 and install
`benchmarks/thutmose-requirements.txt`. Obtain the artifact from NVIDIA's model
catalog and follow its license terms. The recorded run used Python 3.12.13,
PyTorch 2.13.0, Transformers 4.40.2, Tokenizers 0.19.1, and NumPy 2.4.6.
NeMo 2.0.0 was present in that environment but is not imported by this worker.
It does not call the current NeMo public model API. The worker uses the archived
inference procedure, 128 wordpiece padding, tag heads, and detokenization rules.
This limitation must accompany comparisons with NVIDIA's implementation. The
exact environment versions used for a published run belong with that run.

## Evidence and regression data

To rebuild the report and prediction audit from stored results, without
running any model:

```sh
.venv-benchmark/bin/python benchmarks/summarize_comparison.py \
  eval/voice_agent_itn/results/first-evaluation
```

First Evaluation contains the repository's retained accuracy and release
latency evidence. It was run after exploratory work and is not a blind run.
Do not tune the model, candidates, or scoring rules on this dataset.

`verify_equivalence.py` performs an explicit audit of all frozen inputs against
the pre-batch candidate builder. Run this only when an equivalence audit is
needed, and retain its result.

`tests/fixtures/normalization_regression.json` is for normalization regression
and latency development tests. It is not training data or a quality benchmark.
Local exploratory latency results under `eval/golden_latency/` stay ignored.
