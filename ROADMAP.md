# Roadmap

Premove ITN is a contextual inverse text normalization system. Deterministic
Rust candidate generation, neural contextual scoring, and exact decoding form
one public product.

## Completed

- [x] Freeze and publish the inference-only Hugging Face model artifact.
- [x] Add the public `PremoveITN` Python API.
- [x] Make contextual dependencies part of the normal package installation.
- [x] Add the `premove-itn` command-line interface.
- [x] Prepare user-first GitHub and Hugging Face documentation.
- [x] Add automated CI and release gates for Python, Rust, wheels, and frozen
  release boundaries.

## Before public v0.1.0

- [ ] Validate the supported operating-system, Python, and device matrix.
- [ ] Publish PyPI v0.1.0 and the aligned GitHub release.

## After v0.1.0

Future work must be driven by measured product requirements. Possible areas
include smaller or quantized models, broader language support, additional
platform validation, service deployment guidance, and production-distribution
evaluation. None is part of the v0.1.0 release contract today.

See the [README](README.md) for the current public interfaces and measured
results.
