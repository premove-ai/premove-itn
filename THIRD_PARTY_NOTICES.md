# Third-party notices

`premove-itn` depends on
[`text-processing-rs`](https://github.com/FluidInference/text-processing-rs)
version 0.3.0, pinned to commit
[`27d75c4`](https://github.com/FluidInference/text-processing-rs/commit/27d75c401c225771fc16053dc551a887c2fbfc50).

`text-processing-rs` is Copyright 2026 FluidInference and is licensed under the
Apache License, Version 2.0. It is a Rust port of NVIDIA NeMo Text Processing,
which is also licensed under the Apache License, Version 2.0 and carries
Copyright (c) NVIDIA CORPORATION & AFFILIATES.

The complete upstream license and NOTICE text are distributed with this
project:

- [`LICENSES/text-processing-rs-Apache-2.0.txt`](LICENSES/text-processing-rs-Apache-2.0.txt)
- [`LICENSES/text-processing-rs-NOTICE.txt`](LICENSES/text-processing-rs-NOTICE.txt)

The `premove-itn` source code remains licensed under the MIT License. Third-party
components remain subject to their respective license and notice terms.

## Optional Thutmose comparison worker

The worker in `benchmarks/run_comparison.py` implements the inference and
detokenization procedure of NVIDIA NeMo's Thutmose tagger. It is adapted for
loading the older model weights with PyTorch and Transformers in an isolated
process. The source procedure is available in NeMo v1.9.0:

- [BERT examples](https://github.com/NVIDIA-NeMo/Speech/blob/v1.9.0/nemo/collections/nlp/data/text_normalization_as_tagging/bert_example.py)
- [Tag realization](https://github.com/NVIDIA-NeMo/Speech/blob/v1.9.0/nemo/collections/nlp/data/text_normalization_as_tagging/tagging.py)

NeMo source is licensed under Apache License 2.0. The license text is included
in `LICENSES/text-processing-rs-Apache-2.0.txt`. NVIDIA model weights are not
redistributed here; users must obtain them under the applicable model terms.
