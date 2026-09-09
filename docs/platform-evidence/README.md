# Stage 7 certification evidence

This directory records the v0.1.0 compatibility certification.

- All six release-wheel cells passed: macOS arm64 and Linux x86_64 on Python
  3.11, 3.12, and 3.13.
- MPS and CPU each produced **1,500/1,500 exact output matches** against the
  retained First Evaluation Premove predictions.
- `device="auto"` selected MPS on Apple Silicon and CPU on Linux.
- Explicit unavailable-device checks passed for CUDA on macOS and MPS on the
  CPU-only Linux runner.
- The MPS run used the same clean release wheel on an Apple Silicon host. The
  standard public macOS runner cannot fit this model in its MPS memory limit,
  and the larger macOS runner tier is not available to this repository.

The JSON files contain the environment and native-build details recorded by the
certification scripts. This is execution-equivalence evidence, not a new model
accuracy evaluation and not a tuning input.
