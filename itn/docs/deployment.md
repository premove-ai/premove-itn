---
title: Deployment
description: Deploy Premove ITN with correct model lifecycle, device, and platform expectations.
---

Keep one loaded `PremoveITN` instance resident and reuse it across requests.

## Separate the lifecycle costs

| Phase | Meaning |
| --- | --- |
| First download | Fetches about 1.6 GB from Hugging Face. Duration depends on the network. |
| Cached initialization | Builds the model, loads weights, and moves the model to the selected device. This took several seconds on the tested system. |
| Warm normalization | Processes one transcript after load and warm-up. The retained release mean was 56.49 ms on Apple M4/MPS. |

The benchmark excludes download and initialization. A one-shot CLI measurement
is not comparable to retained warm latency.

## Select a device

`device="auto"` selects CUDA when available, then Apple MPS, then CPU. It
prefers CUDA when available, but CUDA is not a validated v0.1.0 platform. Use
CPU on supported Linux or MPS on supported Apple Silicon when you require a
validated runtime. Pass `cpu`, `mps`, or `cuda` to select a device explicitly.
An unavailable explicit device fails with a runtime error.

Release wheels are validated on macOS 14+ arm64 and `manylinux_2_28` x86_64 for
Python 3.11–3.13. Frozen-model inference is validated on Apple Silicon MPS and
Linux x86-64 CPU. CUDA, Windows, macOS Intel, Linux ARM64, and other
accelerators are not validated v0.1.0 support claims.

See the [platform support matrix](/itn/docs/internals/platform-support) for the exact
release boundary.

## Pin production deployments

Pin the package release:

```bash
pip install premove-itn==0.1.0
```

The package pins and verifies the model artifact. For an offline deployment,
download the artifact in advance and pass its local directory to
`PremoveITN.from_pretrained()`.

## Protect service capacity

The scorer has 435.6 million parameters. Account for model memory and startup
time when you choose process counts and readiness checks. Do not load a new
model instance for every request.
