---
title: Premove ITN platform support
description: Check validated macOS and Linux release targets, Python versions, inference devices, and certification evidence.
---

Premove ITN v0.2.0 retains the conservative compatibility target established
for v0.1.0. A platform is listed as validated only after a release wheel is
built, installed without a Rust compiler in a clean environment, and tested by
the release workflow.

## Certification target

| Operating system | Architecture | Python | Package | Inference device |
| --- | --- | --- | --- | --- |
| macOS 14+ | arm64 | 3.11, 3.12, 3.13 | Validated | MPS validated on macOS 15.6 |
| Linux (glibc 2.28+) | x86_64 | 3.11, 3.12, 3.13 | `manylinux_2_28` | CPU |

The retained 1,500-input execution-equivalence records under
`docs/platform-evidence/` are v0.1.0 historical evidence. The v0.2.0 release
workflow separately passed all six release-wheel builds, Hub provenance checks,
and public-install inference smoke checks. These are execution checks, not an
accuracy evaluation and not tuning inputs.

The workflow records the OS, architecture, Python, PyTorch, Transformers, Rust
target, wheel filename and digest, source commit, model revision, reference
prediction digest, selected device, result, and observed peak process RSS. The
validated table above is based on successful retained workflow evidence. The
MPS evidence was produced with the
same clean release wheel on an Apple Silicon host because the standard public
macOS runner does not have enough shared memory for this model, and the
repository does not have access to the larger macOS runner tier.

## Not validated for v0.2.0

- Windows
- macOS Intel
- Linux ARM64
- CUDA
- ROCm and other accelerator backends

The API exposes `device="cuda"`, but CUDA is not a v0.2.0 compatibility claim
until it is tested on real NVIDIA hardware. Unsupported or unavailable explicit
devices fail with a clear runtime error.

## Runtime boundary

`device="auto"` selects CUDA first, then Apple MPS, then CPU. Selection reports
an available backend; it does not certify that backend for this release. The
loader verifies the pinned inference artifact before constructing the scorer,
then keeps the model and tokenizer resident. A non-empty transcript with
candidates is encoded without truncation and fails above 512 encoder tokens.
Empty, whitespace-only, and no-candidate inputs return unchanged. See the
[architecture](/itn/docs/internals/architecture) for the hot path and
[inference artifact](/itn/docs/internals/inference-artifact) for the checks
performed before the model runs.
