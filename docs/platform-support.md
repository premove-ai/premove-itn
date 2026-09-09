# Platform support

Premove ITN v0.1.0 has a conservative compatibility target. A platform is
listed as validated only after a release wheel is built, installed without a
Rust compiler in a clean environment, and tested by the retained Stage 7
workflow.

## Certification target

| Operating system | Architecture | Python | Package | Inference device |
| --- | --- | --- | --- | --- |
| macOS 14+ | arm64 | 3.11, 3.12, 3.13 | Validated | MPS validated on macOS 15.6 |
| Linux (glibc 2.28+) | x86_64 | 3.11, 3.12, 3.13 | `manylinux_2_28` | CPU |

Real model certification uses Python 3.11 once per device family. It runs the
exact frozen 1,500 inputs and requires every output string to equal the retained
Premove release prediction. This is an execution-equivalence check. It is not
an accuracy evaluation and it must not be used to tune the model.

The workflow records the OS, architecture, Python, PyTorch, Transformers, Rust
target, wheel filename and digest, source commit, model revision, reference
prediction digest, selected device, result, and observed peak process RSS. The
final validated table will be populated only from
successful retained workflow evidence. The MPS evidence was produced with the
same clean release wheel on an Apple Silicon host because the standard public
macOS runner does not have enough shared memory for this model, and the
repository does not have access to the larger macOS runner tier.

## Not validated for v0.1.0

- Windows
- macOS Intel
- Linux ARM64
- CUDA
- ROCm and other accelerator backends

The API exposes `device="cuda"`, but CUDA is not a v0.1.0 compatibility claim
until it is tested on real NVIDIA hardware. Unsupported or unavailable explicit
devices fail with a clear runtime error.
