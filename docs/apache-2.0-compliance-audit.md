# Apache-2.0 dependency compliance audit

Date: 2026-08-22

Scope: `text-processing-rs` v0.3.0 as pinned by `rust/Cargo.toml` and
`rust/Cargo.lock`.

This is an engineering compliance audit, not legal advice.

## Conclusion

The dependency is compatible with this repository's MIT license. The compliance
files added with this audit close the repository-level documentation gaps. A
Rust-enabled release must still pass the artifact inspection gate below. The
README attribution is accurate, but it is not by itself a substitute for the
files required by Apache License 2.0 section 4.

Before the Rust extension is distributed, every artifact that contains compiled
or vendored `text-processing-rs` code must include:

1. a full copy of the Apache License 2.0;
2. the attribution text from the upstream `NOTICE` file; and
3. retained applicable copyright, patent, trademark, and attribution notices.

No modified-file notice is required while this project only links the unmodified
upstream dependency. If upstream source files are changed and redistributed,
each changed file must carry a prominent change notice.

## Verified upstream facts

- The pinned crate declares `license = "Apache-2.0"` in its
  [v0.3.0 Cargo manifest](https://github.com/FluidInference/text-processing-rs/blob/v0.3.0/Cargo.toml).
- Its [v0.3.0 LICENSE](https://github.com/FluidInference/text-processing-rs/blob/v0.3.0/LICENSE)
  contains Apache License 2.0 and identifies FluidInference as the 2026
  copyright holder in the appendix notice.
- Its [v0.3.0 NOTICE](https://github.com/FluidInference/text-processing-rs/blob/v0.3.0/NOTICE)
  identifies `text-processing-rs`, credits FluidInference, states that the work
  is a Rust port of NVIDIA NeMo Text Processing, and preserves the NVIDIA
  copyright and Apache-2.0 attribution.
- Apache License 2.0 section 4 permits redistribution under different terms for
  the larger work if its four redistribution conditions are met. It does not
  require `premove-itn` itself to change from MIT. See the
  [official license text](https://www.apache.org/licenses/LICENSE-2.0.txt).

## Requirements that apply

The controlling requirements are in
[Apache License 2.0 section 4](https://www.apache.org/licenses/LICENSE-2.0.html#redistribution):

- Section 4(a): recipients of the Work or a Derivative Work must receive a copy
  of the Apache License 2.0.
- Section 4(b): redistributed upstream files that this project modifies must
  carry prominent notices that the files were changed.
- Section 4(c): applicable notices present in distributed upstream source must
  be retained.
- Section 4(d): because upstream ships a `NOTICE`, a readable copy of its
  applicable attribution notices must accompany a distributed Derivative Work.
  The notice may be in a distributed `NOTICE` file, in accompanying source or
  documentation, or in a customary generated display.
- Section 6 permits reasonable attribution but does not grant broader trademark
  rights. The README's factual credit is appropriate; project branding must not
  imply endorsement by FluidInference, NVIDIA, or the Apache Software
  Foundation.

The ASF's official
[license application guidance](https://www.apache.org/legal/apply-license.html)
confirms that a distribution needs one full license copy and explains that
section 4(d) keeps required attribution notices with derivative distributions.
Its
[LICENSE and NOTICE assembly guidance](https://infra.apache.org/licensing-howto.html)
also states that source and binary distributions can have different dependency
contents and that their license materials must match the contents of each
artifact.

## Current repository status

| Item | Status | Evidence |
| --- | --- | --- |
| Dependency version and source are pinned | Pass | `rust/Cargo.toml` uses tag `v0.3.0`; `rust/Cargo.lock` pins the Git revision. |
| Upstream authorship and license are acknowledged | Pass | `README.md` credits FluidInference and links to the upstream Apache-2.0 license. |
| Project license remains clear | Pass | The root `LICENSE`, Python metadata, and Rust package metadata identify `premove-itn` as MIT. |
| Full Apache-2.0 text is tracked | Pass in this change | `LICENSES/text-processing-rs-Apache-2.0.txt` is a verbatim upstream copy. |
| Upstream `NOTICE` attribution is preserved | Pass in this change | `LICENSES/text-processing-rs-NOTICE.txt` is a verbatim upstream copy, and `THIRD_PARTY_NOTICES.md` points to it. |
| Python package metadata includes legal files | Pass in this change | `pyproject.toml` lists the root MIT license, both upstream text files, and `THIRD_PARTY_NOTICES.md` in `license-files`. |
| Current Hatch artifacts contain the legal files | Pass in this change | `uv build` followed by archive inspection confirms the sdist and wheel contain the MIT license, Apache-2.0 text, upstream NOTICE, and third-party notice. |
| Rust-enabled artifact contains the legal files | Release gate | Inspect the first Maturin-built wheel after the build backend changes. Do not infer its contents from source configuration alone. |
| Modified upstream files are marked | Not applicable | The dependency is fetched and linked without local upstream patches. Re-audit if source is vendored or patched. |

The current pure-Python Hatch wheel does not contain the Rust dependency. The
artifact inspection gate becomes release-blocking when a wheel, executable,
library, source archive, container image, or vendored source distribution
contains `text-processing-rs` in source or object form.

## Implemented documentation

This change uses files whose contents can be copied unchanged into every
release artifact:

1. `LICENSES/text-processing-rs-Apache-2.0.txt` is a verbatim copy of the
   upstream v0.3.0 `LICENSE`.
2. `LICENSES/text-processing-rs-NOTICE.txt` is a verbatim copy of the upstream
   v0.3.0 `NOTICE`.
3. `THIRD_PARTY_NOTICES.md` records the component, pinned revision, copyright,
   license, NeMo provenance, and paths to the verbatim texts.
4. `README.md` links to the bundled compliance record without changing the
   project's MIT license.
5. `pyproject.toml` declares these files as package license files.

Before a Rust-enabled release:

1. Confirm the Maturin wheel and source distribution include all three
   third-party files. Do the same for any container, executable bundle, or
   other release format that embeds the Rust crate.
2. Add an artifact-level test that builds the wheel, lists its contents, and
   asserts that the two upstream files are present and byte-identical to the
   tracked copies.
3. Re-run the dependency license audit whenever the pinned upstream tag changes,
   because its `LICENSE`, `NOTICE`, authorship, or bundled dependencies can
   change.

The copied Apache license must not replace the root MIT `LICENSE`. It documents
the third-party component only. The upstream `NOTICE` text should remain brief
and unchanged; do not add promotional language or terms that appear to modify
the Apache license.

## Release gate

A Rust-enabled artifact is ready for release only when inspection of the built
artifact proves all of the following:

- the project is identified as MIT;
- the complete Apache License 2.0 text is included for `text-processing-rs`;
- the FluidInference and NVIDIA notices from upstream `NOTICE` are readable;
- no unsupported endorsement or trademark claim is present; and
- any locally modified upstream source files state that they were changed.
