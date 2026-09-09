"""Exercise an installed release wheel against the frozen Hub artifact."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import premove_itn
from premove_itn import PremoveITN, _rust
from premove_itn.contextual import DEFAULT_REVISION


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
        help="inference device (default: auto)",
    )
    arguments = parser.parse_args(argv)
    package_path = Path(premove_itn.__file__).resolve()
    if "site-packages" not in package_path.parts:
        raise RuntimeError(
            f"premove_itn was not imported from site-packages: {package_path}"
        )

    build = _rust.build_info()
    if build["profile"] != "release" or build["debug_assertions"] is not False:
        raise RuntimeError(f"wheel contains a non-release Rust extension: {build}")

    itn = PremoveITN.from_pretrained(device=arguments.device)
    if itn.revision != DEFAULT_REVISION:
        raise RuntimeError(f"unexpected model revision: {itn.revision}")
    expected = {
        "call me at four thirty": "call me at 04:30",
        "the total is twenty dollars": "the total is $20",
    }
    actual = {text: itn.normalize(text) for text in expected}
    if actual != expected:
        raise RuntimeError(f"installed-wheel normalization mismatch: {actual}")
    print(f"verified installed package: {package_path}")
    print(f"verified model revision: {itn.revision}")


if __name__ == "__main__":
    main()
