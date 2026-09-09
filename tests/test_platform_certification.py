from pathlib import Path

import pytest

from scripts.certify_installed_platform import (
    normalized_architecture,
    verify_expected_platform,
    verify_wheel_platform,
)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("AMD64", "x86_64"),
        ("x86_64", "x86_64"),
        ("aarch64", "arm64"),
        ("arm64", "arm64"),
    ],
)
def test_normalized_architecture_uses_matrix_names(source, expected) -> None:
    assert normalized_architecture(source) == expected


def test_verify_expected_platform_accepts_patch_python_version() -> None:
    verify_expected_platform(
        {"os": "Linux", "architecture": "x86_64", "python": "3.12.9"},
        expected_os="Linux",
        expected_arch="x86_64",
        expected_python="3.12",
    )


def test_verify_expected_platform_rejects_wrong_matrix_cell() -> None:
    with pytest.raises(RuntimeError, match="platform mismatch"):
        verify_expected_platform(
            {"os": "Darwin", "architecture": "arm64", "python": "3.13.1"},
            expected_os="Darwin",
            expected_arch="arm64",
            expected_python="3.12",
        )


def test_verify_wheel_platform_accepts_exact_tag() -> None:
    verify_wheel_platform(
        Path(
            "premove_itn-0.1.0-cp311-cp311-manylinux_2_28_x86_64.whl"
        ),
        "manylinux_2_28_x86_64",
    )


def test_verify_wheel_platform_rejects_generic_linux_tag() -> None:
    with pytest.raises(RuntimeError, match="wheel platform mismatch"):
        verify_wheel_platform(
            Path(
                "premove_itn-0.1.0-cp311-cp311-linux_x86_64.whl"
            ),
            "manylinux_2_28_x86_64",
        )
