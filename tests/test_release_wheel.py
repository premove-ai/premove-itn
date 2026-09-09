from zipfile import ZipFile

import pytest

from scripts.inspect_release_wheel import inspect_wheel

METADATA = """Metadata-Version: 2.4
Name: premove-itn
Version: 0.1.0
Requires-Dist: huggingface-hub<2,>=0.36
Requires-Dist: safetensors<1,>=0.4
Requires-Dist: torch==2.13.0
Requires-Dist: transformers[sentencepiece]==5.16.1
"""


def _write_wheel(path, *, metadata=METADATA, extra_name=None, entry_point=True) -> None:
    with ZipFile(path, "w") as archive:
        archive.writestr("premove_itn-0.1.0.dist-info/METADATA", metadata)
        if entry_point:
            archive.writestr(
                "premove_itn-0.1.0.dist-info/entry_points.txt",
                "[console_scripts]\npremove-itn = premove_itn.cli:main\n",
            )
        archive.writestr("premove_itn/__init__.py", "")
        if extra_name is not None:
            archive.writestr(extra_name, "unexpected")


def test_release_wheel_requires_the_complete_contextual_runtime(tmp_path) -> None:
    wheel = tmp_path / "release.whl"
    _write_wheel(wheel)

    inspect_wheel(wheel)


def test_release_wheel_rejects_missing_runtime_dependency(tmp_path) -> None:
    wheel = tmp_path / "release.whl"
    _write_wheel(wheel, metadata=METADATA.replace("Requires-Dist: torch==2.13.0\n", ""))

    with pytest.raises(RuntimeError, match="missing runtime dependencies.*torch"):
        inspect_wheel(wheel)


def test_release_wheel_requires_the_cli_entry_point(tmp_path) -> None:
    wheel = tmp_path / "release.whl"
    _write_wheel(wheel, entry_point=False)

    with pytest.raises(RuntimeError, match="CLI entry point"):
        inspect_wheel(wheel)


@pytest.mark.parametrize(
    "artifact",
    [
        "premove_itn/model.safetensors",
        "premove_itn/data/training.jsonl",
        "premove_itn/eval/results.json",
    ],
)
def test_release_wheel_rejects_non_runtime_artifacts(tmp_path, artifact) -> None:
    wheel = tmp_path / "release.whl"
    _write_wheel(wheel, extra_name=artifact)

    with pytest.raises(RuntimeError, match="forbidden artifacts"):
        inspect_wheel(wheel)
