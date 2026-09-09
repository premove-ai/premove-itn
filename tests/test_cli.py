from __future__ import annotations

import io
from types import SimpleNamespace

import pytest

from premove_itn import cli


def test_direct_text_normalizes_with_selected_device(monkeypatch, capsys) -> None:
    loaded = SimpleNamespace(normalize=lambda text: "call me at 04:30")
    devices = []

    def load(*, device):
        devices.append(device)
        return loaded

    monkeypatch.setattr(cli.PremoveITN, "from_pretrained", load)

    status = cli.main(["--device", "cpu", "call me at four thirty"])

    captured = capsys.readouterr()
    assert status == 0
    assert captured.out == "call me at 04:30\n"
    assert captured.err == ""
    assert devices == ["cpu"]


def test_stdin_normalizes_each_line_with_one_loaded_model(monkeypatch, capsys) -> None:
    class Loaded:
        def __init__(self) -> None:
            self.inputs = []

        def normalize(self, text):
            self.inputs.append(text)
            return {"four thirty": "04:30", "twenty dollars": "$20"}[text]

    loaded = Loaded()
    loads = []

    def load(*, device):
        loads.append(device)
        return loaded

    monkeypatch.setattr(cli.PremoveITN, "from_pretrained", load)
    monkeypatch.setattr(cli.sys, "stdin", io.StringIO("four thirty\ntwenty dollars\n"))

    status = cli.main([])

    captured = capsys.readouterr()
    assert status == 0
    assert captured.out == "04:30\n$20\n"
    assert captured.err == ""
    assert loads == ["auto"]
    assert loaded.inputs == ["four thirty", "twenty dollars"]


def test_no_input_from_terminal_fails_before_model_load(monkeypatch, capsys) -> None:
    class TerminalInput(io.StringIO):
        def isatty(self):
            return True

    def unexpected_load(**kwargs):
        raise AssertionError(f"model must not load: {kwargs}")

    monkeypatch.setattr(cli.PremoveITN, "from_pretrained", unexpected_load)
    monkeypatch.setattr(cli.sys, "stdin", TerminalInput())

    with pytest.raises(SystemExit) as exit_info:
        cli.main([])

    captured = capsys.readouterr()
    assert exit_info.value.code == 2
    assert captured.out == ""
    assert "no input provided" in captured.err


def test_model_load_error_is_clean_and_nonzero(monkeypatch, capsys) -> None:
    def failed_load(**kwargs):
        raise RuntimeError("artifact is unavailable")

    monkeypatch.setattr(cli.PremoveITN, "from_pretrained", failed_load)

    status = cli.main(["hello"])

    captured = capsys.readouterr()
    assert status == 1
    assert captured.out == ""
    assert captured.err == (
        "premove-itn: failed to load model: artifact is unavailable\n"
    )


def test_normalization_error_is_clean_and_nonzero(monkeypatch, capsys) -> None:
    def fail(text):
        raise RuntimeError("input is too long")

    monkeypatch.setattr(
        cli.PremoveITN,
        "from_pretrained",
        lambda **kwargs: SimpleNamespace(normalize=fail),
    )

    status = cli.main(["hello"])

    captured = capsys.readouterr()
    assert status == 1
    assert captured.out == ""
    assert captured.err == "premove-itn: failed to normalize input: input is too long\n"


def test_version_reports_installed_package_version(capsys) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--version"])

    captured = capsys.readouterr()
    assert exit_info.value.code == 0
    assert captured.out == "premove-itn 0.1.0\n"
    assert captured.err == ""


def test_invalid_device_fails_before_model_load(monkeypatch, capsys) -> None:
    def unexpected_load(**kwargs):
        raise AssertionError(f"model must not load: {kwargs}")

    monkeypatch.setattr(cli.PremoveITN, "from_pretrained", unexpected_load)

    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--device", "banana", "hello"])

    captured = capsys.readouterr()
    assert exit_info.value.code == 2
    assert captured.out == ""
    assert "invalid choice: 'banana'" in captured.err


def test_help_explains_direct_and_stdin_modes(capsys) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--help"])

    captured = capsys.readouterr()
    assert exit_info.value.code == 0
    assert "transcript to normalize" in captured.out
    assert "If text is omitted" in captured.out
    assert captured.err == ""
