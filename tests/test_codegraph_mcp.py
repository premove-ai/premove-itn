from pathlib import Path

from scripts.agent.codegraph_mcp import resolve_codegraph, standalone_codegraph_path


def test_standalone_path_is_shared_for_non_windows(tmp_path: Path) -> None:
    assert standalone_codegraph_path(home=tmp_path, system="Darwin") == (
        tmp_path / ".local" / "bin" / "codegraph"
    )


def test_windows_path_requires_local_app_data() -> None:
    assert standalone_codegraph_path(system="Windows", environment={}) is None


def test_resolves_standalone_install_when_codegraph_is_not_on_path(
    tmp_path: Path,
) -> None:
    codegraph = tmp_path / ".local" / "bin" / "codegraph"
    codegraph.parent.mkdir(parents=True)
    codegraph.write_text("#!/bin/sh\n")

    resolved = resolve_codegraph(
        which=lambda _tool: None,
        home=tmp_path,
        system="Darwin",
    )

    assert resolved == str(codegraph)
