from pathlib import Path

from scripts.agent.codegraph_mcp import resolve_codegraph


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
