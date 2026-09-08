from pathlib import Path

import scripts.supervise_conversational_adaptation as supervisor


def test_supervisor_creates_a_clean_run_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output = tmp_path / "new-run"
    log = output / "supervisor.log"
    monkeypatch.setattr(supervisor, "OUTPUT", output)
    monkeypatch.setattr(supervisor, "LOG", log)
    monkeypatch.setattr(supervisor, "PROGRESS", output / "progress.json")
    monkeypatch.setattr(supervisor, "run_supervised", lambda *args, **kwargs: 0)

    supervisor.main()

    assert log.is_file()
