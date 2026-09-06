import json

from scripts.evaluate_voicecodebench import (
    DEFAULT_RUNS,
    SOURCE_REVISION,
    load_rows,
    reachability_cache_matches,
    reachability_fingerprint,
)


def test_default_runs_include_both_retained_checkpoints() -> None:
    assert [checkpoint.parent.name for checkpoint, _ in DEFAULT_RUNS] == [
        "google_selected_378k",
        "conversational_selected_20k",
    ]


def test_load_rows_builds_sentence_and_entity_views(tmp_path) -> None:
    source = tmp_path / "metadata.jsonl"
    source.write_text(
        json.dumps(
            {
                "audio_id": "sample-1",
                "domain": "technical",
                "transcripts": {
                    "acoustic": "use version one point two point three",
                    "canonical": "use version 1.2.3",
                },
                "entities": [
                    {
                        "id": "entity-1",
                        "type": "version",
                        "acoustic": "one point two point three",
                        "canonical": "1.2.3",
                    }
                ],
            }
        )
        + "\n"
    )

    sentences, entities = load_rows(source)
    assert sentences == [
        {
            "id": "sample-1",
            "category": "technical",
            "text": "use version one point two point three",
            "original_text": "use version 1.2.3",
        }
    ]
    assert entities == [
        {
            "id": "entity-1",
            "category": "version",
            "text": "one point two point three",
            "original_text": "1.2.3",
        }
    ]


def test_reachability_fingerprint_hashes_loaded_implementation_files(tmp_path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    files = (("first", first), ("second", second))
    original = reachability_fingerprint(files)
    assert original == reachability_fingerprint(files)
    second.write_bytes(b"changed")
    assert reachability_fingerprint(files) != original


def test_reachability_cache_requires_complete_fingerprint() -> None:
    cached = {
        "source_revision": SOURCE_REVISION,
        "dataset_sha256": "current-dataset",
        "identities": ["entity-1"],
        "reachability_fingerprint": "current-implementation",
    }
    assert reachability_cache_matches(
        cached, ["entity-1"], "current-dataset", "current-implementation"
    )
    assert not reachability_cache_matches(
        cached, ["entity-1"], "current-dataset", "changed-implementation"
    )
    assert not reachability_cache_matches(
        cached, ["entity-2"], "current-dataset", "current-implementation"
    )
    assert not reachability_cache_matches(
        cached, ["entity-1"], "changed-dataset", "current-implementation"
    )
    assert not reachability_cache_matches(
        {"source_revision": SOURCE_REVISION, "identities": ["entity-1"]},
        ["entity-1"],
        "current-dataset",
        "current-implementation",
    )
