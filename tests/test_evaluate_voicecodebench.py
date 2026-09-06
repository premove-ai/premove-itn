import json

from scripts.evaluate_voicecodebench import (
    DEFAULT_RUNS,
    SOURCE_REVISION,
    load_rows,
    reachability_cache_matches,
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


def test_reachability_cache_includes_rust_source_hash() -> None:
    cached = {
        "source_revision": SOURCE_REVISION,
        "identities": ["entity-1"],
        "rust_source_sha256": "current-rust",
    }
    assert reachability_cache_matches(cached, ["entity-1"], "current-rust")
    assert not reachability_cache_matches(cached, ["entity-1"], "changed-rust")
    assert not reachability_cache_matches(
        {"source_revision": SOURCE_REVISION, "identities": ["entity-1"]},
        ["entity-1"],
        "current-rust",
    )
