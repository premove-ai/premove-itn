import json
from pathlib import Path

from premove_itn import build_gold_graph
from scripts.build_taskmaster_1_dataset import build_dataset


def test_builds_only_spoken_user_turns_with_reachable_targets(tmp_path: Path) -> None:
    source = tmp_path / "woz-dialogs.json"
    source.write_text(
        json.dumps(
            [
                {
                    "conversation_id": "dlg-1",
                    "utterances": [
                        {
                            "index": 0,
                            "speaker": "ASSISTANT",
                            "text": "At seven thirty.",
                        },
                        {
                            "index": 1,
                            "speaker": "USER",
                            "text": "Book for three people.",
                            "segments": [
                                {
                                    "start_index": 9,
                                    "end_index": 21,
                                    "text": "three people",
                                    "annotations": [{"name": "uber_lyft.num.people"}],
                                }
                            ],
                        },
                        {
                            "index": 2,
                            "speaker": "USER",
                            "text": "No normalization here.",
                        },
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "dataset"
    manifest = build_dataset(source, output)
    rows = [
        json.loads(line) for line in (output / "dataset.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]["text"] == "Book for three people."
    assert rows[0]["expected_text"] == "Book for 3 people."
    assert rows[0]["kinds"] == ["CARDINAL"]
    assert build_gold_graph(rows[0]["text"], rows[0]["expected_text"]) is not None
    assert manifest["utterances_processed"] == 2
    assert manifest["records"] == 1
    assert manifest["quarantine_reasons"] == {"no_unambiguous_itn_edit": 1}


def test_rejects_bad_segment_offsets(tmp_path: Path) -> None:
    source = tmp_path / "woz-dialogs.json"
    source.write_text(
        json.dumps(
            [
                {
                    "conversation_id": "dlg-2",
                    "utterances": [
                        {
                            "index": 0,
                            "speaker": "USER",
                            "text": "three people",
                            "segments": [
                                {
                                    "start_index": 0,
                                    "end_index": 12,
                                    "text": "wrong",
                                    "annotations": [{"name": "uber_lyft.num.people"}],
                                }
                            ],
                        }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    manifest = build_dataset(source, tmp_path / "dataset")
    assert manifest["records"] == 0
    assert manifest["quarantine_reasons"] == {"segment_text_mismatch": 1}
