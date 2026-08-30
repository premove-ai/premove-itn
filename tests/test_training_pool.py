import json
from pathlib import Path

import pytest

from scripts.audit_training_pool import audit
from scripts.build_training_pool import build_training_pool


class FakeTokenizer:
    def __call__(self, text: str, **options: object) -> dict[str, list[int]]:
        del options
        return {"input_ids": [0, *(1 for _ in text.split()), 2]}


def _write_compiled(
    root: Path,
    source: str,
    records: list[dict[str, object]],
    *,
    license_id: str = "CC-BY-4.0",
) -> None:
    root.mkdir()
    (root / "dataset.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    (root / "provenance.jsonl").write_text(
        "".join(
            json.dumps({"source_id": f"{source}-{index}"}) + "\n"
            for index in range(len(records))
        ),
        encoding="utf-8",
    )
    (root / "manifest.json").write_text(
        json.dumps({"kind": source, "records": len(records), "license": license_id}),
        encoding="utf-8",
    )


def _record(
    text: str,
    expected: str,
    partition: str,
    kind: str = "KEEP",
) -> dict[str, object]:
    return {
        "text": text,
        "expected_text": expected,
        "partition": partition,
        "kind": kind,
        "kinds": [] if kind == "KEEP" else [kind],
    }


def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_build_training_pool_deduplicates_repartitions_and_limits_tokens(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    long_text = " ".join(["word"] * 511)
    _write_compiled(
        first,
        "first_source",
        [
            _record("one", "1", "train", "CARDINAL"),
            _record("shared text", "shared text", "train"),
            _record(long_text, long_text, "train"),
        ],
    )
    _write_compiled(
        second,
        "second_source",
        [
            _record("one", "1", "validation", "CARDINAL"),
            _record("1", "1", "test"),
        ],
    )

    output = tmp_path / "pool"
    manifest = build_training_pool([second, first], output, tokenizer=FakeTokenizer())
    rows = _read_jsonl(output / "dataset.jsonl")
    provenance = _read_jsonl(output / "provenance.jsonl")

    assert manifest["records_before_deduplication"] == 5
    assert manifest["duplicates"] == 1
    assert manifest["records"] == 3
    assert manifest["model_token_gate"]["excluded"] == 1
    assert {row["text"] for row in rows} == {"one", "1", "shared text"}
    one_partition = next(row["partition"] for row in rows if row["text"] == "one")
    assert next(row["partition"] for row in rows if row["text"] == "1") == one_partition
    duplicate_provenance = next(
        item for item in provenance if len(item["sources"]) == 2
    )
    assert [source["source"] for source in duplicate_provenance["sources"]] == [
        "first_source",
        "second_source",
    ]
    assert audit(output, tokenizer=FakeTokenizer()) == 0


def test_build_training_pool_is_input_order_independent(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_compiled(first, "a", [_record("one", "1", "train", "CARDINAL")])
    _write_compiled(second, "b", [_record("two", "2", "test", "CARDINAL")])

    left = tmp_path / "left"
    right = tmp_path / "right"
    build_training_pool([first, second], left)
    build_training_pool([second, first], right)

    for name in (
        "dataset.jsonl",
        "train.jsonl",
        "validation.jsonl",
        "test.jsonl",
        "provenance.jsonl",
        "excluded.jsonl",
    ):
        assert (left / name).read_bytes() == (right / name).read_bytes()


def test_build_training_pool_refuses_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_compiled(source, "source", [_record("keep", "keep", "train")])
    output = tmp_path / "pool"
    output.mkdir()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_training_pool([source], output)


def test_build_training_pool_never_publishes_an_unreachable_record(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _write_compiled(source, "source", [_record("one", "9", "train", "CARDINAL")])
    output = tmp_path / "pool"

    with pytest.raises(ValueError, match="not GoldGraph-reachable"):
        build_training_pool([source], output)

    assert not output.exists()


def test_build_training_pool_requires_restrictive_license_opt_in(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _write_compiled(
        source,
        "restricted_source",
        [_record("keep", "keep", "train")],
        license_id="CC-BY-NC-4.0",
    )

    with pytest.raises(ValueError, match="non-commercial research opt-in"):
        build_training_pool([source], tmp_path / "refused")

    output = tmp_path / "accepted"
    manifest = build_training_pool([source], output, non_commercial_research=True)
    assert manifest["licensing"] == {
        "use_scope": "non_commercial_research",
        "sources": [
            {
                "source": "restricted_source",
                "license": "CC-BY-NC-4.0",
                "restrictions": [
                    "attribution",
                    "change_indication",
                    "non_commercial",
                ],
            }
        ],
        "effective_restrictions": [
            "attribution",
            "change_indication",
            "non_commercial",
        ],
    }


def test_build_training_pool_requires_missing_license_override(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _write_compiled(source, "legacy_source", [_record("keep", "keep", "train")])
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("license")
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="missing license metadata"):
        build_training_pool([source], tmp_path / "refused")

    with pytest.raises(ValueError, match="non-commercial research opt-in"):
        build_training_pool(
            [source],
            tmp_path / "restricted",
            source_licenses={"legacy_source": "LicenseRef-Kaggle-Competition-Rules"},
        )

    accepted = build_training_pool(
        [source],
        tmp_path / "accepted",
        non_commercial_research=True,
        source_licenses={"legacy_source": "LicenseRef-Kaggle-Competition-Rules"},
    )
    assert accepted["licensing"]["sources"][0] == {
        "source": "legacy_source",
        "license": "LicenseRef-Kaggle-Competition-Rules",
        "restrictions": [
            "competition_rules",
            "no_redistribution_without_permission",
        ],
    }
