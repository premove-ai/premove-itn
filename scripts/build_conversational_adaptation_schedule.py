"""Build the deterministic 44k conversational-adaptation record schedule."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from premove_itn import build_candidate_graph

ROOT = Path(__file__).resolve().parents[1]
CONVERSATIONAL_POOL = ROOT / "data/generated/conversational_pool"
CONVERSATIONAL_DATASET = CONVERSATIONAL_POOL / "dataset.jsonl"
CONVERSATIONAL_PROVENANCE = CONVERSATIONAL_POOL / "provenance.jsonl"
GOOGLE_DATASET = ROOT / "datasets/google_tn/dataset_1/dataset.jsonl"
OUTPUT = ROOT / "data/generated/conversational_adaptation_44k"
SEED = "premove-itn/conversational-adaptation-44k/v1"
CANDIDATE_KEEP_COUNT = 25_176
CONVERSATIONAL_POSITIVE_COUNT = 12_588
GOOGLE_REPLAY_COUNT = 6_294
BUCKET_PATTERN = (
    "candidate_bearing_keep",
    "conversational_positive",
    "candidate_bearing_keep",
    "google_positive_replay",
    "candidate_bearing_keep",
    "conversational_positive",
    "candidate_bearing_keep",
)
PREFIXES = (1_000, 10_000, 20_000, 30_000)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(*parts: object) -> str:
    value = "\0".join((SEED, *(str(part) for part in parts)))
    return hashlib.sha256(value.encode()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.open()]


def _candidate_profile(item: tuple[int, str]) -> tuple[int, int, tuple[str, ...]]:
    record_id, text = item
    candidates = build_candidate_graph(text)
    kinds = tuple(sorted({kind.value for item in candidates for kind in item.kinds}))
    return record_id, len(candidates), kinds


def _profiles(
    items: Iterable[tuple[int, str]],
    *,
    workers: int,
    profiler: Callable[
        [tuple[int, str]], tuple[int, int, tuple[str, ...]]
    ] = _candidate_profile,
) -> Iterable[tuple[int, int, tuple[str, ...]]]:
    if workers == 1:
        yield from map(profiler, items)
        return
    with ProcessPoolExecutor(max_workers=workers) as executor:
        yield from executor.map(profiler, items, chunksize=128)


def _source_names(provenance: dict[str, object]) -> list[str]:
    sources = provenance.get("sources")
    if not isinstance(sources, list):
        raise ValueError("conversational provenance has no sources")
    names = sorted(
        {
            str(source["source"])
            for source in sources
            if isinstance(source, dict) and source.get("source")
        }
    )
    if not names:
        raise ValueError("conversational provenance has no source names")
    return names


def _reference(
    *,
    dataset: Path,
    record_id: int,
    bucket: str,
    row: dict[str, object],
    sources: list[str],
    candidate_kinds: tuple[str, ...],
) -> dict[str, object]:
    return {
        "source_dataset": str(dataset.relative_to(ROOT)),
        "partition": "train",
        "record_id": record_id,
        "bucket": bucket,
        "kind": str(row["kind"]),
        "kinds": list(row.get("kinds", [])),
        "sources": sources,
        "candidate_bearing": True,
        "candidate_kinds": list(candidate_kinds),
    }


def _rank(records: Iterable[dict[str, object]], label: str) -> list[dict[str, object]]:
    return sorted(
        records,
        key=lambda record: _digest(
            label,
            record["source_dataset"],
            record["record_id"],
        ),
    )


def _balanced_google_replay(
    records: list[dict[str, object]], count: int
) -> list[dict[str, object]]:
    by_kind: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        by_kind[str(record["kind"])].append(record)
    for kind, items in by_kind.items():
        by_kind[kind] = _rank(items, f"google:{kind}")
    selected = []
    offsets = Counter()
    kinds = sorted(by_kind)
    while len(selected) < count:
        progressed = False
        for kind in kinds:
            offset = offsets[kind]
            if offset >= len(by_kind[kind]):
                continue
            selected.append(by_kind[kind][offset])
            offsets[kind] += 1
            progressed = True
            if len(selected) == count:
                break
        if not progressed:
            raise ValueError(f"Google train has fewer than {count} positive records")
    return selected


def _interleave(
    keep: list[dict[str, object]],
    positive: list[dict[str, object]],
    google: list[dict[str, object]],
) -> list[dict[str, object]]:
    streams = {
        "candidate_bearing_keep": iter(keep),
        "conversational_positive": iter(positive),
        "google_positive_replay": iter(google),
    }
    schedule = []
    cycles = len(google)
    for _ in range(cycles):
        for bucket in BUCKET_PATTERN:
            schedule.append(next(streams[bucket]))
    for bucket, stream in streams.items():
        try:
            next(stream)
        except StopIteration:
            continue
        raise ValueError(f"{bucket} count does not match the 2:1:0.5 schedule")
    return schedule


def _distribution(schedule: list[dict[str, object]], limit: int) -> dict[str, object]:
    selected = schedule[:limit]
    return {
        "records": len(selected),
        "buckets": dict(sorted(Counter(row["bucket"] for row in selected).items())),
        "kinds": dict(sorted(Counter(row["kind"] for row in selected).items())),
        "sources": dict(
            sorted(
                Counter(source for row in selected for source in row["sources"]).items()
            )
        ),
    }


def build_schedule(
    conversational_dataset: Path,
    conversational_provenance: Path,
    google_dataset: Path,
    output: Path,
    *,
    candidate_keep_count: int = CANDIDATE_KEEP_COUNT,
    conversational_positive_count: int = CONVERSATIONAL_POSITIVE_COUNT,
    google_replay_count: int = GOOGLE_REPLAY_COUNT,
    workers: int = min(7, os.cpu_count() or 1),
    profiler: Callable[
        [tuple[int, str]], tuple[int, int, tuple[str, ...]]
    ] = _candidate_profile,
) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if workers < 1:
        raise ValueError("workers must be positive")
    if candidate_keep_count != google_replay_count * 4:
        raise ValueError("candidate-bearing KEEP count must be four times replay")
    if conversational_positive_count != google_replay_count * 2:
        raise ValueError("conversational positive count must be twice replay")

    conversational = _read_jsonl(conversational_dataset)
    provenance = _read_jsonl(conversational_provenance)
    if len(conversational) != len(provenance):
        raise ValueError("conversational rows and provenance are misaligned")

    positives = []
    keep_rows: dict[int, tuple[dict[str, object], list[str]]] = {}
    keep_inputs = []
    for record_id, (row, source) in enumerate(
        zip(conversational, provenance, strict=True)
    ):
        if source.get("record_index") != record_id:
            raise ValueError("conversational provenance record index is misaligned")
        if row.get("partition") != "train":
            continue
        sources = _source_names(source)
        if row.get("kind") == "KEEP":
            keep_rows[record_id] = (row, sources)
            keep_inputs.append((record_id, str(row["text"])))
            continue
        positives.append(
            _reference(
                dataset=conversational_dataset,
                record_id=record_id,
                bucket="conversational_positive",
                row=row,
                sources=sources,
                candidate_kinds=tuple(str(kind) for kind in row.get("kinds", [])),
            )
        )
    if len(positives) != conversational_positive_count:
        raise ValueError(
            f"expected {conversational_positive_count} conversational positives; "
            f"found {len(positives)}"
        )

    candidate_keep = []
    for record_id, count, kinds in _profiles(
        keep_inputs, workers=workers, profiler=profiler
    ):
        if count == 0:
            continue
        row, sources = keep_rows[record_id]
        candidate_keep.append(
            _reference(
                dataset=conversational_dataset,
                record_id=record_id,
                bucket="candidate_bearing_keep",
                row=row,
                sources=sources,
                candidate_kinds=kinds,
            )
        )
    if len(candidate_keep) < candidate_keep_count:
        raise ValueError(
            f"only {len(candidate_keep)} candidate-bearing KEEP records are available"
        )
    candidate_keep = _rank(candidate_keep, "candidate-bearing-keep")[
        :candidate_keep_count
    ]
    positives = _rank(positives, "conversational-positive")

    google = []
    for record_id, row in enumerate(_read_jsonl(google_dataset)):
        if row.get("partition") != "train" or row.get("kind") == "KEEP":
            continue
        google.append(
            _reference(
                dataset=google_dataset,
                record_id=record_id,
                bucket="google_positive_replay",
                row=row,
                sources=["google_tn_dataset_1"],
                candidate_kinds=tuple(str(kind) for kind in row.get("kinds", [])),
            )
        )
    google = _balanced_google_replay(google, google_replay_count)
    schedule = _interleave(candidate_keep, positives, google)

    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        schedule_path = temporary / "training_schedule.jsonl"
        with schedule_path.open("w") as handle:
            for record in schedule:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        prefixes = sorted({*PREFIXES, len(schedule)})
        manifest = {
            "schema_version": 1,
            "kind": "training_schedule",
            "seed": SEED,
            "records": len(schedule),
            "bucket_pattern": list(BUCKET_PATTERN),
            "bucket_distribution": dict(
                sorted(Counter(row["bucket"] for row in schedule).items())
            ),
            "kind_distribution": dict(
                sorted(Counter(row["kind"] for row in schedule).items())
            ),
            "source_distribution": dict(
                sorted(
                    Counter(
                        source for row in schedule for source in row["sources"]
                    ).items()
                )
            ),
            "prefix_distribution": {
                str(prefix): _distribution(schedule, prefix) for prefix in prefixes
            },
            "selection": {
                "candidate_bearing_keep": "seeded SHA-256 sample without replacement",
                "conversational_positive": (
                    "every train-partition positive exactly once"
                ),
                "google_positive_replay": "seeded round-robin across positive kinds",
                "loss_weighting": "none",
                "validation_included": False,
                "test_included": False,
            },
            "inputs": {
                str(conversational_dataset.relative_to(ROOT)): sha256(
                    conversational_dataset
                ),
                str(conversational_provenance.relative_to(ROOT)): sha256(
                    conversational_provenance
                ),
                str(google_dataset.relative_to(ROOT)): sha256(google_dataset),
            },
            "artifacts": {
                "training_schedule.jsonl": {
                    "bytes": schedule_path.stat().st_size,
                    "sha256": sha256(schedule_path),
                }
            },
        }
        manifest_path = temporary / "schedule_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=min(7, os.cpu_count() or 1))
    args = parser.parse_args()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_schedule(
        CONVERSATIONAL_DATASET,
        CONVERSATIONAL_PROVENANCE,
        GOOGLE_DATASET,
        OUTPUT,
        workers=args.workers,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
