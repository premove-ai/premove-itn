from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from premove_itn.dataset.enrichment.sgd_context import (
    SgdSlotSpan,
    SgdUserTurn,
    iter_sgd_train_user_turns,
)

_SAMPLE_SEED = "sgd-context-audit-v1"


@dataclass(frozen=True, slots=True)
class SgdAuditExample:
    source_file: str
    dialogue_id: str
    turn_index: int
    utterance: str
    start: int
    end: int
    value: str


@dataclass(frozen=True, slots=True)
class SgdSlotAudit:
    service: str
    slot: str
    description: str
    count: int
    examples: tuple[SgdAuditExample, ...]


@dataclass(frozen=True, slots=True)
class SgdContextAudit:
    kind: str
    schema_version: int
    source_split: str
    dialogue_files: int
    dialogues: int
    user_turns: int
    context_only_user_turns: int
    non_categorical_slot_spans: int
    overlapping_span_turns: int
    service_slot_pairs: tuple[SgdSlotAudit, ...]


def _sample_priority(turn: SgdUserTurn, span: SgdSlotSpan) -> int:
    identity = "\0".join(
        (
            _SAMPLE_SEED,
            turn.source_file,
            turn.dialogue_id,
            str(turn.turn_index),
            span.service,
            span.slot,
            str(span.start),
            str(span.end),
            span.value,
        )
    )
    return int.from_bytes(hashlib.sha256(identity.encode("utf-8")).digest(), "big")


def _has_overlapping_spans(turn: SgdUserTurn) -> bool:
    offsets = sorted((span.start, span.end) for span in turn.spans)
    return any(
        previous_end > start
        for (_, previous_end), (start, _) in zip(offsets, offsets[1:], strict=False)
    )


def audit_sgd_train(
    train_directory: Path, *, examples_per_slot: int = 50
) -> SgdContextAudit:
    """Audit SGD train USER contexts without assigning ITN labels."""
    if examples_per_slot < 0:
        raise ValueError("examples_per_slot must be non-negative")

    counts: Counter[tuple[str, str]] = Counter()
    descriptions: dict[tuple[str, str], str] = {}
    samples: dict[tuple[str, str], list[tuple[int, int, SgdAuditExample]]] = {}
    source_files: set[str] = set()
    dialogue_ids: set[str] = set()
    user_turns = 0
    context_only_user_turns = 0
    overlapping_span_turns = 0
    sequence = 0

    for turn in iter_sgd_train_user_turns(train_directory):
        user_turns += 1
        source_files.add(turn.source_file)
        dialogue_ids.add(turn.dialogue_id)
        if not turn.spans:
            context_only_user_turns += 1
        if _has_overlapping_spans(turn):
            overlapping_span_turns += 1

        for span in turn.spans:
            key = (span.service, span.slot)
            counts[key] += 1
            descriptions[key] = span.description
            if examples_per_slot == 0:
                continue

            example = SgdAuditExample(
                source_file=turn.source_file,
                dialogue_id=turn.dialogue_id,
                turn_index=turn.turn_index,
                utterance=turn.utterance,
                start=span.start,
                end=span.end,
                value=span.value,
            )
            priority = _sample_priority(turn, span)
            candidate = (-priority, sequence, example)
            reservoir = samples.setdefault(key, [])
            if len(reservoir) < examples_per_slot:
                heapq.heappush(reservoir, candidate)
            elif priority < -reservoir[0][0]:
                heapq.heapreplace(reservoir, candidate)
            sequence += 1

    slot_audits = tuple(
        SgdSlotAudit(
            service=service,
            slot=slot,
            description=descriptions[(service, slot)],
            count=count,
            examples=tuple(
                candidate[2]
                for candidate in sorted(
                    samples.get((service, slot), ()),
                    key=lambda candidate: (-candidate[0], candidate[1]),
                )
            ),
        )
        for (service, slot), count in sorted(
            counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1])
        )
    )
    return SgdContextAudit(
        kind="sgd_context_audit",
        schema_version=1,
        source_split="train",
        dialogue_files=len(source_files),
        dialogues=len(dialogue_ids),
        user_turns=user_turns,
        context_only_user_turns=context_only_user_turns,
        non_categorical_slot_spans=sum(counts.values()),
        overlapping_span_turns=overlapping_span_turns,
        service_slot_pairs=slot_audits,
    )


def write_sgd_context_audit(
    train_directory: Path,
    output_path: Path,
    *,
    examples_per_slot: int = 50,
) -> SgdContextAudit:
    """Audit SGD train and atomically write a new report."""
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {output_path}")

    report = audit_sgd_train(train_directory, examples_per_slot=examples_per_slot)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(asdict(report), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, output_path)
        temporary_path.unlink()
        return report
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit non-categorical USER slot contexts in SGD train."
    )
    parser.add_argument("train_directory", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--examples-per-slot", type=int, default=50)
    args = parser.parse_args()

    report = write_sgd_context_audit(
        args.train_directory,
        args.output_path,
        examples_per_slot=args.examples_per_slot,
    )
    summary = {
        "dialogue_files": report.dialogue_files,
        "dialogues": report.dialogues,
        "user_turns": report.user_turns,
        "context_only_user_turns": report.context_only_user_turns,
        "non_categorical_slot_spans": report.non_categorical_slot_spans,
        "overlapping_span_turns": report.overlapping_span_turns,
        "service_slot_pairs": len(report.service_slot_pairs),
        "output_path": str(args.output_path),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
