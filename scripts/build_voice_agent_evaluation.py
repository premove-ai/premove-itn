"""Build the frozen main VoiceAgent ITN evaluation dataset."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from premove_itn.labels import SPAN_KINDS

try:
    from scripts import _voice_agent_evaluation_source as source
except ModuleNotFoundError:  # Direct script execution adds scripts/ to sys.path.
    import _voice_agent_evaluation_source as source

ROOT = Path(__file__).resolve().parents[1]
# The seed and row IDs retain their audited V6 values to preserve the frozen hash.
OUTPUT = ROOT / "eval/voice_agent_itn"
SEED = "premove-itn/voice-agent-itn/v6"
VALID_PREMOVE_KINDS = {kind.value for kind in SPAN_KINDS}
FROZEN_ARTIFACT_SHA256 = (
    "782b14d0291e4e176e4ba22d0d756019e906a5ea4c7d267d9703e80b082cbd61"
)
FROZEN_AT = "2026-09-08T00:00:00+05:30"
INDEPENDENT_AUDIT_SHA256 = (
    "9ae41656f6e3af761658cfc5645b40b8cc8846802d10ec74bf226aa806f4a933"
)


def _date_identifier_rows() -> list[dict[str, object]]:
    """Replace the single-payload date/modal family with 25 real contrasts."""
    result = []
    months = ["may", "march", "june", "april", "august"]
    old_pair_numbers = range(3, 175, 7)
    for case_index, pair_number in enumerate(old_pair_numbers):
        month = months[case_index % len(months)]
        day = case_index // len(months) + 1
        spoken = f"{month} {source.ordinal(day)}"
        written_date = f"{month.title()} {day}"
        written_id = f"{month.upper()}{day}"
        pair_id = f"collision_pair_{pair_number:03d}"
        variants = [
            (
                "DATE",
                written_date,
                {
                    "month": source.MONTHS.index(month.title()) + 1,
                    "day": day,
                    "year": None,
                },
                "DATE",
                "the filing deadline is {x}",
            ),
            (
                "REFERENCE_ID",
                written_id,
                written_id,
                "WORD",
                "the archive reference is {x}",
            ),
        ]
        for category, written, semantic_value, kind, template in variants:
            result.append(
                source.row(
                    f"va6_collision_date_{len(result) + 1:03d}",
                    "collision",
                    template,
                    [
                        {
                            "category": category,
                            "spoken": spoken,
                            "written": written,
                            "semantic_value": semantic_value,
                            "premove_span_kind": kind,
                        }
                    ],
                    "general",
                    "hard",
                    "date_vs_identifier",
                    pair_id,
                )
            )
    return result


def _insert_before_span(row: dict[str, object], old: str, new: str) -> None:
    text = str(row["text"])
    expected = str(row["expected_text"])
    if old not in text or old not in expected:
        raise ValueError(f"missing context to repair: {row['id']}")
    delta = len(new) - len(old)
    row["text"] = text.replace(old, new, 1)
    row["expected_text"] = expected.replace(old, new, 1)
    for span in row["spans"]:
        span["start"] += delta
        span["end"] += delta


def build_rows() -> list[dict[str, object]]:
    rows = [r for r in source.build_rows() if r["collision"] != "date_vs_modal"]
    rows.extend(_date_identifier_rows())
    for row in rows:
        row["id"] = str(row["id"]).replace("source_", "va6_", 1)
        if row["collision"] == "money_vs_time" and row["categories"] == ["MONEY"]:
            _insert_before_span(
                row, "the cash price was ", "the cash price in dollars was "
            )
        if row["collision"] == "identifier_vs_time" and row["categories"] == [
            "ORDER_ID"
        ]:
            row["categories"] = ["REFERENCE_ID"]
            row["spans"][0]["category"] = "REFERENCE_ID"
        for span in row["spans"]:
            category = span["category"]
            if category == "URL":
                span["premove_span_kind"] = "ELECTRONIC"
            elif category == "PERCENT":
                span["premove_span_kind"] = None
            if category == "PHONE":
                span["semantic_value"] = {
                    "digits": span["semantic_value"]["digits"],
                    "international": str(span["replacement"]).startswith("+"),
                }
            if category in {"ORDER_ID", "REFERENCE_ID", "FLIGHT_ID"}:
                span["semantic_comparison"] = "case_insensitive"
    return rows


def validate(rows: list[dict[str, object]]) -> dict[str, object]:
    report = source.validate(rows)
    category_kinds: dict[str, set[str | None]] = {}
    payloads: dict[str, set[str]] = {}
    for row in rows:
        if row["collision_pair_id"] and row["spans"]:
            payloads.setdefault(str(row["collision"]), set()).add(
                str(row["spans"][0]["source"])
            )
        for span in row["spans"]:
            kind = span["premove_span_kind"]
            if kind is not None and kind not in VALID_PREMOVE_KINDS:
                raise ValueError(f"invalid premove_span_kind: {row['id']}:{kind}")
            category_kinds.setdefault(str(span["category"]), set()).add(kind)
            if span["category"] == "PHONE" and set(span["semantic_value"]) != {
                "digits",
                "international",
            }:
                raise ValueError(f"incomplete phone semantics: {row['id']}")
    ambiguous = {
        category: kinds for category, kinds in category_kinds.items() if len(kinds) > 1
    }
    if ambiguous:
        raise ValueError(f"category has multiple Premove mappings: {ambiguous}")
    if (
        payloads.get("date_vs_identifier") is None
        or len(payloads["date_vs_identifier"]) != 25
    ):
        raise ValueError("date collision payload diversity failure")
    if any(row["collision"] == "date_vs_modal" for row in rows):
        raise ValueError("obsolete date collision family remains")
    report.update(
        {
            "invalid_premove_span_kinds": 0,
            "category_to_premove_mapping_conflicts": 0,
            "percent_premove_mapping": None,
            "url_premove_mapping": "ELECTRONIC",
            "implicit_currency_collision_rows": 0,
            "room_codes_labeled_order_id": 0,
            "international_phone_semantics_missing": 0,
            "identifier_semantic_comparison": "case_insensitive",
            "date_collision_unique_payloads": 25,
        }
    )
    return report


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_existing(output: Path = OUTPUT) -> dict[str, object]:
    """Freeze an already audited artifact without rewriting its JSONL content."""
    data = output / "voice_agent_eval.jsonl"
    manifest_path = output / "manifest.json"
    artifact_sha256 = sha(data)
    if artifact_sha256 != FROZEN_ARTIFACT_SHA256:
        raise ValueError(
            "refusing to freeze unexpected artifact: "
            f"expected {FROZEN_ARTIFACT_SHA256}, got {artifact_sha256}"
        )
    rows = [json.loads(line) for line in data.read_text().splitlines()]
    validate(rows)
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("review_status", None)
    manifest.update(
        {
            "name": "VoiceAgent-ITN",
            "content_status": "frozen",
            "frozen_at": FROZEN_AT,
            "freeze_policy": (
                "no content changes after independent full-dataset audit"
            ),
            "independent_audit": {
                "scope": "all_1500_rows",
                "verdict": "no_dataset_blockers",
                "sha256": INDEPENDENT_AUDIT_SHA256,
            },
            "gold_review_status": "pending_blind_human_adjudication",
            "contamination_audit_status": "pending",
        }
    )
    manifest["artifact"]["sha256"] = artifact_sha256
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def build(output: Path = OUTPUT) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen benchmark: {output}")
    rows = build_rows()
    checks = validate(rows)
    rows.sort(key=lambda r: hashlib.sha256(f"{SEED}\0{r['id']}".encode()).hexdigest())
    output.mkdir(parents=True)
    data = output / "voice_agent_eval.jsonl"
    data.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    artifact_sha256 = sha(data)
    if artifact_sha256 != FROZEN_ARTIFACT_SHA256:
        raise ValueError(
            "frozen artifact drift: "
            f"expected {FROZEN_ARTIFACT_SHA256}, got {artifact_sha256}"
        )
    manifest = {
        "schema_version": 3,
        "name": "VoiceAgent-ITN",
        "created_at": datetime.now(UTC).isoformat(),
        "content_status": "frozen",
        "frozen_at": FROZEN_AT,
        "freeze_policy": "no content changes after independent full-dataset audit",
        "independent_audit": {
            "scope": "all_1500_rows",
            "verdict": "no_dataset_blockers",
            "sha256": INDEPENDENT_AUDIT_SHA256,
        },
        "rows": 1500,
        "quotas": source.QUOTAS,
        "selection_basis": "backend-neutral voice-agent requirements",
        "evaluated_backends": ["Premove ITN", "Thutmose", "text-processing-rs"],
        "backend_outputs_inspected_during_authoring": False,
        "model_frozen_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "frozen_checkpoint": (
            "data/models/structured_value_selected_20k/checkpoint.pt"
        ),
        "frozen_checkpoint_sha256": source.FROZEN_CHECKPOINT_SHA256,
        "used_for_training": False,
        "used_for_model_selection": False,
        "first_model_run_at": None,
        "permitted_inference_input_fields": ["text"],
        "semantic_comparison_contract": {
            "identifiers": "case_insensitive_when_speech_does_not_encode_case",
            "phone": "preserve_digits_and_international_dialing_form",
        },
        "metric_hierarchy": {
            "primary": "macro_semantic_entity_accuracy",
            "secondary": "strict_sentence_exact_match",
            "collision": "pair_accuracy",
            "multi": ["span_accuracy", "all_entities_correct"],
            "keep": {
                "standalone_false_normalization_rate": {
                    "group": "negative",
                    "rows": 50,
                },
                "all_keep_preservation_accuracy": {"no_span": True, "rows": 100},
            },
            "voice_agent": {
                "metric": "macro_domain_accuracy",
                "row_filter": {"group": "voice_agent"},
                "excluded_groups": ["multi"],
            },
            "additional": "micro_semantic_entity_accuracy",
            "category_aggregates": [
                "standard_itn_core",
                "voice_agent_structured_extension",
            ],
        },
        "gold_review_status": "pending_blind_human_adjudication",
        "contamination_audit_status": "pending",
        "required_pre_run_gates": [
            "independent_blind_gold_adjudication",
            "exact_normalized_ngram_and_embedding_overlap_review",
            "freeze_dataset_and_backend_commits",
        ],
        "validation": checks,
        "difficulty_distribution": dict(Counter(r["difficulty"] for r in rows)),
        "category_distribution": dict(
            Counter(c for r in rows for c in r["categories"])
        ),
        "artifact": {"path": data.name, "sha256": artifact_sha256},
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, sort_keys=True))
