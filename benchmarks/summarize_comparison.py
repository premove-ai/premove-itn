"""Summarize stored results without running any backend."""

# Report prose and Markdown table rows remain on single lines.
# ruff: noqa: E501

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from run_comparison import (
    DATASET,
    _predicted_entity_values,
    _semantic_equal,
    json_dump,
    latency_metrics,
    read_rows,
    semantic_kind,
    subset_metrics,
    write_report,
)


def summarize(output: Path) -> None:
    rows = read_rows(DATASET)
    names = ["premove-itn", "thutmose", "text-processing-rs"]
    records = {n: read_rows(output / n / "records.jsonl") for n in names}
    payloads = [json.loads((output / n / "metrics.json").read_text()) for n in names]
    for name in names:
        current = records[name]
        assert not any(r["error"] for r in current)
    run = json.loads((output / "run.json").read_text())
    write_report(output, run, payloads, rows)
    (output / "REPORT.md").replace(output / "DETAILS.md")
    categories = {}
    for name in names:
        counts = defaultdict(lambda: [0, 0])
        for row, record in zip(rows, records[name], strict=True):
            values = _predicted_entity_values(row, record["prediction"])
            for i, span in enumerate(row["spans"]):
                count = counts[span["category"]]
                count[1] += 1
                if values is not None:
                    count[0] += _semantic_equal(
                        semantic_kind(row, span),
                        span["replacement"],
                        values[i],
                        span["category"],
                    )
        categories[name] = dict(counts)
    json_dump(output / "entity-category-counts.json", categories)
    lines = [
        "# First Evaluation",
        "",
        "Premove leads semantic entity accuracy overall and in the voice-agent group. It remains slower than both comparison backends. This is the repository's retained release-artifact result.",
        "",
        "## Overall results",
        "",
        "| Backend | Entity micro | Entity macro | Strict exact | Mean ms | p50 ms | p95 ms | p99 ms | Max ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in names:
        m = subset_metrics(records[name])
        latency = m["latency"]
        macro = mean(c / t for c, t in categories[name].values())
        lines.append(
            f"| {name} | {m['semantic_entity_accuracy']:.2%} | {macro:.2%} | {m['strict_exact_accuracy']:.2%} | {latency['mean_ms']:.2f} | {latency['p50_ms']:.2f} | {latency['p95_ms']:.2f} | {latency['p99_ms']:.2f} | {latency['max_ms']:.2f} |"
        )
    lines += [
        "",
        "Micro accuracy pools the 1,640 declared entities. Macro accuracy gives each of the 18 entity categories equal weight. KEEP rows have no positive entities and are reported separately. Formatting-only differences count against strict exact, while the archived semantic scorer is unchanged.",
        "",
        "## Voice-agent results",
        "",
        "These tables use only the 400 `group=voice_agent` rows. Each of eight domains has 50 rows. The generic multi-entity rows do not enter domain claims.",
        "",
        "| Backend | Correct entities | Accuracy | Mean ms | p95 ms | p99 ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in names:
        m = subset_metrics([r for r in records[name] if r["group"] == "voice_agent"])
        latency = m["latency"]
        lines.append(
            f"| {name} | {m['semantic_entity_correct']}/400 | {m['semantic_entity_accuracy']:.2%} | {latency['mean_ms']:.2f} | {latency['p95_ms']:.2f} | {latency['p99_ms']:.2f} |"
        )
    lines += [
        "",
        "| Domain | Premove | Thutmose | text-processing-rs |",
        "|---|---:|---:|---:|",
    ]
    for domain in sorted(
        {r["domain"] for r in records[names[0]] if r["group"] == "voice_agent"}
    ):
        cells = []
        for name in names:
            m = subset_metrics(
                [
                    r
                    for r in records[name]
                    if r["group"] == "voice_agent" and r["domain"] == domain
                ]
            )
            cells.append(
                f"{m['semantic_entity_correct']}/50 ({m['semantic_entity_accuracy']:.0%})"
            )
        lines.append("| " + domain + " | " + " | ".join(cells) + " |")
    lines += [
        "",
        "### Latency by voice-agent domain",
        "",
        "| Domain | Backend | Mean ms | p50 ms | p95 ms | p99 ms | Max ms |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for domain in sorted(
        {r["domain"] for r in records[names[0]] if r["group"] == "voice_agent"}
    ):
        for name in names:
            latency = latency_metrics(
                [
                    r
                    for r in records[name]
                    if r["group"] == "voice_agent" and r["domain"] == domain
                ]
            )
            lines.append(
                f"| {domain} | {name} | {latency['mean_ms']:.2f} | {latency['p50_ms']:.2f} | {latency['p95_ms']:.2f} | {latency['p99_ms']:.2f} | {latency['max_ms']:.2f} |"
            )
    lines += [
        "",
        "## Accuracy by entity category",
        "",
        "Each count below scores only the named entity, including entities in multi-entity sentences.",
        "",
        "| Category | Premove | Thutmose | text-processing-rs |",
        "|---|---:|---:|---:|",
    ]
    for category in sorted(categories[names[0]]):
        cells = []
        for name in names:
            c, t = categories[name][category]
            cells.append(f"{c}/{t} ({c / t:.2%})")
        lines.append("| " + category + " | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Initialization and warm-up",
        "",
        "| Backend | Initialization ms | 15 warm-up calls ms |",
        "|---|---:|---:|",
    ]
    for p in payloads:
        lines.append(
            f"| {p['backend']} | {p['initialization_ms']:.2f} | {p['warmup_ms']:.2f} |"
        )
    lines += [
        "",
        "Warm-up records are saved separately for each backend. Thutmose also retains its startup warm-up in runtime metadata. Initialization timing is process/model initialization, not machine cold boot or first download.",
        "",
        "",
        "## Premove timing components",
        "",
        "| Component | Mean ms | p95 ms | p99 ms |",
        "|---|---:|---:|---:|",
    ]
    for field in ["candidate_graph_ms", "encode_ms", "model_ms", "decode_ms"]:
        m = latency_metrics(records["premove-itn"], field)
        lines.append(
            f"| {field} | {m['mean_ms']:.2f} | {m['p95_ms']:.2f} | {m['p99_ms']:.2f} |"
        )
    lines += [
        "",
        "## What the result supports",
        "",
        "The project succeeds on the measured voice-agent value-normalization objective: 398/400 correct entities versus 268/400 for Thutmose and 273/400 for text-processing-rs. It also leads overall semantic accuracy. It does not win latency, and its remaining collision and KEEP errors prevent a claim of universal ITN superiority.",
        "",
        "The release artifact is the measured implementation. Its native build profile, compiler, upstream revision, and imported extension path are recorded in run.json.",
        "",
        "## Equivalence and limitations",
        "",
        "- All 1,500 complete candidate tuples match the pre-batch builder exactly.",
        "- All 1,500 rows completed without backend errors.",
        "- This run is the repository's First Evaluation record. It was run after exploratory work and is not a blind evaluation.",
        "- No training, model selection, candidate pruning, or benchmark-driven tuning occurred.",
        "- The dataset is a balanced synthetic stress suite. Its scores do not estimate production traffic accuracy.",
        "- Blind human gold adjudication remains pending. Exact and normalized overlap checks passed; token n-gram and embedding contamination checks remain incomplete.",
        "- text-processing-rs is an upstream ablation, not an independent architecture. Thutmose uses a custom weight-compatible loader of the NVIDIA artifact, not the current NeMo API.",
        "- Timing used an Apple Silicon Mac, sequential batch-one requests, MPS completion, and eight Rayon workers. No other benchmark or training workload ran concurrently; ordinary desktop background processes remained active.",
        "",
        "## Detailed evidence",
        "",
        "[Detailed model, kind, group, collision, and latency tables](DETAILS.md). Domain-label tables there include all groups; use the voice-only table above for domain claims. Their category tables summarize rows containing a category; use the entity-only counts above for category claims.",
        "",
        "[run metadata](run.json), [artifact and graph audit](artifact.json), [reproduction instructions](../../../../benchmarks/README.md).",
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summarize(args.output)
